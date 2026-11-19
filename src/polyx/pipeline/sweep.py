"""Optuna sweep over the shared training protocol.

Hydra composes the sweep config (``conf/sweep.yaml``) and the Optuna sweeper
plugin consumes the same ``hydra.sweeper`` node this module reads. The driver
runs the trials in one process so the tiny-fixture smoke run is testable through
the same headless command as a single execution; real multiruns use the plugin
and its launcher. Both paths call :func:`run_execution`, so the objective is
always the shared evaluator's with-class score and never a family-specific
figure.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import hydra_plugins.hydra_optuna_sweeper.config  # noqa: F401 - registers the config groups
import optuna
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from polyx.pipeline.artifacts import ArtifactLayout, write_json, write_yaml
from polyx.pipeline.config import CONF_DIR, ExecutionConfig, execution_config_from_raw
from polyx.pipeline.execution import run_execution
from polyx.pipeline.sweep_record import (
    acquire_trial_lease,
    completed_score,
    read_trial_record,
    release_trial_lease,
    trial_directory,
    write_trial_record,
)

#: Protocol hyperparameters a trial may change. Split, seed, augmentation
#: subset and the evaluator are outside this set and cannot be suggested.
SEARCHABLE_PROTOCOL_FIELDS = frozenset(
    {
        "epochs",
        "batch",
        "image_size",
        "optimizer",
        "learning_rate",
        "momentum",
        "weight_decay",
        "schedule",
        "warmup_epochs",
    }
)


@dataclass(frozen=True)
class SweepConfig:
    """The resolved sweep: one execution config plus the sweeper node."""

    execution: ExecutionConfig
    direction: str
    study_name: str
    storage: str
    n_trials: int
    n_jobs: int
    sampler_target: str
    sampler_seed: int
    search_space: dict[str, dict[str, Any]]
    pruner: str


def _study_url(storage: Any, study_name: str, output_dir: Path) -> str:
    """SQLite URL on the work filesystem, never a personal path."""
    if storage:
        return str(storage)
    from polyx.hpc.layout import ClusterLayout

    if os.environ.get("POLYX_HPC_WORK"):
        path = ClusterLayout.from_env().study_path(study_name)
    else:
        path = output_dir / "studies" / f"{study_name}.db"
    return f"sqlite:///{path}"


def _yaml_scalar(text: str) -> Any:
    """One Hydra override value as the Python scalar it denotes."""
    text = text.strip()
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _declaration(key: str, value: Any) -> dict[str, Any]:
    """Translate one ``hydra.sweeper.params`` entry into a trial declaration.

    The plugin consumes the same string using Hydra's override syntax
    (``choice(...)``, ``interval(...)``, ``range(...)``); parsing it here keeps
    the one-process driver and the real multirun on one declared space.
    """
    text = str(value).strip()
    kind, _, argument = text.partition("(")
    if not argument.endswith(")"):
        raise ValueError(f"unsupported search-space declaration for {key}: {text}")
    parts = [part.strip() for part in argument[:-1].split(",") if part.strip()]
    kind = kind.strip()
    if kind == "choice":
        return {"type": "categorical", "choices": [_yaml_scalar(part) for part in parts]}
    if kind in {"interval", "range"} and len(parts) >= 2:
        low, high = _yaml_scalar(parts[0]), _yaml_scalar(parts[1])
        declaration: dict[str, Any] = {"type": "int" if kind == "range" else "float"}
        declaration["low"], declaration["high"] = low, high
        if len(parts) > 2:
            if parts[2].lower() in {"log", "log=True"}:
                declaration["log"] = True
            else:
                declaration["step"] = _yaml_scalar(parts[2])
        return declaration
    raise ValueError(f"unsupported search-space declaration for {key}: {text}")


def _search_space(sweeper: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The declared protocol-only space, from `params` or a legacy `search_space`."""
    params = sweeper.get("params")
    if params:
        space = {str(key): _declaration(str(key), value) for key, value in params.items()}
    else:
        space = dict(sweeper.get("search_space") or {})
    for key in space:
        parts = key.split(".")
        if len(parts) != 2 or parts[0] != "protocol" or parts[1] not in SEARCHABLE_PROTOCOL_FIELDS:
            raise ValueError(f"search space may only declare protocol hyperparameters: {key}")
    return space


def load_sweep_config(overrides: list[str] | None = None) -> SweepConfig:
    """Compose ``sweep.yaml`` with Hydra and validate the declared space."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        composed = compose(
            config_name="sweep",
            overrides=list(overrides or []),
            return_hydra_config=True,
        )
    task = OmegaConf.to_container(composed, resolve=False)
    assert isinstance(task, dict)
    task.pop("hydra", None)
    sweeper = OmegaConf.to_container(composed.hydra.sweeper, resolve=True)
    assert isinstance(sweeper, dict)

    execution = execution_config_from_raw(task)
    space = _search_space(sweeper)

    sampler = sweeper.get("sampler") or {}
    return SweepConfig(
        execution=execution,
        direction=str(sweeper["direction"]),
        study_name=str(sweeper["study_name"]),
        storage=_study_url(
            sweeper.get("storage"), str(sweeper["study_name"]), execution.output_dir
        ),
        n_trials=int(sweeper["n_trials"]),
        n_jobs=int(sweeper["n_jobs"]),
        sampler_target=str(sampler["_target_"]),
        sampler_seed=int(sampler["seed"]),
        search_space=space,
        pruner="none" if not sweeper.get("pruner") else str(sweeper["pruner"]),
    )


def _suggest(trial: optuna.trial.Trial, name: str, declaration: dict[str, Any]) -> Any:
    """Suggest one declared value with Optuna's public trial API."""
    kind = str(declaration["type"])
    if kind == "categorical":
        return trial.suggest_categorical(name, list(declaration["choices"]))
    if kind in {"int", "float"}:
        low, high = declaration["low"], declaration["high"]
        log = bool(declaration.get("log", False))
        step = declaration.get("step")
        if kind == "int":
            return trial.suggest_int(name, int(low), int(high), step=int(step or 1), log=log)
        if step is not None:
            return trial.suggest_float(name, float(low), float(high), step=float(step), log=log)
        return trial.suggest_float(name, float(low), float(high), log=log)
    raise ValueError(f"unsupported search-space declaration for {name}: {kind}")


def _trial_config(
    base: ExecutionConfig, params: dict[str, Any], output_dir: Path, *, resume: bool = False
) -> ExecutionConfig:
    """One trial changes only declared protocol values and its output folder."""
    protocol = deepcopy(base.protocol)
    for key, value in params.items():
        setattr(protocol, key.split(".", 1)[1], value)
    return replace(base, protocol=protocol, output_dir=output_dir, resume=resume)


def _sweep_config_view(config: SweepConfig) -> dict[str, Any]:
    """Plain, YAML-serialisable record of the configuration that produced a study."""
    return {
        "direction": config.direction,
        "study_name": config.study_name,
        "storage": config.storage,
        "n_trials": config.n_trials,
        "n_jobs": config.n_jobs,
        "sampler": {"target": config.sampler_target, "seed": config.sampler_seed},
        "pruner": config.pruner,
        "search_space": config.search_space,
        "data_yaml": (
            str(config.execution.data_yaml) if config.execution.data_yaml is not None else ""
        ),
        "coco_dataset": (
            str(config.execution.coco_dataset) if config.execution.coco_dataset else ""
        ),
        "family": config.execution.family.name,
        "backend": config.execution.backend.name,
        "fixed": {
            "split": config.execution.protocol.split,
            "seed": config.execution.protocol.seed,
            "augmentation_subset": config.execution.protocol.augmentation_subset,
            "evaluator": "polyx.pipeline.evaluation.SharedEvaluator",
            "objective": "with_class_score",
        },
    }


def run_sweep(config: SweepConfig) -> dict[str, Any]:
    """Run the declared trials, reusing the SQLite study, and save the best."""
    layout = ArtifactLayout(config.execution.output_dir)
    layout.root.mkdir(parents=True, exist_ok=True)
    database = Path(config.storage.removeprefix("sqlite:///"))
    database.parent.mkdir(parents=True, exist_ok=True)
    reused = database.is_file()

    sampler = instantiate({"_target_": config.sampler_target, "seed": config.sampler_seed})
    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage,
        direction=config.direction,
        sampler=sampler,
        # Hydra trials run as separate jobs: no Optuna pruning between them.
        # Lightning's early stopping handles stopping inside each trial.
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    write_yaml(_sweep_config_view(config), layout.sweep_config)

    recovered = _recover_pending(config, study, layout.root)
    for _index in range(max(0, config.n_trials - recovered)):
        trial = study.ask()
        suggestions = {
            name: _suggest(trial, name, rule) for name, rule in config.search_space.items()
        }
        trial_dir = trial_directory(layout.root, trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "trial_id": trial.number,
            "parameters": suggestions,
            "execution": trial_dir,
            "study_name": config.study_name,
            "storage": config.storage,
        }
        write_trial_record(**record)
        resume = (trial_dir / "progress.json").is_file()
        try:
            score = float(
                run_execution(
                    _trial_config(config.execution, suggestions, trial_dir, resume=resume)
                )["selection"]["value"]
            )
        except Exception:
            write_trial_record(**record, status="failed")
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            raise
        write_trial_record(**record, status="completed")
        study.tell(trial.number, score)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("the study has no completed trial")
    best = (
        max(completed, key=lambda trial: trial.value)
        if config.direction == "maximize"
        else min(completed, key=lambda trial: trial.value)
    )
    best_metrics = layout.root / "trials" / f"trial_{best.number:03d}" / "metrics.json"
    selection = (
        _read_selection(best_metrics, best.value)
        if best_metrics.is_file()
        else {"metric": "with_class_score", "value": best.value}
    )
    artifact = {
        "study_name": config.study_name,
        "storage": config.storage,
        "direction": config.direction,
        "metric": "with_class_score",
        "criterion": selection.get("criterion", ""),
        "best_trial": best.number,
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "trials_run": config.n_trials,
        "total_trials": len(study.trials),
        "reused_study": reused,
        "pruner": config.pruner,
    }
    write_json(artifact, layout.sweep_best)
    return {
        "output_dir": layout.root,
        "selection": selection,
        "study_name": config.study_name,
        "storage": config.storage,
        "trials_run": config.n_trials,
        "total_trials": len(study.trials),
        "reused_study": reused,
        "best_params": dict(best.params),
        "best_value": float(best.value),
        "best_trial": best.number,
        "metric": "with_class_score",
    }


def _recover_pending(config: SweepConfig, study: Any, root: Path) -> int:
    """Finish trials a previous invocation left ``RUNNING``.

    The in-process driver owns ``study.ask()``, so it can recover the same
    trial: run the recorded parameters in the recorded directory (resuming its
    checkpoint when one exists), then tell the score to the *existing* trial
    number. A trial whose execution already finished is told from its metrics
    instead of being retrained.
    """
    pending: list[int] = []
    trials_dir = root / "trials"
    if trials_dir.is_dir():
        for directory in sorted(trials_dir.glob("trial_*")):
            record = read_trial_record(directory)
            if record is None or record.get("status") != "running":
                continue
            try:
                pending.append(int(record["trial_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    running = [
        int(trial.number)
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    recovered = 0
    for trial_id in sorted({*pending, *running}):
        trial = study.trials[trial_id] if 0 <= trial_id < len(study.trials) else None
        if trial is None or trial.state != optuna.trial.TrialState.RUNNING:
            continue
        directory = trial_directory(root, trial_id)
        lease = acquire_trial_lease(directory)
        if lease is None:
            # Another runner already recovered this trial.
            continue
        try:
            score = (
                completed_score(directory) if (directory / "completion.json").is_file() else None
            )
            if score is not None:
                study.tell(trial_id, score)
                recovered += 1
                continue
            record = read_trial_record(directory)
            parameters = dict(record.get("parameters") or {}) if record else dict(trial.params)
            if not parameters:
                continue
            resume = (directory / "progress.json").is_file()
            try:
                score = float(
                    run_execution(
                        _trial_config(config.execution, parameters, directory, resume=resume)
                    )["selection"]["value"]
                )
            except Exception:
                study.tell(trial_id, state=optuna.trial.TrialState.FAIL)
                raise
            recovered += 1
            study.tell(trial_id, score)
        finally:
            release_trial_lease(lease)
    return recovered


def _read_selection(path: Path, fallback: float) -> dict[str, Any]:
    """The shared evaluator's selection record from a trial's metrics file."""
    data = _read_json(path)
    selection = data.get("selection") if isinstance(data, dict) else None
    if isinstance(selection, dict):
        return selection
    return {"metric": "with_class_score", "value": fallback}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
