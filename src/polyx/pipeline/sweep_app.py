"""Hydra objective for the real Optuna sweeper multirun.

``python -m polyx.pipeline.sweep_app --multirun`` hands each configuration here
and records the returned number as the study value. It is the shared evaluator's
with-class score from :func:`run_execution`, the same number the one-process
driver returns through ``polyx-pipeline --sweep``.

Each trial writes a normal execution under ``<output_dir>/trials/trial_<N>``.
The number comes from the study trial, not from Hydra's job index: a resumed
multirun starts its job index at zero again, so the job index would overwrite
the previous invocation's first trial directory. A trial interrupted by a
time limit is left ``RUNNING`` in Optuna; matching the sampled parameters
against the running trials identifies the new one even when an old zombie is
still in the study. A trial index records every directory that was written so
the best-artifact step can find the winner unambiguously. One trial at a time
is the configured contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import optuna
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from polyx.pipeline.artifacts import write_json_atomic


@main(config_path="conf", config_name="sweep", version_base="1.3")
def objective(cfg: DictConfig) -> float:
    """Run one suggested trial and return only the with-class score."""
    from polyx.pipeline.config import execution_config_from_raw
    from polyx.pipeline.execution import run_execution

    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    output = _trial_output_dir(raw)
    raw["output_dir"] = str(output)
    summary = run_execution(execution_config_from_raw(raw))
    _register_trial(output, _trial_number(raw))
    return float(summary["selection"]["value"])


def _index_path(root: Path) -> Path:
    return root / "trials" / "index.json"


def _read_index(root: Path) -> dict[str, list[str]]:
    path = _index_path(root)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {
        str(key): [str(name) for name in value]
        for key, value in loaded.items()
        if isinstance(value, list)
    }


def _register_trial(output: Path, number: int | None) -> None:
    """Record this execution's directory against its study trial number."""
    root = output.parent.parent
    index = _read_index(root)
    key = str(number) if number is not None else f"job_{HydraConfig.get().job.num}"
    entries = index.setdefault(key, [])
    name = output.name
    if name not in entries:
        entries.append(name)
    write_json_atomic(index, _index_path(root))


def _numbers_match(observed: Any, expected: Any) -> bool:
    """Trial parameters round-trip through Hydra; compare numbers by value."""
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    return observed == expected


def _sampled_params(raw: dict) -> dict[str, Any]:
    """The protocol values this trial's declared space could have suggested."""
    sweeper = HydraConfig.get().sweeper
    declared = sweeper.get("params") or {}
    protocol = raw.get("protocol") or {}
    if not isinstance(protocol, dict):
        return {}
    sampled: dict[str, Any] = {}
    for key in declared:
        name = str(key)
        if not name.startswith("protocol."):
            continue
        field = name.split(".", 1)[1]
        if field in protocol:
            sampled[name] = protocol[field]
    return sampled


def _trial_number(raw: dict) -> int | None:
    """The study trial this job is executing, not Hydra's resettable job id."""
    sweeper = HydraConfig.get().sweeper
    storage = sweeper.get("storage")
    study_name = sweeper.get("study_name")
    if not storage or not study_name:
        return None
    try:
        study = optuna.load_study(study_name=str(study_name), storage=str(storage))
    except Exception:  # noqa: BLE001 - a fresh study may not exist yet
        return None
    running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    if not running:
        return None
    sampled = _sampled_params(raw)
    matching = [
        trial
        for trial in running
        if all(
            name in trial.params and _numbers_match(trial.params[name], value)
            for name, value in sampled.items()
        )
    ]
    # The current trial is the newest one, whether an interrupted zombie with
    # the same parameters exists or not.
    pool = matching or running
    return max(int(trial.number) for trial in pool)


def _trial_output_dir(raw: dict) -> Path:
    """The unique execution directory of this trial under the sweep root."""
    root = Path(str(raw["output_dir"]))
    job_index = int(HydraConfig.get().job.num)
    number = _trial_number(raw)
    label = f"trial_{number:03d}" if number is not None else f"job_{job_index:03d}"
    directory = root / "trials" / label
    index = _read_index(root)
    if directory.name in index.get(str(number) if number is not None else f"job_{job_index}", []):
        directory = root / "trials" / f"{label}_job{job_index:03d}"
    if (directory / "config.yaml").is_file():
        # Never overwrite an earlier trial's artifacts, even if the index was
        # lost or the job was restarted outside Hydra.
        directory = (
            root
            / "trials"
            / f"{label}_{job_index:03d}_{len(list(directory.parent.glob(label + '*')))}"
        )
    return directory


class PrepareStudyDirectory:
    """Create the SQLite study's parent before the sweeper opens it.

    The Optuna plugin opens the study as soon as the multirun starts, and
    SQLAlchemy will not create a missing directory. This Hydra callback runs
    between composition and sweeper setup.
    """

    def on_multirun_start(self, config: DictConfig, **_kwargs: object) -> None:
        """Make the directory that holds the study file."""
        storage = OmegaConf.select(config, "hydra.sweeper.storage")
        if not storage:
            return
        url = str(storage)
        if not url.startswith("sqlite:///"):
            return
        Path(url.removeprefix("sqlite:///")).expanduser().parent.mkdir(parents=True, exist_ok=True)

    # Hydra warns about a callback that does not expose every hook, so the
    # ones this step does not need are explicit no-ops.
    def on_run_start(self, config: DictConfig, **_kwargs: object) -> None:
        """No per-run work."""

    def on_run_end(self, config: DictConfig, **_kwargs: object) -> None:
        """No per-run work."""

    def on_multirun_end(self, config: DictConfig, **_kwargs: object) -> None:
        """No per-multirun work."""

    def on_job_start(self, config: DictConfig, **_kwargs: object) -> None:
        """No per-job work."""

    def on_job_end(self, config: DictConfig, **_kwargs: object) -> None:
        """No per-job work."""


if __name__ == "__main__":
    objective()
