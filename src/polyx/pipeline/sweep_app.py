"""Hydra objective for the real Optuna sweeper multirun.

``python -m polyx.pipeline.sweep_app --multirun`` hands each configuration here
and records the returned number as the study value. It is the shared evaluator's
with-class score from :func:`run_execution`, the same number the one-process
driver returns through ``polyx-pipeline --sweep``.

``PolyxOptunaSweeper`` injects the real study trial number as
``sweep_trial_id``. That is the only accepted identity: parameters and Hydra's
resettable job index are ambiguous after a preemption. Each trial persists a
record before training; a resubmission with the same trial id resumes the
recorded execution when it has a progress snapshot, or retries the same
parameters in a clean directory when it does not. The resolved study name and
storage are written once and reused by ``sweep_best``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from polyx.core.i18n import N_
from polyx.pipeline.artifacts import write_yaml
from polyx.pipeline.errors import PipelineError
from polyx.pipeline.optuna_sweeper import install_trial_id_passthrough
from polyx.pipeline.sweep_record import (
    acquire_trial_lease,
    release_trial_lease,
    trial_directory,
    write_trial_record,
)

install_trial_id_passthrough()


@main(config_path="conf", config_name="sweep", version_base="1.3")
def objective(cfg: DictConfig) -> float:
    """Run one suggested trial and return only the with-class score."""
    from polyx.pipeline.config import execution_config_from_raw
    from polyx.pipeline.execution import run_execution

    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    root = Path(str(raw["output_dir"]))
    trial_id = int(raw["sweep_trial_id"])
    study = _study_record()
    _persist_study(root, study)
    output = _trial_output_dir(root, trial_id)
    lease = acquire_trial_lease(output)
    if lease is None:
        # This trial is already being trained elsewhere; returning a value
        # here would score one runner's work twice.
        raise PipelineError(
            "sweep.trial_busy",
            N_("study trial %(trial)d is already running in another process"),
            trial=trial_id,
        )
    try:
        record = {
            "trial_id": trial_id,
            "parameters": _sampled_parameters(raw),
            "execution": output,
            "study_name": study["study_name"],
            "storage": study["storage"],
        }
        write_trial_record(**record)
        raw["output_dir"] = str(output)
        if (output / "progress.json").is_file():
            raw["resume"] = True
        summary = run_execution(execution_config_from_raw(raw))
        write_trial_record(**record, status="completed")
    finally:
        release_trial_lease(lease)
    return float(summary["selection"]["value"])


def _study_record() -> dict[str, Any]:
    """The study identity resolved by Hydra once for this sweep root."""
    sweeper = HydraConfig.get().sweeper
    return {
        "study_name": str(sweeper.get("study_name") or "polyx-optuna"),
        "storage": str(sweeper.get("storage") or ""),
        "direction": str(sweeper.get("direction") or "maximize"),
        "params": dict(sweeper.get("params") or {}),
    }


def _persist_study(root: Path, study: dict[str, Any]) -> None:
    """Write the study identity once; a different study in the same root fails."""
    path = root / "sweep_config.yaml"
    if path.is_file():
        import yaml

        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if (
            existing.get("study_name") != study["study_name"]
            or existing.get("storage") != study["storage"]
        ):
            raise RuntimeError(
                f"{root} already belongs to study {existing.get('study_name')!r} at "
                f"{existing.get('storage')!r}; use a new sweep root"
            )
        return
    write_yaml(study, path)


def _sampled_parameters(raw: dict) -> dict[str, Any]:
    """The protocol values this trial's declared search space suggested."""
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


def _trial_output_dir(root: Path, trial_id: int) -> Path:
    """The stable execution directory for one study trial id."""
    directory = trial_directory(root, trial_id)
    if (directory / "progress.json").is_file():
        return directory
    if directory.exists() and not (directory / "completion.json").is_file():
        # A record without a progress snapshot cannot be resumed; retrying the
        # same parameters in a clean directory is explicit, not an overwrite of
        # usable work. A *completed* execution is never deleted: the engine
        # refuses to reuse it, which is loud instead of destructive.
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
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
