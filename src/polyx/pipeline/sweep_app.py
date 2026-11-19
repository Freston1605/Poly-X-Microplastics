"""Hydra objective for the real Optuna sweeper multirun.

``python -m polyx.pipeline.sweep_app --multirun`` hands each configuration here
and records the returned number as the study value. It is the shared evaluator's
with-class score from :func:`run_execution`, the same number the one-process
driver returns through ``polyx-pipeline --sweep``.

Each trial writes a normal execution under ``<output_dir>/trials/trial_<N>``.
The number comes from the study's running trial, not from Hydra's job index:
a resumed multirun starts its job index at zero again, so the job index would
overwrite the previous invocation's first trial directory. One trial at a time
is the configured contract; if more than one is running, the job index is the
fallback.
"""

from __future__ import annotations

from pathlib import Path

import optuna
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@main(config_path="conf", config_name="sweep", version_base="1.3")
def objective(cfg: DictConfig) -> float:
    """Run one suggested trial and return only the with-class score."""
    from polyx.pipeline.config import execution_config_from_raw
    from polyx.pipeline.execution import run_execution

    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    raw["output_dir"] = str(_trial_output_dir(raw))
    return float(run_execution(execution_config_from_raw(raw))["selection"]["value"])


def _trial_output_dir(raw: dict) -> Path:
    """The unique execution directory of this trial under the sweep root."""
    root = Path(str(raw["output_dir"]))
    number = _running_trial_number()
    if number is None:
        number = int(HydraConfig.get().job.num)
    return root / "trials" / f"trial_{number:03d}"


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


def _running_trial_number() -> int | None:
    """The study's single running trial, or ``None`` when it is ambiguous."""
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
    if len(running) != 1:
        return None
    return int(running[0].number)


if __name__ == "__main__":
    objective()
