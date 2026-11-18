"""Hydra objective for the real Optuna sweeper multirun.

``python -m polyx.pipeline.sweep_app --multirun`` hands each configuration here
and records the returned number as the study value. It is the shared evaluator's
with-class score from :func:`run_execution`, the same number the one-process
driver returns through ``polyx-pipeline --sweep``.
"""

from __future__ import annotations

from hydra import main
from omegaconf import DictConfig, OmegaConf

import polyx.pipeline.sweep  # noqa: F401 - registers the Optuna config groups
from polyx.pipeline.config import execution_config_from_raw
from polyx.pipeline.execution import run_execution


@main(config_path="conf", config_name="sweep", version_base="1.3")
def objective(cfg: DictConfig) -> float:
    """Run one suggested trial and return only the with-class score."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    return float(run_execution(execution_config_from_raw(raw))["selection"]["value"])


if __name__ == "__main__":
    objective()
