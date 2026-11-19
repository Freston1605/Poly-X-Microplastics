"""Hydra multirun entry point for the benchmark campaign.

``python -m polyx.pipeline.campaign_app --multirun`` launches one job per model
family inside a single allocation. Overriding ``family`` with a comma-separated
list is what makes it a campaign rather than a single execution; every job reads
the same protocol composition and writes its artifacts under
``<campaign_root>/<run>``. Nothing here is family-specific: the adapters,
evaluator, early stopping and artifact layout are the same ones a single
execution uses.

A campaign is resumable without a separate per-family switch. With
``resume=true`` a job resumes a family that has a progress snapshot, skips a
family whose execution already finished under the same protocol, and runs a
family that never started normally. Passing the flag to every job and letting
each job decide is what makes resubmission safe.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from hydra import main
from omegaconf import DictConfig, OmegaConf

from polyx.pipeline.artifacts import ArtifactLayout
from polyx.pipeline.config import (
    ExecutionConfig,
    execution_config_from_raw,
    protocol_to_dict,
)
from polyx.pipeline.dataset import validate_campaign_cohort
from polyx.pipeline.execution import run_execution


def campaign_raw(cfg: DictConfig) -> dict:
    """Resolve one campaign job into the plain mapping the engine validates."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    root = str(raw.get("campaign_root") or "").strip()
    if not root:
        raise ValueError("campaign_root is a required campaign override")
    if not raw.get("output_dir"):
        family = raw.get("family") or {}
        run_name = str(family.get("run_name") or family.get("name") or "")
        raw["output_dir"] = str(Path(root) / run_name)
    return raw


def _selection_from(layout: ArtifactLayout, state: dict) -> float:
    """The finished execution's score, preferring its recorded selection."""
    try:
        metrics = json.loads(layout.metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return float(state["early_stopping"].get("best_score") or 0.0)
    return float(metrics.get("selection", {}).get("value", 0.0))


def _resume_decision(config: ExecutionConfig) -> tuple[str, dict]:
    """Whether this family should resume, be skipped, or start fresh.

    Only a family with a progress snapshot resumes; one whose execution already
    finished under the same protocol is skipped so its metrics are not
    recomputed or overwritten; everything else starts normally even when the
    campaign was submitted with ``resume=true``.
    """
    layout = ArtifactLayout(config.output_dir)
    if layout.training_state.is_file():
        state = json.loads(layout.training_state.read_text(encoding="utf-8"))
        if state.get("protocol") != protocol_to_dict(config):
            raise RuntimeError(
                f"{config.output_dir} already holds an execution for a different "
                "protocol; use a new campaign root instead of resuming over it"
            )
        return "finished", state
    if layout.progress.is_file():
        return "resume", {}
    return "start", {}


@main(config_path="conf", config_name="campaign", version_base="1.3")
def campaign(cfg: DictConfig) -> float:
    """Run one family's training execution and return its with-class score."""
    config = execution_config_from_raw(campaign_raw(cfg))
    if config.data_yaml is not None and config.coco_dataset is not None:
        # Both sources travel together in a campaign. They must describe the
        # same real-domain cohort; otherwise the benchmark would vary the
        # dataset as well as the model.
        validate_campaign_cohort(config.data_yaml, config.coco_dataset, config.protocol.split)
    run_name = config.output_dir.name
    if config.resume:
        decision, state = _resume_decision(config)
        if decision == "finished":
            score = _selection_from(ArtifactLayout(config.output_dir), state)
            print(f"SKIPPED: {run_name} already finished; keeping its metrics")
            print(f"OUTPUT: {config.output_dir}")
            print(f"SELECTION: with_class_score={score:.6f}")
            return score
        # A family that never started must not be asked to resume.
        config = replace(config, resume=decision == "resume")

    summary = run_execution(config)
    print(f"CAMPAIGN_FAMILY: {summary['training_state']['family']['name']}")
    print(f"OUTPUT: {summary['output_dir']}")
    print(f"SELECTION: {summary['selection']['metric']}={summary['selection']['value']:.6f}")
    return float(summary["selection"]["value"])


if __name__ == "__main__":
    campaign()
