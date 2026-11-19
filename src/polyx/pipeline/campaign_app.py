"""Hydra multirun entry point for the benchmark campaign.

``python -m polyx.pipeline.campaign_app --multirun`` launches one job per model
family inside a single allocation. Overriding ``family`` with a comma-separated
list is what makes it a campaign rather than a single execution; every job reads
the same protocol composition and writes its artifacts under
``<campaign_root>/<family>``. Nothing here is family-specific: the adapters,
evaluator, early stopping and artifact layout are the same ones a single
execution uses.
"""

from __future__ import annotations

from pathlib import Path

from hydra import main
from omegaconf import DictConfig, OmegaConf

from polyx.pipeline.config import execution_config_from_raw
from polyx.pipeline.execution import run_execution


def campaign_raw(cfg: DictConfig) -> dict:
    """Resolve one campaign job into the plain mapping the engine validates."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    root = str(raw.get("campaign_root") or "").strip()
    if not root:
        raise ValueError("campaign_root is a required campaign override")
    if not raw.get("output_dir"):
        family = str((raw.get("family") or {}).get("name") or "")
        raw["output_dir"] = str(Path(root) / family)
    return raw


@main(config_path="conf", config_name="campaign", version_base="1.3")
def campaign(cfg: DictConfig) -> float:
    """Run one family's training execution and return its with-class score."""
    summary = run_execution(execution_config_from_raw(campaign_raw(cfg)))
    print(f"CAMPAIGN_FAMILY: {summary['training_state']['family']['name']}")
    print(f"OUTPUT: {summary['output_dir']}")
    print(f"SELECTION: {summary['selection']['metric']}={summary['selection']['value']:.6f}")
    return float(summary["selection"]["value"])


if __name__ == "__main__":
    campaign()
