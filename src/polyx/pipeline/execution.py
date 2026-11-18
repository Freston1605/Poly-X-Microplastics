"""One config-driven training execution on the local backend.

The engine owns the parts that must be identical for every family: dataset
audit and hashing, provenance, the shared evaluator, early stopping on the
with-class score, best-weight restoration and the artifact layout. The family
adapter owns only the framework-specific fit.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from polyx.core.paths import ROOT
from polyx.core.provenance import commit_git
from polyx.core.quality.dataset_audit import audit_dataset
from polyx.pipeline.adapters import EpochResult, build_adapter
from polyx.pipeline.artifacts import (
    ArtifactLayout,
    append_log,
    write_json,
    write_report,
    write_yaml,
)
from polyx.pipeline.config import ExecutionConfig, config_to_dict, protocol_to_dict
from polyx.pipeline.dataset import coco_ground_truth, dataset_hashes
from polyx.pipeline.evaluation import SharedEvaluator
from polyx.pipeline.training import EarlyStoppingState


def run_execution(config: ExecutionConfig) -> dict[str, Any]:
    """Run one execution and return a small summary of its artifacts."""
    layout = ArtifactLayout(config.output_dir)
    layout.create()

    write_yaml(config_to_dict(config), layout.config)
    write_yaml(protocol_to_dict(config), layout.resolved_protocol)
    append_log(layout.training_log, f"family={config.family.name} backend={config.backend.name}")

    audit = audit_dataset(config.data_yaml)
    if not audit.ok:
        problems = [f"{finding.title}: {finding.detail}" for finding in audit.findings]
        raise RuntimeError("dataset audit failed: " + "; ".join(problems))

    ground_truth = coco_ground_truth(config.data_yaml, config.protocol.split)
    evaluator = SharedEvaluator(ground_truth, config.protocol.split)
    stopping = EarlyStoppingState(asdict(config.protocol.early_stopping))

    best: dict[str, Any] = {"predictions": [], "adapter_figures": {}, "epoch": -1}
    first_checkpoint: Path | None = None

    def on_epoch(result: EpochResult) -> bool:
        nonlocal first_checkpoint
        if first_checkpoint is None:
            first_checkpoint = result.checkpoint
        metrics = evaluator.evaluate(
            result.predictions,
            adapter_figures=result.adapter_figures,
        )
        score = float(metrics["selection"]["value"])
        decision = stopping.update(score, result.epoch, result.checkpoint)
        append_log(
            layout.training_log,
            f"epoch={result.epoch} with_class_score={score:.6f} improved={decision.improved}",
        )
        if decision.improved:
            best.update(
                predictions=result.predictions,
                adapter_figures=result.adapter_figures,
                epoch=result.epoch,
            )
        return decision.stop

    adapter = build_adapter(config.family)
    fit = adapter.fit(config, layout.root, on_epoch)

    if stopping.best_checkpoint is None and first_checkpoint is not None:
        stopping.update(0.0, 0, first_checkpoint)
    best_artifact = stopping.restore(layout.best_checkpoint)

    final_metrics = evaluator.evaluate(best["predictions"], adapter_figures=best["adapter_figures"])
    final_metrics["selected_epoch"] = best["epoch"]
    write_json(final_metrics, layout.metrics)

    training_state: dict[str, Any] = {
        "protocol": protocol_to_dict(config),
        "family": asdict(config.family),
        "backend": asdict(config.backend),
        "amp": asdict(config.protocol.amp),
        "precision": fit.precision,
        "amp_active": fit.amp_active,
        "device": fit.device,
        "framework": fit.framework,
        "weights": fit.weights,
        "epochs_completed": fit.epochs_completed,
        "early_stopping": stopping.to_dict(fit.epochs_completed),
        "best_checkpoint": str(best_artifact),
    }
    write_json(training_state, layout.training_state)

    provenance: dict[str, Any] = {
        "git": commit_git(ROOT),
        "dataset_hashes": dataset_hashes(config.data_yaml),
        "dataset_audit": asdict(audit),
        "data_yaml": str(config.data_yaml),
        "dataset_name": config.dataset_name,
    }
    write_json(provenance, layout.provenance)
    write_report(layout.report, final_metrics, training_state, config_to_dict(config))
    append_log(
        layout.training_log,
        f"done best={best_artifact} selection={final_metrics['selection']['value']:.6f}",
    )

    return {
        "output_dir": layout.root,
        "checkpoint": best_artifact,
        "selection": final_metrics["selection"],
        "metrics": final_metrics,
        "training_state": training_state,
    }
