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
from polyx.pipeline.adapters import EpochResult, build_adapter
from polyx.pipeline.artifacts import (
    ArtifactLayout,
    append_log,
    write_json,
    write_report,
    write_yaml,
)
from polyx.pipeline.config import ExecutionConfig, config_to_dict, protocol_to_dict
from polyx.pipeline.evaluation import SharedEvaluator
from polyx.pipeline.resume import ResumeState
from polyx.pipeline.training import EarlyStoppingState


def run_execution(config: ExecutionConfig) -> dict[str, Any]:
    """Run one execution and return a small summary of its artifacts."""
    layout = ArtifactLayout(config.output_dir)
    resume_state: ResumeState | None = None
    if config.resume:
        resume_state = ResumeState.for_output_dir(config.output_dir)
        if resume_state.protocol != protocol_to_dict(config):
            raise RuntimeError("protocol changed since the progress snapshot was written")
    layout.create()

    write_yaml(config_to_dict(config), layout.config)
    write_yaml(protocol_to_dict(config), layout.resolved_protocol)
    append_log(
        layout.training_log,
        f"family={config.family.name} backend={config.backend.name} resume={config.resume}",
    )
    if resume_state is not None:
        append_log(
            layout.training_log,
            f"resume from epoch={resume_state.epochs_completed} "
            f"checkpoint={resume_state.last_checkpoint}",
        )

    adapter = build_adapter(config.family)
    prepared = adapter.prepare(config)
    audit = prepared.audit
    if not audit.ok:
        problems = [f"{finding.title}: {finding.detail}" for finding in audit.findings]
        raise RuntimeError("dataset audit failed: " + "; ".join(problems))
    if prepared.conversion is not None:
        append_log(
            layout.training_log,
            "dataset=coco_instances "
            f"conversion={prepared.conversion['root']} "
            f"cache_key={prepared.conversion['cache_key']} "
            f"cached={str(prepared.conversion['cached']).lower()}",
        )

    evaluator = SharedEvaluator(
        prepared.ground_truth, config.protocol.split, task=config.family.task
    )
    stopping = EarlyStoppingState(asdict(config.protocol.early_stopping))
    if resume_state is not None:
        stopping.load_state(
            {
                "best_score": resume_state.best_score,
                "best_epoch": resume_state.best_epoch,
                "best_checkpoint": (
                    str(resume_state.best_checkpoint) if resume_state.best_checkpoint else None
                ),
                "non_improving": resume_state.non_improving,
                "stopped_early": resume_state.stopped_early,
            }
        )

    best: dict[str, Any] = {"predictions": [], "adapter_figures": {}, "epoch": -1}
    first_checkpoint: Path | None = (
        resume_state.last_checkpoint if resume_state is not None else None
    )

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
        progress = stopping.to_dict(result.epoch + 1)
        progress.update(
            {
                "epoch_completed": result.epoch,
                "last_checkpoint": str(result.checkpoint),
                "resume_checkpoint": (
                    str(result.resume_checkpoint) if result.resume_checkpoint else None
                ),
                "best_checkpoint": (
                    str(stopping.best_checkpoint) if stopping.best_checkpoint else None
                ),
                "protocol": protocol_to_dict(config),
            }
        )
        write_json(progress, layout.progress)
        return decision.stop

    adapter = build_adapter(config.family)
    fit = adapter.fit(config, layout.root, prepared, on_epoch, resume=resume_state)

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
        "augmentation": fit.augmentation,
        "early_stopping": stopping.to_dict(fit.epochs_completed),
        "best_checkpoint": str(best_artifact),
    }
    if resume_state is not None:
        training_state["resumed"] = True
        training_state["resumed_from_epoch"] = resume_state.epochs_completed
        training_state["resumed_from"] = (
            str(resume_state.last_checkpoint) if resume_state.last_checkpoint else None
        )
    write_json(training_state, layout.training_state)

    provenance: dict[str, Any] = {
        "git": commit_git(ROOT),
        "dataset_hashes": prepared.hashes,
        "dataset_audit": asdict(audit),
        "data_yaml": (
            str(config.data_yaml)
            if prepared.source_kind == "yolo_boxes" and config.data_yaml is not None
            else ""
        ),
        "dataset_source": {
            "kind": prepared.source_kind,
            "path": str(prepared.source_path),
        },
        "dataset_name": config.dataset_name,
    }
    if prepared.conversion is not None:
        provenance["conversion"] = prepared.conversion
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
