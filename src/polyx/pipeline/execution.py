"""One config-driven training execution on the local backend.

The engine owns the parts that must be identical for every family: dataset
audit and hashing, provenance, the shared evaluator, early stopping on the
with-class score, best-weight restoration and the artifact layout. The family
adapter owns only the framework-specific fit.
"""

from __future__ import annotations

import json
import os
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
    write_json_atomic,
    write_report,
    write_yaml,
)
from polyx.pipeline.coco_import import audit_as_dict
from polyx.pipeline.config import ExecutionConfig, config_to_dict, protocol_to_dict
from polyx.pipeline.evaluation import SharedEvaluator
from polyx.pipeline.resume import ResumeState
from polyx.pipeline.training import EarlyStoppingState


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact written by this engine."""
    loaded = _try_read_json(path)
    if loaded is None:
        raise RuntimeError(f"cannot read artifact {path}")
    return loaded


def _try_read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON artifact, or ``None`` when it is absent or truncated."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _completion_record(
    config: ExecutionConfig,
    *,
    selected_epoch: int,
    selection: dict[str, Any],
    checkpoint: Path,
    epochs_completed: int,
) -> dict[str, Any]:
    """The last artifact written, naming the execution as finished."""
    return {
        "status": "completed",
        "protocol": protocol_to_dict(config),
        "family": asdict(config.family),
        "backend": asdict(config.backend),
        "selected_epoch": selected_epoch,
        "selection": selection,
        "checkpoint": str(checkpoint),
        "epochs_completed": epochs_completed,
    }


def run_execution(config: ExecutionConfig) -> dict[str, Any]:
    """Run one execution and return a small summary of its artifacts."""
    layout = ArtifactLayout(config.output_dir)
    resume_state: ResumeState | None = None
    if config.resume:
        resume_state = ResumeState.for_output_dir(config.output_dir)
        if resume_state.protocol != protocol_to_dict(config):
            raise RuntimeError("protocol changed since the progress snapshot was written")
    layout.create()
    if not config.resume and layout.completion.is_file():
        # A fresh execution in a reused directory must not inherit an old
        # completion marker. Remove it and the resume records before training;
        # if this run stops early it will not be mistaken for finished.
        for stale in (layout.completion, layout.progress, layout.metrics, layout.training_state):
            stale.unlink(missing_ok=True)
    if resume_state is not None and layout.completion.is_file():
        # A completed execution is never rebuilt; the completion record is the
        # one authority.
        metrics = _read_json(layout.metrics)
        return {
            "output_dir": layout.root,
            "checkpoint": Path(str(_read_json(layout.completion)["checkpoint"])),
            "selection": metrics["selection"],
            "metrics": metrics,
            "training_state": _read_json(layout.training_state),
        }

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
        prepared.ground_truth,
        config.protocol.split,
        task=config.family.task,
        confidence_threshold=config.protocol.confidence,
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

    best: dict[str, Any] = {
        "predictions": [],
        "adapter_figures": {},
        "metrics": None,
        "epoch": -1,
    }
    if resume_state is not None and resume_state.best_epoch >= 0:
        # A resumed execution inherits the best epoch it never re-evaluates.
        # Without its figures a non-improving remaining epoch would score an
        # empty prediction list and overwrite the real metrics with zeros.
        if resume_state.best_metrics is None:
            raise RuntimeError(
                "the progress snapshot names a best epoch but carries no best "
                "metrics; it predates the resumable selection record"
            )
        best.update(metrics=resume_state.best_metrics, epoch=resume_state.best_epoch)
    first_checkpoint: Path | None = (
        resume_state.last_checkpoint if resume_state is not None else None
    )

    git = commit_git(ROOT)
    # A container has no source checkout: the build embeds the commit as
    # POLYX_SOURCE_SHA. Keep the repository answer when one is available.
    image_commit = os.environ.get("POLYX_SOURCE_SHA", "").strip()
    if not git.get("commit") and image_commit:
        git = {**git, "commit": image_commit, "source": "image"}
    provenance: dict[str, Any] = {
        "git": git,
        "dataset_hashes": prepared.hashes,
        "dataset_audit": audit_as_dict(audit),
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
                metrics=metrics,
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
                "best_metrics": best["metrics"],
                "protocol": protocol_to_dict(config),
            }
        )
        write_json_atomic(progress, layout.progress)
        return decision.stop

    def publish(
        final_metrics: dict[str, Any],
        training_state: dict[str, Any],
        best_artifact: Path,
        epochs_completed: int,
    ) -> dict[str, Any]:
        """Write every required artifact, then the completion record last."""
        write_json(final_metrics, layout.metrics)
        write_json(training_state, layout.training_state)
        write_json(provenance, layout.provenance)
        write_report(layout.report, final_metrics, training_state, config_to_dict(config))
        append_log(
            layout.training_log,
            f"done best={best_artifact} selection={final_metrics['selection']['value']:.6f}",
        )
        # The completion record is the final write: no artifact or log line may
        # fail after the run has already been marked complete.
        write_json_atomic(
            _completion_record(
                config,
                selected_epoch=int(final_metrics.get("selected_epoch", -1)),
                selection=final_metrics["selection"],
                checkpoint=best_artifact,
                epochs_completed=epochs_completed,
            ),
            layout.completion,
        )
        return {
            "output_dir": layout.root,
            "checkpoint": best_artifact,
            "selection": final_metrics["selection"],
            "metrics": final_metrics,
            "training_state": training_state,
        }

    training_done = resume_state is not None and (
        resume_state.epochs_completed >= config.protocol.epochs or resume_state.stopped_early
    )
    if training_done and not layout.completion.is_file():
        # The final epoch is already on disk. Finish publishing what is
        # missing instead of retraining or skipping the execution.
        if stopping.best_checkpoint is None:
            raise RuntimeError("the progress snapshot has no best checkpoint to publish")
        best_artifact = stopping.restore(layout.best_checkpoint)
        final_metrics = _try_read_json(layout.metrics)
        if final_metrics is None:
            if resume_state.best_metrics is None:
                raise RuntimeError("the progress snapshot has no best metrics to publish")
            final_metrics = dict(resume_state.best_metrics)
        final_metrics.setdefault("selected_epoch", resume_state.best_epoch)
        training_state = _try_read_json(layout.training_state)
        if training_state is None:
            training_state = {
                "protocol": protocol_to_dict(config),
                "family": asdict(config.family),
                "backend": asdict(config.backend),
                "amp": asdict(config.protocol.amp),
                "epochs_completed": resume_state.epochs_completed,
                "early_stopping": stopping.to_dict(resume_state.epochs_completed),
                "best_checkpoint": str(best_artifact),
                "recovered_publication": True,
            }
        return publish(final_metrics, training_state, best_artifact, resume_state.epochs_completed)

    adapter = build_adapter(config.family)
    fit = adapter.fit(config, layout.root, prepared, on_epoch, resume=resume_state)

    if stopping.best_checkpoint is None and first_checkpoint is not None:
        stopping.update(0.0, 0, first_checkpoint)
    best_artifact = stopping.restore(layout.best_checkpoint)

    if best["metrics"] is not None:
        final_metrics = dict(best["metrics"])
    else:
        final_metrics = evaluator.evaluate(
            best["predictions"], adapter_figures=best["adapter_figures"]
        )
    final_metrics["selected_epoch"] = best["epoch"]

    training_state: dict[str, Any] = {
        "protocol": protocol_to_dict(config),
        "family": asdict(config.family),
        "backend": asdict(config.backend),
        "amp": asdict(config.protocol.amp),
        "precision": fit.precision,
        "amp_requested": fit.amp_requested,
        "amp_active": fit.amp_active,
        "device": fit.device,
        "framework": fit.framework,
        "weights": fit.weights,
        "weights_verified": fit.weights_verified,
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
    return publish(final_metrics, training_state, best_artifact, fit.epochs_completed)
