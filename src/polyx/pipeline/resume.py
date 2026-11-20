"""The progress snapshot that makes one execution resumable.

A preemption or a time limit can stop a SLURM job after a checkpoint has been
written but before the execution has finished. Lightning writes the checkpoint
itself and restores everything it needs from it: weights, optimizer, scheduler,
epoch progress and the state of the early-stopping and checkpoint callbacks.
This small file adds the facts Lightning does not carry: the protocol and the
dataset fingerprint the run belongs to, the fit it resolved (precision, device,
verified weights), and the path of the best checkpoint the loop kept.

The protocol that produced the snapshot is stored with it: resuming a different
protocol would mix methodologies and is refused. The dataset fingerprint
travels with it for the same reason: relabelled images or an edited COCO export
are a different dataset, and the checkpoint belongs to the bytes that trained
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyx.pipeline.artifacts import ArtifactLayout


@dataclass(frozen=True)
class ResumeState:
    """Observable state read from a progress snapshot."""

    epochs_completed: int
    resume_checkpoint: Path | None
    best_score: float | None
    stopped_early: bool
    protocol: dict[str, Any]
    #: Digest of every dataset file the interrupted execution trained on.
    dataset_fingerprint: str
    #: The fit resolved by the interrupted execution (precision, device,
    #: verified weights, augmentation), carried so a recovered publication
    #: does not have to invent them.
    fit: dict[str, Any] | None = None
    #: Non-improving validation epochs the early-stopping callback counted.
    wait_count: int = 0
    #: The epoch that set the kept checkpoint, and that checkpoint's path.
    best_epoch: int = -1
    best_lightning_checkpoint: Path | None = None

    @classmethod
    def load(cls, path: Path) -> ResumeState:
        """Read and validate a progress snapshot written by a previous run."""
        path = Path(path)
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable progress snapshot {path}: {exc}") from exc
        try:
            resume_checkpoint = raw.get("resume_checkpoint")
            fit = raw.get("fit")
            if fit is not None and not isinstance(fit, dict):
                raise TypeError("fit must be a mapping")
            fingerprint = raw.get("dataset_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(
                    "the progress snapshot predates the dataset identity record, so the "
                    "dataset behind this checkpoint cannot be verified; start a new output "
                    "directory instead of resuming over it"
                )
            return cls(
                epochs_completed=int(raw["epochs_completed"]),
                resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
                best_score=raw.get("best_score"),
                stopped_early=bool(raw.get("stopped_early", False)),
                protocol=dict(raw["protocol"]),
                dataset_fingerprint=fingerprint,
                fit=fit,
                wait_count=int((raw.get("early_stopping") or {}).get("wait_count", 0)),
                best_epoch=int(raw.get("best_epoch", -1)),
                best_lightning_checkpoint=(
                    Path(raw["best_lightning_checkpoint"])
                    if raw.get("best_lightning_checkpoint")
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid progress snapshot {path}: {exc}") from exc

    @classmethod
    def for_output_dir(cls, output_dir: Path) -> ResumeState:
        """Load the snapshot for ``output_dir`` or explain why it cannot resume."""
        layout = ArtifactLayout(output_dir)
        if not layout.progress.is_file():
            raise RuntimeError(
                f"resume requested but no progress snapshot exists at {layout.progress}"
            )
        return cls.load(layout.progress)
