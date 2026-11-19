"""The progress snapshot that makes one execution resumable.

A preemption or a time limit can stop a SLURM job after a checkpoint has been
written but before the execution has finished. The engine refreshes this small
file after every epoch so a later invocation with ``resume=true`` can continue
from the last saved epoch without rerunning completed work. It names two
checkpoints: ``last_checkpoint`` is the family-native export used for
best-weight restoration, and ``resume_checkpoint`` is the real Lightning
checkpoint whose weights, optimizer, scheduler and epoch progress continue the
loop. The protocol that produced the snapshot is stored with it: resuming a
different protocol would mix methodologies and is refused.
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
    last_checkpoint: Path | None
    resume_checkpoint: Path | None
    best_score: float | None
    best_epoch: int
    best_checkpoint: Path | None
    non_improving: int
    stopped_early: bool
    protocol: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ResumeState:
        """Read and validate a progress snapshot written by a previous run."""
        path = Path(path)
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable progress snapshot {path}: {exc}") from exc
        try:
            checkpoint = raw["last_checkpoint"]
            resume_checkpoint = raw.get("resume_checkpoint")
            best_checkpoint = raw.get("best_checkpoint")
            return cls(
                epochs_completed=int(raw["epochs_completed"]),
                last_checkpoint=Path(checkpoint) if checkpoint else None,
                resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
                best_score=raw.get("best_score"),
                best_epoch=int(raw["best_epoch"]),
                best_checkpoint=Path(best_checkpoint) if best_checkpoint else None,
                non_improving=int(raw.get("non_improving", 0)),
                stopped_early=bool(raw.get("stopped_early", False)),
                protocol=dict(raw["protocol"]),
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
