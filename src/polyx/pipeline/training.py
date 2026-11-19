"""Shared early-stopping bookkeeping for the single training loop.

The with-class score from the shared evaluator is the only monitored value.
Patience counts non-improving epochs beyond the minimum delta, and the best
weights are copied back to the canonical checkpoint path so a later ticket can
consume one stable file regardless of how many epochs ran.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StopDecision:
    """Whether this epoch improved and whether the loop must stop."""

    improved: bool
    stop: bool


class EarlyStoppingState:
    """Tracks the best with-class score and restores its weights."""

    def __init__(self, config: dict) -> None:
        self.monitor = str(config.get("monitor", "with_class_score"))
        self.mode = str(config.get("mode", "max"))
        self.patience = int(config.get("patience", 30))
        self.min_delta = float(config.get("min_delta", 0.001))
        self.restore_best = bool(config.get("restore_best", True))
        if self.monitor != "with_class_score":
            raise ValueError(f"unsupported early-stopping monitor: {self.monitor}")
        if self.mode != "max":
            raise ValueError(f"unsupported early-stopping mode: {self.mode}")
        if not self.restore_best:
            # The loop always keeps the best epoch's weights as `best.pt`; the
            # artifact layout has no mode that omits it, so the flag is
            # validated out instead of crashing at the end of training.
            raise ValueError(
                "early_stopping.restore_best=false is not supported: the loop "
                "always keeps the best checkpoint as best.pt"
            )
        self.best_score: float | None = None
        self.best_epoch = -1
        self.best_checkpoint: Path | None = None
        self.non_improving = 0
        self.stopped_early = False

    def update(self, score: float, epoch: int, checkpoint: Path) -> StopDecision:
        """Record one epoch's score and decide whether to stop."""
        improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_score = float(score)
            self.best_epoch = epoch
            self.best_checkpoint = Path(checkpoint)
            self.non_improving = 0
        else:
            self.non_improving += 1
        stop = self.non_improving >= self.patience
        if stop:
            self.stopped_early = True
        return StopDecision(improved=improved, stop=stop)

    def load_state(self, state: dict) -> None:
        """Restore bookkeeping from a progress snapshot of an earlier run."""
        self.best_score = state.get("best_score")
        self.best_epoch = int(state["best_epoch"])
        checkpoint = state.get("best_checkpoint")
        self.best_checkpoint = Path(checkpoint) if checkpoint else None
        self.non_improving = int(state.get("non_improving", 0))
        self.stopped_early = bool(state.get("stopped_early", False))

    def restore(self, destination: Path) -> Path:
        """Copy the best epoch's weights to the canonical checkpoint path."""
        if self.best_checkpoint is None or not self.restore_best:
            raise RuntimeError("no best checkpoint to restore")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.best_checkpoint, destination)
        return destination

    def to_dict(self, epochs_completed: int) -> dict:
        """The observable early-stopping state written to the artifacts."""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "restore_best": self.restore_best,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "epochs_completed": epochs_completed,
            "non_improving": self.non_improving,
            "stopped_early": self.stopped_early,
        }
