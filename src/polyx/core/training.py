"""Headless training engine for Poly-X (no Qt, no GUI).

Wraps Ultralytics YOLO training behind a plain-Python API so it can run in
CI, containers and sweeps without a display. The GUI Trainer is a thin
adapter over this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class TrainingConfig:
    """Plain-data training configuration (no Qt)."""
    weights: str                          # base weights (e.g. "yolov8m.pt")
    data_yaml: str                        # path to data.yaml
    project: str = "runs"                 # output root
    name: str = "train"                   # run name
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: Optional[str] = None      # None = let Ultralytics auto-select
    optimizer: str = "SGD"
    lr0: float = 0.01
    extra: dict = field(default_factory=dict)   # forwarded to model.train()


def train(cfg: TrainingConfig,
          on_epoch: Optional[Callable[[dict], None]] = None,
          on_log: Optional[Callable[[str], None]] = None,
          callbacks: Optional[dict[str, Callable]] = None) -> Path:
    """Run Ultralytics training and return the path to best.pt.

    ``callbacks`` is a mapping of Ultralytics event name -> callable(trainer),
    registered on the model before training (e.g. GUI progress callbacks).
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Ultralytics unavailable: {exc}") from exc

    if on_log:
        on_log(f"[train] loading base weights: {cfg.weights}")
    model = YOLO(cfg.weights)

    def _on_epoch_end(trainer):
        if on_epoch is None:
            return
        metrics = getattr(trainer, "metrics", {}) or {}
        on_epoch({"epoch": getattr(trainer, "epoch", None), "metrics": metrics, "trainer": trainer})

    model.add_callback("on_train_epoch_end", _on_epoch_end)
    for name, cb in (callbacks or {}).items():
        try:
            model.add_callback(name, cb)
        except Exception as exc:
            # A dropped progress callback is a silent bug downstream: say so.
            if on_log:
                on_log(f"[train] WARNING: callback '{name}' not registered: {exc}")

    kwargs = {
        "data": cfg.data_yaml,
        "epochs": cfg.epochs,
        "batch": cfg.batch,
        "imgsz": cfg.imgsz,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "project": cfg.project,
        "name": cfg.name,
        "exist_ok": True,
    }
    if cfg.device:
        kwargs["device"] = cfg.device
    kwargs.update(cfg.extra or {})

    if on_log:
        on_log(f"[train] starting: epochs={cfg.epochs} batch={cfg.batch} imgsz={cfg.imgsz}")
    model.train(**kwargs)

    run_dir = Path(cfg.project) / cfg.name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        candidates = list(Path(cfg.project).rglob("best.pt"))
        if candidates:
            best = max(candidates, key=lambda p: p.stat().st_mtime)
    if on_log:
        on_log(f"[train] done: best={best}")
    return best


__all__ = ["TrainingConfig", "train"]