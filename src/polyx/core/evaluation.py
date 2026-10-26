"""Headless evaluation engine for Poly-X models (no Qt).

Wraps Ultralytics validation behind a plain-Python API and writes a JSON
report with per-class P/R/AP50 and mAP. The metric-extraction half is a
pure function on the Ultralytics result object so it can be unit-tested
without loading a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalConfig:
    weights: str
    data_yaml: str
    split: str = "val"
    imgsz: int = 640
    batch: int = 16
    device: str | None = None  # None = let Ultralytics auto-select
    iou: float = 0.5
    extra: dict = field(default_factory=dict)


def _metrics_from_results(results: Any, names: dict) -> dict:
    """Extract mAP and per-class P/R/AP50 from a DetMetrics-like object.

    Ultralytics only reports per-class arrays (p, r, ap50) for the classes
    that actually appear (``box.ap_class``), so the per-class loop maps
    through that index list instead of assuming dense class indices.
    """
    metrics: dict = {}
    try:
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50-95"] = float(results.box.map)
    except (AttributeError, TypeError):
        metrics["mAP50"] = None
        metrics["mAP50-95"] = None

    per_class = {}
    try:
        p = list(results.box.p)
        r = list(results.box.r)
        ap50 = list(results.box.ap50)
        # Ultralytics names the mapping "ap_class_index" (the class id for
        # each entry of p/r/ap50); older/newer shapes are guarded, and the
        # dense fallback only applies when the mapping is absent.
        ap_classes = list(
            getattr(
                results.box,
                "ap_class_index",
                getattr(results.box, "ap_class", range(len(p))) or range(len(p)),
            )
        )
        for i, cid in enumerate(ap_classes):
            name = names.get(int(cid), str(cid))
            per_class[str(name)] = {
                "precision": float(p[i]) if i < len(p) else None,
                "recall": float(r[i]) if i < len(r) else None,
                "AP50": float(ap50[i]) if i < len(ap50) else None,
            }
    except (AttributeError, TypeError, IndexError):
        pass
    metrics["per_class"] = per_class
    metrics["names"] = {str(k): str(v) for k, v in names.items()}
    return metrics


def evaluate(cfg: EvalConfig, on_log: Any | None = None) -> dict:
    """Run Ultralytics validation and return a metrics dict."""
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Ultralytics unavailable: {exc}") from exc

    if on_log:
        on_log(f"[eval] loading: {cfg.weights}")
    model = YOLO(cfg.weights)
    kwargs = {
        "data": cfg.data_yaml,
        "split": cfg.split,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "iou": cfg.iou,
    }
    # Ultralytics chooses CUDA when available when device is not forced.
    if cfg.device:
        kwargs["device"] = cfg.device
    kwargs.update(cfg.extra or {})
    results = model.val(**kwargs)

    names = getattr(results, "names", {}) or getattr(model, "names", {}) or {}
    return _metrics_from_results(results, names)


def write_report(metrics: dict, path: Path) -> None:
    """Persist the metrics dict as formatted JSON, creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


__all__ = ["EvalConfig", "_metrics_from_results", "evaluate", "write_report"]
