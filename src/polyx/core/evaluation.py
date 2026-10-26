"""Headless evaluation engine for Poly-X models (no Qt).

Wraps Ultralytics validation behind a plain-Python API and writes a JSON
report with per-class P/R/F1 and mAP.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvalConfig:
    weights: str
    data_yaml: str
    split: str = "val"
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    iou: float = 0.5
    extra: dict = field(default_factory=dict)


def evaluate(cfg: EvalConfig, on_log: Optional[Any] = None) -> dict:
    """Run Ultralytics validation and return a metrics dict."""
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Ultralytics unavailable: {exc}") from exc

    if on_log:
        on_log(f"[eval] loading: {cfg.weights}")
    model = YOLO(cfg.weights)
    kwargs = {
        "data": cfg.data_yaml,
        "split": cfg.split,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "device": cfg.device,
        "iou": cfg.iou,
    }
    kwargs.update(cfg.extra or {})
    results = model.val(**kwargs)

    # Ultralytics returns a DetMetrics-like object.
    metrics = {}
    try:
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50-95"] = float(results.box.map)
    except Exception:
        metrics["mAP50"] = None
        metrics["mAP50-95"] = None

    names = getattr(results, "names", {}) or getattr(model, "names", {}) or {}
    per_class = {}
    try:
        p = results.box.p  # precision per class
        r = results.box.r
        ap50 = results.box.ap50
        for i, name in names.items():
            per_class[str(name)] = {
                "precision": float(p[i]) if i < len(p) else None,
                "recall": float(r[i]) if i < len(r) else None,
                "AP50": float(ap50[i]) if i < len(ap50) else None,
            }
    except Exception:
        pass
    metrics["per_class"] = per_class
    metrics["names"] = {str(k): str(v) for k, v in names.items()}
    return metrics


def write_report(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


__all__ = ["EvalConfig", "evaluate", "write_report"]