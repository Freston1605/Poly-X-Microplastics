"""Headless batch detection for Poly-X (no Qt).

Runs a DetectionSource over a folder of images, writes annotated PNGs,
centroids.csv and metrics.json (when GT is present). No HTML report.

Masks produced by a segmentation backend (``yolo-seg``, SAM2, two-stage)
are honored where available: measurement uses them before falling back to
box-derived Otsu segmentation, and GT matching switches to mask IoU when
both sides of a comparison carry a mask.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from polyx.core.detect.protocol import (
    DetectionSource, Instance, read_yolo_txt, find_gt_for_image, compute_box_size_um,
)
from polyx.core.quality.metrics import match_image
from polyx.core.measure.morphology import apply_to_detection


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _draw(img, dets):
    out = img.copy()
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 200, 0), 2)
        label = f"{d.class_name} {d.conf:.2f}"
        cv2.putText(out, label, (int(d.x1), max(12, int(d.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return out


def _measure(det: Instance, img: np.ndarray, um_per_px: Optional[float]) -> None:
    """Box-size (+ shape) measurement, preferring an attached mask."""
    compute_box_size_um(det, um_per_px if um_per_px and um_per_px > 0 else None)
    apply_to_detection(det, img, um_per_px if um_per_px and um_per_px > 0 else None,
                       mask=getattr(det, "mask", None))


def run_batch(model: DetectionSource, image_dir: Path, output_dir: Path,
              conf: float = 0.25, iou: float = 0.45, imgsz: int = 1280,
              device: Optional[str] = None, um_per_px: float = 0.0,
              measure_shape: bool = True, gt_folder: Optional[Path] = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    images = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    rows = []
    metrics_agg = {"tp": 0, "fp": 0, "fn": 0, "miscls": 0}
    has_gt_any = False

    for img_path in images:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = model.predict(str(img_path), conf=conf, iou=iou, imgsz=imgsz, device=device)
        if measure_shape:
            for d in dets:
                _measure(d, img, um_per_px)

        # GT matching
        gt_txt = find_gt_for_image(img_path, gt_folder) if gt_folder else None
        gts = []
        if gt_txt is not None:
            has_gt_any = True
            names = getattr(model, "names", {}) or {}
            gts = read_yolo_txt(gt_txt, W, H, names)
            if measure_shape:
                for g in gts:
                    _measure(g, img, um_per_px)
            # Mask IoU only when both sides can be evaluated through masks;
            # otherwise the greedy match degrades to box IoU, which is the
            # historical behavior.
            use_masks = all(getattr(d, "mask", None) is not None for d in dets) and \
                all(getattr(g, "mask", None) is not None for g in gts)
            m = match_image(dets, gts, iou_thr=0.5, use_mask_iou=use_masks)
            metrics_agg["tp"] += m.tp
            metrics_agg["fp"] += m.fp
            metrics_agg["fn"] += m.fn
            metrics_agg["miscls"] += m.miscls

        # save annotated
        annotated = _draw(img, dets)
        out_img = img_dir / (img_path.stem + ".png")
        cv2.imencode(".png", annotated)[1].tofile(str(out_img))

        for d in dets:
            rows.append({
                "image": img_path.name,
                "class_id": int(d.class_id),
                "class": d.class_name,
                "conf": round(float(d.conf), 4),
                "x1": int(d.x1), "y1": int(d.y1), "x2": int(d.x2), "y2": int(d.y2),
                "diam_um": d.diam_um,
                "area_um2": d.area_um2,
                "length_um": d.length_um,
                "width_um": d.width_um,
                "aspect": d.aspect,
                "has_mask": bool(getattr(d, "mask", None) is not None),
            })

    # centroids.csv (always written, even empty, so downstream tools can rely on it)
    csv_path = output_dir / "centroids.csv"
    fieldnames = ["image", "class_id", "class", "conf", "x1", "y1", "x2", "y2",
                  "diam_um", "area_um2", "length_um", "width_um", "aspect", "has_mask"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # metrics.json (zeros even without GT; "has_gt" is the honesty flag)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_agg, indent=2), encoding="utf-8")

    summary = {
        "images": len(images),
        "detections": len(rows),
        "has_gt": has_gt_any,
        "metrics": metrics_agg,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = ["run_batch", "IMAGE_EXTS"]
