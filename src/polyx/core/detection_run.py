"""Headless batch detection for Poly-X (no Qt).

Runs a DetectionSource over a folder of images, writes annotated PNGs,
centroids.csv and metrics.json (when GT is present). No HTML report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from polyx.core.detect.protocol import (
    DetectionSource, Instance, read_yolo_txt, find_gt_for_image, compute_box_size_um,
)
from polyx.core.detect.yolo import size_image
from polyx.core.quality.metrics import match_image, iou
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


def run_batch(model: DetectionSource, image_dir: Path, output_dir: Path,
              conf: float = 0.25, iou: float = 0.45, imgsz: int = 1280,
              device: str = "0", um_per_px: float = 0.0,
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
        for d in dets:
            compute_box_size_um(d, um_per_px if um_per_px > 0 else None)
            if measure_shape:
                apply_to_detection(d, img, um_per_px if um_per_px > 0 else None)

        # GT matching
        gt_txt = find_gt_for_image(img_path, gt_folder) if gt_folder else None
        gts = []
        if gt_txt is not None:
            has_gt_any = True
            names = getattr(model, "names", {}) or {}
            gts = read_yolo_txt(gt_txt, W, H, names)
            for g in gts:
                compute_box_size_um(g, um_per_px if um_per_px > 0 else None)
                if measure_shape:
                    apply_to_detection(g, img, um_per_px if um_per_px > 0 else None)
            m = match_image(dets, gts, iou_thr=0.5)
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
                "class": d.class_name,
                "conf": round(float(d.conf), 4),
                "x1": int(d.x1), "y1": int(d.y1), "x2": int(d.x2), "y2": int(d.y2),
                "diam_um": d.diam_um,
                "area_um2": d.area_um2,
                "length_um": d.length_um,
                "width_um": d.width_um,
                "aspect": d.aspect,
            })

    # centroids.csv
    csv_path = output_dir / "centroids.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # metrics.json
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