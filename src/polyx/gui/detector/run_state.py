"""Rebuild a DetectorState from a headless run folder.

The headless CLI (``polyx-detect``) writes centroids.csv, summary.json and
annotated PNGs. When a paper-quality report is wanted afterwards, this
module reassembles the Qt-state shape that ``polyx.gui.report_html``
expects. It lives in ``polyx.gui`` — not in the CLI — so the no-Qt-in-CLI
guards stay true by construction instead of by importlib deferment.

Honesty note: the report sums ``tp/fp/fn`` per image, so those counts are
taken from ``summary["per_image"]`` (written by ``run_batch``) and never
from the run aggregate. An aggregate copied into every row would be
multiplied by the number of images.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from polyx.core.detection_run import IMAGE_EXTS


def _load_summary(run_dir: Path) -> dict:
    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_state_from_run(run_dir: Path, image_dir: Path):
    """Load a headless run into a ``DetectorState`` for report generation.

    Reads centroids.csv (predictions, one row per detection incl. class_id),
    summary.json (run params + per-image tp/fp/fn) and points each
    ImageResult at its annotated PNG. Returns the state ready for
    ``polyx.gui.report_html.generate_report``.
    """
    from polyx.core.detect.yolo import Detection
    from polyx.gui.detector.state import DetectorState, ImageResult, InferenceParams

    run_dir = Path(run_dir)
    summary = _load_summary(run_dir)
    params_raw = summary.get("params", {}) or {}
    per_image = summary.get("per_image", {}) or {}

    # One ImageResult per image, predictions grouped by image name.
    by_image: dict[str, list] = {}
    centroids = run_dir / "centroids.csv"
    if centroids.is_file():
        with open(centroids, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                by_image.setdefault(r["image"], []).append(r)

    state = DetectorState()
    state.params = InferenceParams(
        conf=float(params_raw.get("conf", 0.25)),
        iou_nms=float(params_raw.get("iou", 0.45)),
        imgsz=int(params_raw.get("imgsz", 1280)),
        um_per_px=float(params_raw.get("um_per_px", 0.0) or 0.0),
    )
    if params_raw.get("device"):
        state.params.device = params_raw["device"]
    state.run_dir = run_dir

    images = sorted(p for p in Path(image_dir).rglob("*")
                    if p.suffix.lower() in IMAGE_EXTS)
    results: dict[int, list] = {}
    for img_path in images:
        preds = []
        for r in by_image.get(img_path.name, []):
            preds.append(Detection(
                class_id=int(r.get("class_id", 0) or 0),
                class_name=r.get("class", "0"),
                conf=float(r.get("conf", 0.0) or 0.0),
                x1=float(r["x1"]), y1=float(r["y1"]),
                x2=float(r["x2"]), y2=float(r["y2"]),
                diam_um=float(r["diam_um"]) if r.get("diam_um") not in (None, "") else None,
            ))
        counts = per_image.get(img_path.name) or {}
        img_result = ImageResult(
            image_path=img_path,
            model_idx=0,
            predictions=preds,
            has_gt=bool(counts),
            tp=int(counts.get("tp", 0)),
            fp=int(counts.get("fp", 0)),
            fn=int(counts.get("fn", 0)),
            miscls=int(counts.get("miscls", 0)),
            annotated_path=run_dir / "images" / (img_path.stem + ".png"),
        )
        results.setdefault(0, []).append(img_result)

    state.results = results
    state.images = images
    # Class names ride on the Instance objects themselves («class» column),
    # so the report's per-class sections resolve without extra bookkeeping.
    state.model_slots[0].alias = "model"
    return state
