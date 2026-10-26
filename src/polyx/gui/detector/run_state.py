"""Rebuild a DetectorState from a headless run folder.

The headless CLI (``polyx-detect``) writes centroids.csv + summary.json +
annotated PNGs. When a paper-quality report is wanted afterwards, this
module reassembles the Qt-state shape that ``polyx.gui.report_html``
expects. It lives in ``polyx.gui`` — not in the CLI — so the no-Qt-in-CLI
guards stay true by construction instead of by importlib deferment.
"""
from __future__ import annotations

import csv
from pathlib import Path

from polyx.core.detection_run import IMAGE_EXTS


def build_state_from_run(run_dir: Path, image_dir: Path):
    """Load a headless run into a ``DetectorState`` for report generation.

    Reads centroids.csv (predictions, one row per detection incl. class_id)
    and summary.json (has_gt flag). Returns the state ready for
    ``polyx.gui.report_html.generate_report``.
    """
    from polyx.core.detect.yolo import Detection
    from polyx.gui.detector.state import DetectorState, ImageResult, InferenceParams

    run_dir = Path(run_dir)
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        import json
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    has_gt_any = bool(summary.get("has_gt", False))

    # One ImageResult per image, predictions grouped by image name.
    by_image: dict[str, list] = {}
    with open(run_dir / "centroids.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_image.setdefault(r["image"], []).append(r)

    state = DetectorState()
    state.params = InferenceParams()
    state.run_dir = run_dir

    images = sorted(p for p in Path(image_dir).rglob("*")
                    if p.suffix.lower() in IMAGE_EXTS)
    results: dict[int, list] = {}
    total_metrics = {"tp": 0, "fp": 0, "fn": 0, "miscls": 0}
    summary_metrics = summary.get("metrics", {}) or {}

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
        img_result = ImageResult(
            image_path=img_path,
            model_idx=0,
            predictions=preds,
            has_gt=has_gt_any,
            tp=int(summary_metrics.get("tp", 0)),
            fp=int(summary_metrics.get("fp", 0)),
            fn=int(summary_metrics.get("fn", 0)),
            miscls=int(summary_metrics.get("miscls", 0)),
            annotated_path=run_dir / "images" / (img_path.stem + ".png"),
        )
        results.setdefault(0, []).append(img_result)

    state.results = results
    state.images = images
    # Class names ride on the Instance objects themselves («class» column),
    # so the report's per-class sections resolve without extra bookkeeping.
    # Only the first slot alias is re-pointed to the run's model name.
    state.model_slots[0].alias = "model"
    return state
