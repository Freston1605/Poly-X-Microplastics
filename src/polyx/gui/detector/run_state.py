"""Rebuild a DetectorState from a detection run for report generation.

``build_state_from_run`` reads the old headless folder (``centroids.csv``,
``summary.json`` and ``annotated`` PNGs) and stays for pre-canonical runs.
``build_state_from_result`` is ticket #59's transitional adapter from a
canonical :class:`polyx.core.detection_run.RunResult`; ticket #63 replaces it
with direct ``RunResult`` rendering and it must be deleted then. Both live in
``polyx.gui`` — not in the CLI — so the no-Qt-in-CLI guards stay true.

Honesty note: the report sums ``tp/fp/fn`` per image, so those counts are
taken from ``summary["per_image"]`` (written by ``run_batch``) or from each
canonical ``ImageResult``, never from the run aggregate. An aggregate copied
into every row would be multiplied by the number of images.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from polyx.core.detection_run import IMAGE_EXTS, RunResult


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

    images = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    results: dict[int, list] = {}
    for img_path in images:
        preds = []
        for r in by_image.get(img_path.name, []):
            preds.append(  # noqa: PERF401
                Detection(
                    class_id=int(r.get("class_id", 0) or 0),
                    class_name=r.get("class", "0"),
                    conf=float(r.get("conf", 0.0) or 0.0),
                    x1=float(r["x1"]),
                    y1=float(r["y1"]),
                    x2=float(r["x2"]),
                    y2=float(r["y2"]),
                    diam_um=float(r["diam_um"]) if r.get("diam_um") not in (None, "") else None,
                )
            )
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


def build_state_from_result(result: RunResult):
    """Build a report state from a canonical :class:`RunResult`.

    TRANSITIONAL ADAPTER, delete in #63. The report still speaks
    ``DetectorState``; ticket #63 makes ``generate_report`` consume
    ``RunResult`` directly and this function has no reason to exist then. It
    deliberately copies data instead of adding a second run-folder reader: the
    canonical folder is already parsed by ``load_run``, and the CLI calls this
    with the result it just produced.
    """
    from polyx.gui.detector.state import DetectorState, ImageResult, ModelSlot

    state = DetectorState()
    state.params = result.params
    state.run_dir = result.run_dir
    state.images = list(result.images)
    state.calibrations = dict(result.calibrations_by_name)
    # The report keys results by the same dense indices as the engine, so the
    # slot list must be exactly the run's models, not the GUI's three slots.
    state.model_slots = [
        ModelSlot(
            alias=info.alias,
            path=info.path if info.path is not None else Path(info.alias),
            model_type=info.model_type,
            names=dict(info.names),
            sha256=info.sha256,
        )
        for info in result.models
    ]
    state.results = {
        index: [
            ImageResult(
                image_path=row.image_path,
                model_idx=row.model_idx,
                predictions=list(row.predictions),
                gt=list(row.gt),
                has_gt=row.has_gt,
                tp=row.tp,
                fp=row.fp,
                fn=row.fn,
                miscls=row.miscls,
                annotated_path=row.annotated_path,
                pred_path=row.pred_path,
                gt_path=row.gt_path,
                tiling=row.tiling,
            )
            for row in rows
        ]
        for index, rows in result.results.items()
    }
    return state
