"""Read a pre-canonical headless run folder into a ``RunResult``.

The old headless writer stored ``summary.json`` (run parameters, aggregate and
per-image counts) and ``centroids.csv`` (one row per prediction) under a flat
``images/`` folder. That folder is still data a user may want to report on, so
this module rebuilds the canonical result without touching the folder, without
Qt and without inventing values the old writer never recorded.

Ground truth is the one unavoidable gap: the old writer kept the per-image
``tp/fp/fn/miscls`` counts but not the GT boxes. The rebuilt ``ImageResult``
therefore carries those counts and an empty ``gt`` list, which is exactly what
the old GUI loader could offer the report.
"""

from __future__ import annotations

from pathlib import Path

from polyx.core.measure.calibration import ORIGIN_MANUAL, Calibration
from polyx.core.paths import find_images

from .result import ImageResult, ModelExecution, ModelInfo, RunParams, RunResult, RunStatus
from .schema import (
    aggregate_metrics,
    detection_from_row,
    read_detections_csv,
    read_json_object,
)

LEGACY_SUMMARY = "summary.json"
LEGACY_CENTROIDS = "centroids.csv"
LEGACY_IMAGES = "images"


def _legacy_params(raw: dict) -> RunParams:
    """Map the old parameter names onto ``RunParams`` without inventing values."""
    params = RunParams()
    if raw.get("conf") is not None:
        params.conf = float(raw["conf"])
    iou = raw.get("iou", raw.get("iou_nms"))
    if iou is not None:
        params.iou_nms = float(iou)
    if raw.get("imgsz") is not None:
        params.imgsz = int(raw["imgsz"])
    if raw.get("device"):
        params.device = str(raw["device"])
    if raw.get("measure_shape") is not None:
        params.measure_shape = bool(raw["measure_shape"])
    return params


def _legacy_calibrations(images: list[Path], raw: dict) -> dict[Path, Calibration]:
    """Turn the old batch-wide manual scale into the effective per-image record.

    The old writer had one ``um_per_px`` for the whole batch, which is exactly
    the manual policy. The report reads effective calibrations, so keeping it
    here (and out of ``RunParams``) preserves the old numbers without letting
    the parameters claim a scale the run did not apply per photo.
    """
    try:
        um_per_px = float(raw.get("um_per_px") or 0.0)
    except (TypeError, ValueError):
        return {}
    if um_per_px <= 0:
        return {}
    return {image: Calibration(um_por_px=um_per_px, origin=ORIGIN_MANUAL) for image in images}


def _class_names(rows: list[dict]) -> dict[int, str]:
    names: dict[int, str] = {}
    for row in rows:
        try:
            class_id = int(row.get("class_id") or 0)
        except (TypeError, ValueError):
            continue
        name = str(row.get("class") or "").strip()
        if name:
            names.setdefault(class_id, name)
    return names


def _image_paths(run_dir: Path, image_dir: Path | None, names: list[str]) -> list[Path]:
    """Original photos when the caller still has them, otherwise their names."""
    if image_dir is not None:
        return find_images(Path(image_dir))
    return [run_dir / LEGACY_IMAGES / name for name in names]


def load_legacy_run(run_dir: Path, image_dir: Path | None = None) -> RunResult:
    """Rebuild a ``RunResult`` from a pre-canonical headless folder.

    ``image_dir`` is the folder the old batch ran on. When it is gone the
    result keeps the image names and points the annotated view at the old
    ``images/`` folder, so a report can still be produced.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / LEGACY_SUMMARY
    centroids_path = run_dir / LEGACY_CENTROIDS
    if not summary_path.is_file() and not centroids_path.is_file():
        raise FileNotFoundError(f"not a legacy headless run folder: {run_dir}")

    summary = read_json_object(run_dir / LEGACY_SUMMARY)
    params_raw = summary.get("params") or {}
    per_image = summary.get("per_image") or {}
    rows = read_detections_csv(centroids_path)

    by_image: dict[str, list[dict]] = {}
    for row in rows:
        by_image.setdefault(str(row.get("image") or ""), []).append(row)

    names = list(dict.fromkeys([*by_image, *[str(name) for name in per_image]]))
    images = _image_paths(run_dir, image_dir, names)
    seen = {path.name for path in images}
    if image_dir is not None:
        # Old runs keyed everything by file name; keep files the caller chose
        # even when they had no predictions and no GT.
        names = [path.name for path in images]
    else:
        missing = [name for name in names if name not in seen]
        images += [run_dir / LEGACY_IMAGES / name for name in missing]

    model = ModelInfo(
        alias="model",
        model_type=str(params_raw.get("model_type") or "yolo"),
        names=_class_names(rows),
    )

    results: dict[int, list[ImageResult]] = {0: []}
    for image_path in images:
        name = image_path.name
        counts = per_image.get(name) or {}
        predictions = [detection_from_row(row) for row in by_image.get(name, [])]
        results[0].append(
            ImageResult(
                image_path=image_path,
                model_idx=0,
                predictions=predictions,
                has_gt=bool(counts),
                tp=int(counts.get("tp", 0)),
                fp=int(counts.get("fp", 0)),
                fn=int(counts.get("fn", 0)),
                miscls=int(counts.get("miscls", 0)),
                annotated_path=run_dir / LEGACY_IMAGES / f"{Path(name).stem}.png",
            )
        )

    result = RunResult(
        run_dir=run_dir,
        images=images,
        models=[model],
        params=_legacy_params(params_raw),
        calibrations=_legacy_calibrations(images, params_raw),
        results=results,
        status=RunStatus.COMPLETED,
        execution={
            0: ModelExecution(
                images=len(images),
                detections=sum(len(rows) for rows in by_image.values()),
            )
        },
        legacy=True,
    )
    metrics = aggregate_metrics(result)
    legacy_metrics = summary.get("metrics") or {}
    if legacy_metrics:
        metrics[0].update(
            {
                key: int(legacy_metrics.get(key, metrics[0].get(key, 0)))
                for key in ("tp", "fp", "fn", "miscls")
            }
        )
    result.metrics = metrics
    return result


__all__ = ["LEGACY_CENTROIDS", "LEGACY_IMAGES", "LEGACY_SUMMARY", "load_legacy_run"]
