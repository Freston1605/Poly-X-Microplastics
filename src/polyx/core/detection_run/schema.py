"""The canonical run folder schema and its Qt-free reader.

``run.json`` is the self-describing run record; ``detections.csv`` is the
predictions-only table. This module owns both shapes so the writer and
:func:`load_run` cannot drift apart, and it deliberately knows nothing about
Qt or the GUI review state.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

from polyx.core.annotate import run_view_path
from polyx.core.detect.protocol import Instance
from polyx.core.measure.calibration import ORIGIN_NONE, Calibration

from .result import ImageResult, ModelExecution, ModelInfo, RunParams, RunResult

SCHEMA_VERSION = 1

# Order is the published contract: the Viewer export and the compact report
# read it positionally only through the header, but users diff the header.
DETECTION_COLUMNS = (
    "model",
    "image",
    "number",
    "class_id",
    "class",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "um_per_px",
    "calibration_origin",
    "length_um",
    "method_length",
    "feret_um",
    "geodesic_um",
    "equivalent_rect_length_um",
    "width_um",
    "area_um2",
    "diam_um",
    "aspect",
    "curvature",
    "morphotype",
    "size_source",
    "has_mask",
)


# ── Small conversions ───────────────────────────────────────────────────


def utcnow() -> datetime:
    """Timezone-aware "now", so a run folder does not depend on the machine's zone."""
    return datetime.now(UTC)


def isoformat(value: datetime | None) -> str | None:
    """ISO-8601 for JSON, preserving ``None``."""
    return value.isoformat(timespec="seconds") if value is not None else None


def parse_datetime(value: object) -> datetime | None:
    """Read a timestamp written by :func:`isoformat` (or ``None``)."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def params_to_dict(params: RunParams) -> dict:
    return asdict(params)


def params_from_dict(data: dict | None) -> RunParams:
    """Rebuild ``RunParams`` from JSON, ignoring unknown/missing fields."""
    known = {field.name for field in fields(RunParams)}
    return RunParams(**{key: value for key, value in (data or {}).items() if key in known})


def calibration_to_dict(calibration: Calibration) -> dict:
    data = asdict(calibration)
    data["valid"] = calibration.valid
    return data


def calibration_from_dict(data: dict | None) -> Calibration:
    known = {field.name for field in fields(Calibration)}
    return Calibration(**{key: value for key, value in (data or {}).items() if key in known})


def detection_to_dict(det: Instance) -> dict:
    """Plain-data form of a detection for ``run.json`` (ground truth)."""
    return {
        "class_id": int(det.class_id),
        "class": det.class_name,
        "conf": float(det.conf),
        "x1": float(det.x1),
        "y1": float(det.y1),
        "x2": float(det.x2),
        "y2": float(det.y2),
        "diam_um": det.diam_um,
        "area_um2": det.area_um2,
        "length_um": det.length_um,
        "width_um": det.width_um,
        "aspect": det.aspect,
        "curvature": det.curvature,
        "morphotype": det.morphotype,
        "feret_um": det.feret_um,
        "geodesic_um": det.geodesic_um,
        "equivalent_rect_length_um": det.equivalent_rect_length_um,
        "method_length": det.method_length,
        "size_source": det.size_source,
        "number": det.number,
        "assignable": bool(getattr(det, "assignable", True)),
        "review": bool(getattr(det, "review", False)),
        "warning_shape": getattr(det, "warning_shape", None),
        "has_mask": det.mask is not None,
    }


def _instance_from_mapping(data: dict) -> Instance:
    mapping = dict(data)
    if "class" in mapping and "class_name" not in mapping:
        mapping["class_name"] = mapping.pop("class")
    known = {field.name for field in fields(Instance)}
    return Instance(**{key: value for key, value in mapping.items() if key in known})


def detection_from_dict(data: dict) -> Instance:
    return _instance_from_mapping(data)


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def detection_row(
    model_alias: str,
    image_path: Path,
    det: Instance,
    calibration: Calibration | None,
) -> dict:
    """One CSV row; empty cells mean "not available", never an invented zero."""
    return {
        "model": model_alias,
        "image": str(image_path),
        "number": det.number if det.number is not None else "",
        "class_id": int(det.class_id),
        "class": det.class_name,
        "conf": round(float(det.conf), 6),
        "x1": float(det.x1),
        "y1": float(det.y1),
        "x2": float(det.x2),
        "y2": float(det.y2),
        "um_per_px": calibration.um_por_px if calibration and calibration.valid else "",
        "calibration_origin": (calibration.origin if calibration is not None else ORIGIN_NONE),
        "length_um": det.length_um,
        "method_length": det.method_length,
        "feret_um": det.feret_um,
        "geodesic_um": det.geodesic_um,
        "equivalent_rect_length_um": det.equivalent_rect_length_um,
        "width_um": det.width_um,
        "area_um2": det.area_um2,
        "diam_um": det.diam_um,
        "aspect": det.aspect,
        "curvature": det.curvature,
        "morphotype": det.morphotype,
        "size_source": det.size_source,
        "has_mask": bool(det.mask is not None),
    }


def detection_from_row(row: dict) -> Instance:
    """Rebuild a prediction from ``detections.csv``."""
    size_source = row.get("size_source") or None
    return Instance(
        class_id=int(row.get("class_id") or 0),
        class_name=row.get("class") or "",
        conf=float(row.get("conf") or 0.0),
        x1=float(row.get("x1") or 0.0),
        y1=float(row.get("y1") or 0.0),
        x2=float(row.get("x2") or 0.0),
        y2=float(row.get("y2") or 0.0),
        diam_um=_float(row.get("diam_um")),
        area_um2=_float(row.get("area_um2")),
        length_um=_float(row.get("length_um")),
        width_um=_float(row.get("width_um")),
        aspect=_float(row.get("aspect")),
        curvature=_float(row.get("curvature")),
        morphotype=row.get("morphotype") or None,
        feret_um=_float(row.get("feret_um")),
        geodesic_um=_float(row.get("geodesic_um")),
        equivalent_rect_length_um=_float(row.get("equivalent_rect_length_um")),
        method_length=row.get("method_length") or None,
        size_source=size_source if size_source in ("mask", "box") else None,
        number=_int(row.get("number")),
    )


def read_detections_csv(path: Path) -> list[dict]:
    """Read the prediction rows, tolerating a missing or header-only file."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ── Aggregates and the run record ───────────────────────────────────────


def aggregate_metrics(result: RunResult) -> dict[int, dict[str, int]]:
    """Per-model totals, summed over that model's per-image results."""
    metrics: dict[int, dict[str, int]] = {}
    for index in range(len(result.models)):
        rows = result.results.get(index, [])
        metrics[index] = {
            "images": len(rows),
            "images_with_gt": sum(1 for row in rows if row.has_gt),
            "detections": sum(len(row.predictions) for row in rows),
            "tp": sum(row.tp for row in rows),
            "fp": sum(row.fp for row in rows),
            "fn": sum(row.fn for row in rows),
            "miscls": sum(row.miscls for row in rows),
        }
    return metrics


def _model_payload(result: RunResult, index: int, metrics: dict[int, dict[str, int]]) -> dict:
    info = result.models[index]
    execution = result.execution.get(index, ModelExecution())
    return {
        "index": index,
        "alias": info.alias,
        "path": str(info.path) if info.path is not None else None,
        "model_type": info.model_type,
        "names": {str(key): value for key, value in info.names.items()},
        "sha256": info.sha256,
        "execution": {
            "images": execution.images,
            "detections": execution.detections,
            "tiled_images": execution.tiled_images,
            "imgsz_used": list(execution.imgsz_used),
            "memory_fallback": list(execution.memory_fallback),
            "shapeless": execution.shapeless,
        },
        "metrics": metrics.get(index, {}),
    }


def _image_payloads(result: RunResult) -> list[dict]:
    """One entry per selected image, with per-model counts and ground truth."""
    by_image_model: dict[tuple[Path, int], ImageResult] = {}
    for model_idx, rows in result.results.items():
        for row in rows:
            by_image_model[(row.image_path, model_idx)] = row

    skipped = set(result.skipped_images)
    payloads: list[dict] = []
    for index, image_path in enumerate(result.images):
        model_rows: list[dict] = []
        gt: list[dict] = []
        for model_idx in range(len(result.models)):
            row = by_image_model.get((image_path, model_idx))
            if row is None:
                continue
            if row.has_gt and not gt:
                gt = [detection_to_dict(det) for det in row.gt]
            model_rows.append(
                {
                    "model": model_idx,
                    "has_gt": row.has_gt,
                    "predictions": len(row.predictions),
                    "tp": row.tp,
                    "fp": row.fp,
                    "fn": row.fn,
                    "miscls": row.miscls,
                    "tiling": row.tiling,
                }
            )
        payloads.append(
            {
                "index": index,
                "path": str(image_path),
                "name": image_path.name,
                "has_gt": bool(model_rows and any(row["has_gt"] for row in model_rows)),
                "skipped": image_path in skipped,
                "gt": gt,
                "models": model_rows,
            }
        )
    return payloads


def build_run_payload(result: RunResult, provenance_data: dict | None = None) -> dict:
    """The complete ``run.json`` document for the current state of ``result``."""
    metrics = aggregate_metrics(result)
    finished_at = result.finished_at
    duration = (
        (finished_at - result.started_at).total_seconds()
        if finished_at is not None and result.started_at is not None
        else 0.0
    )
    calibrations = [
        {"image": str(path), **calibration_to_dict(calibration)}
        for path, calibration in result.calibrations.items()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": result.status,
        "started_at": isoformat(result.started_at),
        "finished_at": isoformat(finished_at),
        "duration_s": round(duration, 3),
        "provenance": provenance_data or {},
        "params": params_to_dict(result.params),
        "models": [_model_payload(result, index, metrics) for index in range(len(result.models))],
        "metrics": {str(index): values for index, values in metrics.items()},
        "calibrations": calibrations,
        "images": _image_payloads(result),
        "error": result.error,
        "skipped_images": [str(path) for path in result.skipped_images],
    }


# ── Reading a canonical folder ──────────────────────────────────────────


def _model_from_payload(data: dict) -> ModelInfo:
    names_raw = data.get("names") or {}
    names = {}
    for key, value in names_raw.items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    path = data.get("path")
    return ModelInfo(
        alias=str(data.get("alias") or ""),
        path=Path(path) if path else None,
        model_type=str(data.get("model_type") or "yolo"),
        names=names,
        sha256=str(data.get("sha256") or ""),
    )


def _execution_from_payload(data: dict | None) -> ModelExecution:
    payload = data or {}
    return ModelExecution(
        images=int(payload.get("images") or 0),
        detections=int(payload.get("detections") or 0),
        tiled_images=int(payload.get("tiled_images") or 0),
        imgsz_used=[int(value) for value in payload.get("imgsz_used") or []],
        memory_fallback=[str(value) for value in payload.get("memory_fallback") or []],
        shapeless=int(payload.get("shapeless") or 0),
    )


def _calibration_items(data: object) -> list[tuple[str, dict]]:
    if isinstance(data, dict):
        return [(str(key), value) for key, value in data.items() if isinstance(value, dict)]
    if not isinstance(data, list):
        return []
    items: list[tuple[str, dict]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if image:
            items.append((str(image), entry))
    return items


def _detection_rows_by_model_image(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("model") or ""), str(row.get("image") or "")), []).append(
            row
        )
    return grouped


def load_run(run_dir: Path) -> RunResult:
    """Reconstruct a :class:`RunResult` from a canonical folder, without Qt."""
    run_dir = Path(run_dir)
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise FileNotFoundError(f"not a canonical run folder: {run_json}")
    data = json.loads(run_json.read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported run schema version: {version!r}")

    models = [_model_from_payload(entry) for entry in data.get("models") or []]
    execution = {
        index: _execution_from_payload(entry.get("execution"))
        for index, entry in enumerate(data.get("models") or [])
    }

    raw_images = data.get("images") or []
    image_entries: list[dict] = []
    for position, entry in enumerate(raw_images):
        if isinstance(entry, str):
            image_entries.append({"index": position, "path": entry})
        elif isinstance(entry, dict):
            image_entries.append({"index": entry.get("index", position), **entry})
    images = [Path(str(entry.get("path") or "")) for entry in image_entries]

    calibrations: dict[Path, Calibration] = {}
    for image, payload in _calibration_items(data.get("calibrations")):
        calibrations[Path(image)] = calibration_from_dict(payload)

    skipped = [Path(str(path)) for path in data.get("skipped_images") or []]
    if not skipped:
        skipped = [Path(str(entry.get("path"))) for entry in image_entries if entry.get("skipped")]

    rows = _detection_rows_by_model_image(read_detections_csv(run_dir / "detections.csv"))
    results: dict[int, list[ImageResult]] = {index: [] for index in range(len(models))}
    for entry in image_entries:
        image_path = Path(str(entry.get("path") or ""))
        image_index = int(entry.get("index") or 0)
        has_gt = bool(entry.get("has_gt"))
        gt = [detection_from_dict(item) for item in entry.get("gt") or []]
        for model_row in entry.get("models") or []:
            model_idx = int(model_row.get("model") or 0)
            if model_idx < 0 or model_idx >= len(models):
                continue
            alias = models[model_idx].alias
            model_rows = rows.get((alias, str(image_path)), [])
            if not model_rows:
                # A numeric model column is tolerated as well.
                model_rows = rows.get((str(model_idx), str(image_path)), [])
            folder = run_dir / alias
            gt_path = run_view_path(folder, image_index, image_path.stem, "_gt")
            results[model_idx].append(
                ImageResult(
                    image_path=image_path,
                    model_idx=model_idx,
                    predictions=[detection_from_row(row) for row in model_rows],
                    gt=gt,
                    has_gt=bool(model_row.get("has_gt", has_gt)),
                    tp=int(model_row.get("tp") or 0),
                    fp=int(model_row.get("fp") or 0),
                    fn=int(model_row.get("fn") or 0),
                    miscls=int(model_row.get("miscls") or 0),
                    annotated_path=run_view_path(folder, image_index, image_path.stem, "_annot"),
                    pred_path=run_view_path(folder, image_index, image_path.stem, "_pred"),
                    gt_path=gt_path if has_gt and gt_path.is_file() else None,
                    tiling=str(model_row.get("tiling") or ""),
                )
            )

    metrics = {
        int(index): {key: int(value) for key, value in values.items()}
        for index, values in (data.get("metrics") or {}).items()
    }
    return RunResult(
        run_dir=run_dir,
        images=images,
        models=models,
        params=params_from_dict(data.get("params")),
        calibrations=calibrations,
        results=results,
        status=str(data.get("status") or "completed"),
        error=data.get("error"),
        skipped_images=skipped,
        started_at=parse_datetime(data.get("started_at")),
        finished_at=parse_datetime(data.get("finished_at")),
        execution=execution,
        metrics=metrics,
    )


__all__ = [
    "DETECTION_COLUMNS",
    "SCHEMA_VERSION",
    "aggregate_metrics",
    "build_run_payload",
    "calibration_from_dict",
    "calibration_to_dict",
    "detection_from_dict",
    "detection_from_row",
    "detection_row",
    "detection_to_dict",
    "isoformat",
    "load_run",
    "params_from_dict",
    "params_to_dict",
    "parse_datetime",
    "read_detections_csv",
    "utcnow",
]
