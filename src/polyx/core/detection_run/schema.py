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
from polyx.core.detect.tiling import TilingMode, TilingPlan
from polyx.core.measure.calibration import ORIGIN_NONE, Calibration
from polyx.core.measure.warnings import warning_from_dict, warning_to_dict
from polyx.core.quality.metrics import MatchResult, match_image

from .result import ImageResult, ModelExecution, ModelInfo, RunParams, RunResult, RunStatus

SCHEMA_VERSION = 2

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
    "assignable",
)

# The Results page exports the columns it can fill from the review state: the
# same names, in the canonical order, without the five that need the run
# folder (``number``, ``class_id``, ``size_source``, ``has_mask``,
# ``assignable``). A shorter view of one table, never a second vocabulary.
PAGE_COLUMNS = (
    "model",
    "image",
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
)

# The Viewer exports a particle from a photo no run owns. Nine of its ten
# columns are the canonical tokens above; ``diam_px`` is the box descriptor
# documented in CLAUDE.md, which only the Viewer publishes. Its values follow
# the schema too: ``size_source`` is ``mask`` or ``box``, never translated.
VIEWER_COLUMNS = (
    "class",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "diam_px",
    "diam_um",
    "area_um2",
    "size_source",
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
    """Plain JSON form; the tiling mode always leaves as its English value.

    ``TilingMode`` is a ``str``, so ``asdict`` plus ``json`` already wrote a
    string. Converting explicitly keeps that true for a caller that assigned a
    plain string after construction.
    """
    data = asdict(params)
    data["tiling"] = TilingMode.parse(data.get("tiling")).value
    return data


def params_from_dict(data: dict | None) -> RunParams:
    """Rebuild ``RunParams`` from JSON, ignoring unknown/missing fields.

    Only the English modes are read. An unknown value — including a legacy
    spelling — falls back to ``AUTO`` with a warning; it is never translated to
    a different English mode.
    """
    known = {field.name for field in fields(RunParams)}
    values = {key: value for key, value in (data or {}).items() if key in known}
    if "tiling" in values:
        values["tiling"] = TilingMode.parse(values["tiling"])
    return RunParams(**values)


def calibration_to_dict(calibration: Calibration) -> dict:
    data = asdict(calibration)
    # When the plate was measured, ``Calibration.valid`` tests a numpy float
    # and yields ``numpy.bool_``, which is not JSON serialisable.
    data["valid"] = bool(calibration.valid)
    # The artifact uses the English machine token; the Python field keeps its
    # historic name so existing callers of Calibration are unaffected.
    data["um_per_px"] = data.pop("um_por_px")
    # A warning leaves as a stable code plus structured parameters, never as
    # the sentence a person reads. ``asdict`` already expands the dataclass;
    # writing it explicitly keeps only the parameters that occurred.
    data["warning"] = warning_to_dict(calibration.warning)
    return data


def calibration_from_dict(data: dict | None) -> Calibration:
    known = {field.name for field in fields(Calibration)}
    values = {}
    for key, value in (data or {}).items():
        if key == "um_per_px":
            values["um_por_px"] = value
        elif key in known:
            values[key] = value
    # Keep the warning out of ``**values``: widening that dict's value type to
    # include ``Warning | None`` would make ``ty`` reject every other field.
    warning = warning_from_dict(values.pop("warning", None))
    return Calibration(**values, warning=warning)


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
        "warning_shape": warning_to_dict(getattr(det, "warning_shape", None)),
        "has_mask": det.mask is not None,
    }


def _instance_from_mapping(data: dict) -> Instance:
    mapping = dict(data)
    if "class" in mapping and "class_name" not in mapping:
        mapping["class_name"] = mapping.pop("class")
    if "warning_shape" in mapping:
        mapping["warning_shape"] = warning_from_dict(mapping["warning_shape"])
    known = {field.name for field in fields(Instance)}
    return Instance(**{key: value for key, value in mapping.items() if key in known})


def detection_from_dict(data: dict) -> Instance:
    return _instance_from_mapping(data)


def tiling_plan_to_dict(plan: TilingPlan | None) -> dict | None:
    """Plain-data form of a tiling plan; ``None`` means a single pass."""
    if plan is None:
        return None
    return {
        "tile": int(plan.tile),
        "overlap": float(plan.overlap),
        "n_tiles": int(plan.n_tiles),
        "side": int(plan.side),
        "threshold_px": int(plan.threshold_px),
    }


def tiling_plan_from_dict(data: object) -> TilingPlan | None:
    """Rebuild a recorded plan; a missing or malformed object reads as one piece.

    A folder written before the plan was recorded only has the legacy
    ``tiling`` sentence, and a folder whose plan is unusable must still open.
    """
    if not isinstance(data, dict):
        return None
    try:
        return TilingPlan(
            tile=int(data["tile"]),
            overlap=float(data["overlap"]),
            n_tiles=int(data["n_tiles"]),
            side=int(data["side"]),
            threshold_px=int(data["threshold_px"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


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


def _bool(value: object, *, default: bool = True) -> bool:
    """Read a persisted boolean; ``true`` and ``false`` are the only spellings.

    An empty cell keeps ``default``: runs written before abstention existed
    have no ``assignable`` column, and those detections were all assignable.
    Every other spelling — including the old ``1``, ``yes``, ``y``, ``si`` and
    ``sí`` — is rejected with an error naming the canonical format, so a stale
    artifact fails clearly instead of being silently translated.
    """
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"invalid boolean cell {value!r}: expected 'true' or 'false'")


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
        "assignable": bool(getattr(det, "assignable", True)),
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
        assignable=_bool(row.get("assignable")),
    )


def read_json_object(path: Path) -> dict:
    """Read a JSON object, returning ``{}`` for a missing or malformed file.

    Both the canonical reader and the compact/legacy report paths need the same
    forgiving read; keeping one helper stops their error handling drifting.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV into dict rows, tolerating a missing or header-only file."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_detections_csv(path: Path) -> list[dict]:
    """Read the prediction rows, tolerating a missing or header-only file."""
    return read_csv_rows(path)


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
        # Invariant boolean, not a translated label: the report renders the
        # label in the language of whoever is reading it (ADR-0004 permits a
        # flag, never prose).
        "alias_generated": bool(info.alias_generated),
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
                    # The rule behind those four numbers. Without it a reader
                    # cannot tell a box-matched run from a mask-matched one,
                    # nor reproduce the decision on reload.
                    "match_iou_thr": row.match_iou_thr,
                    "match_use_mask_iou": row.match_use_mask_iou,
                    # The invariant decision, not its sentence: the reader
                    # phrases it in their language through ``describe()``.
                    "tiling_plan": tiling_plan_to_dict(row.tiling_plan),
                    # Legacy sentence kept so an old reader still shows what a
                    # new folder did; new readers prefer ``tiling_plan``.
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
        "status": result.status.value,
        "started_at": isoformat(result.started_at),
        "finished_at": isoformat(finished_at),
        "duration_s": round(duration, 3),
        "provenance": provenance_data or {},
        "params": params_to_dict(result.params),
        "models": [_model_payload(result, index, metrics) for index in range(len(result.models))],
        "metrics": {str(index): values for index, values in metrics.items()},
        "calibrations": calibrations,
        "calibration_summary": result.calibration_summary,
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
        # Old folders have no flag: their alias was a name, shown verbatim.
        alias_generated=_bool(data.get("alias_generated"), default=False),
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


def _recorded_rule(model_row: dict, params: RunParams) -> tuple[float, bool]:
    """The matching rule a persisted row recorded.

    Folders written before #120 did not record it: they matched at
    ``params.iou_tp`` and masks were never persisted, so box IoU is the only
    rule those counters can have come from.
    """
    iou_thr = _float(model_row.get("match_iou_thr"))
    return (
        params.iou_tp if iou_thr is None else iou_thr,
        _bool(model_row.get("match_use_mask_iou"), default=False),
    )


def _match_from_record(
    predictions: list[Instance],
    gts: list[Instance],
    model_row: dict,
    iou_thr: float,
    use_mask_iou: bool,
) -> MatchResult:
    """Rebuild the match a row recorded, claiming only the identity it can prove.

    The stored counters are the run's own record. Recomputing with the
    recorded rule must reproduce them exactly for the pair indices to be the
    ones the run decided; when it does not -- a mask-matched run reopened from
    disk, where masks are gone -- the match keeps the counts and reports no
    indices rather than a second, contradicting decision.
    """
    recorded = MatchResult(
        tp=int(model_row.get("tp") or 0),
        fp=int(model_row.get("fp") or 0),
        fn=int(model_row.get("fn") or 0),
        miscls=int(model_row.get("miscls") or 0),
        has_identity=False,
    )
    recomputed = match_image(predictions, gts, iou_thr=iou_thr, use_mask_iou=use_mask_iou)
    if (recomputed.tp, recomputed.fp, recomputed.fn, recomputed.miscls) == (
        recorded.tp,
        recorded.fp,
        recorded.fn,
        recorded.miscls,
    ):
        return recomputed
    return recorded


def load_run(run_dir: Path) -> RunResult:
    """Reconstruct a :class:`RunResult` from a canonical folder, without Qt."""
    run_dir = Path(run_dir)
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise FileNotFoundError(f"not a canonical run folder: {run_json}")
    data = json.loads(run_json.read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        detail = (
            "version 1 is the old run format and is not supported"
            if version == 1
            else f"expected schema version {SCHEMA_VERSION}"
        )
        raise ValueError(
            f"unsupported run schema version {version!r}: {detail}; "
            "see docs/specs/run-artifact-vocabulary.md"
        )

    models = [_model_from_payload(entry) for entry in data.get("models") or []]
    params = params_from_dict(data.get("params"))
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
            predictions = [detection_from_row(row) for row in model_rows]
            row_has_gt = bool(model_row.get("has_gt", has_gt))
            match_iou_thr, match_use_mask_iou = _recorded_rule(model_row, params)
            results[model_idx].append(
                ImageResult(
                    image_path=image_path,
                    model_idx=model_idx,
                    predictions=predictions,
                    gt=gt,
                    has_gt=row_has_gt,
                    tp=int(model_row.get("tp") or 0),
                    fp=int(model_row.get("fp") or 0),
                    fn=int(model_row.get("fn") or 0),
                    miscls=int(model_row.get("miscls") or 0),
                    match=(
                        _match_from_record(
                            predictions, gt, model_row, match_iou_thr, match_use_mask_iou
                        )
                        if row_has_gt
                        else None
                    ),
                    match_iou_thr=match_iou_thr,
                    match_use_mask_iou=match_use_mask_iou,
                    annotated_path=run_view_path(folder, image_index, image_path.stem, "_annot"),
                    pred_path=run_view_path(folder, image_index, image_path.stem, "_pred"),
                    gt_path=gt_path if has_gt and gt_path.is_file() else None,
                    tiling_plan=tiling_plan_from_dict(model_row.get("tiling_plan")),
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
        params=params,
        calibrations=calibrations,
        results=results,
        status=RunStatus.parse(data.get("status")),
        error=data.get("error"),
        skipped_images=skipped,
        started_at=parse_datetime(data.get("started_at")),
        finished_at=parse_datetime(data.get("finished_at")),
        execution=execution,
        metrics=metrics,
    )


__all__ = [
    "DETECTION_COLUMNS",
    "PAGE_COLUMNS",
    "SCHEMA_VERSION",
    "VIEWER_COLUMNS",
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
    "read_csv_rows",
    "read_detections_csv",
    "read_json_object",
    "tiling_plan_from_dict",
    "tiling_plan_to_dict",
    "utcnow",
]
