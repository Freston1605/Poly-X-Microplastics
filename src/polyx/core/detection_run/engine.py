"""The canonical, Qt-free detection run engine.

``run_detection`` is synchronous by design: the GUI owns the thread and turns
the callbacks into Qt signals. Callers load the models and pass the loaded
``DetectionSource`` objects in; the engine never calls ``load()``. It decodes
each photo once, reads ground truth once with the first model's class names,
measures predictions and GT (mask-aware), filters and numbers the predictions,
marks abstentions, matches them (mask IoU when either side carries a mask),
draws the triptych and writes the canonical folder as each result appears.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from polyx.core.annotate import triptych
from polyx.core.detect.protocol import Instance
from polyx.core.image_io import read_image
from polyx.core.labels import find_label_file, read_label_file
from polyx.core.measure.calibration import (
    ORIGIN_NONE,
    Calibration,
    carry_index,
    resolver,
    search_index,
)
from polyx.core.measure.shape import measure_detections
from polyx.core.provenance import logger_for, provenance, sha256_file
from polyx.core.quality.assignment import mark_assignables
from polyx.core.quality.metrics import match_image

from .config import CALIBRATION_MODES, CalibrationPolicy, RunConfig
from .result import ImageResult, ModelExecution, ModelInfo, RunResult
from .schema import aggregate_metrics, utcnow
from .writer import RunWriter

_STATUS_RUNNING = "running"
_STATUS_COMPLETED = "completed"
_STATUS_ABORTED = "aborted"
_STATUS_FAILED = "failed"


class RunDirNotEmpty(RuntimeError):
    """The caller chose an output folder that already has content."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.run_dir = self.path
        super().__init__(f"output directory is not empty: {self.path}")


class RunFailed(RuntimeError):
    """An unexpected error ended the run; ``result`` is the partial run record."""

    def __init__(self, result: RunResult, message: str | None = None):
        self.result = result
        self.run_result = result
        super().__init__(message or result.error or "detection run failed")


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    """Create the run folder, refusing content chosen by someone else."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RunDirNotEmpty(output_dir)
        if any(output_dir.iterdir()) and not force:
            raise RunDirNotEmpty(output_dir)
        return
    output_dir.mkdir(parents=True)


def _build_models(config: RunConfig) -> list[ModelInfo]:
    """Resolve aliases, class maps and fingerprints; reject ambiguous folders."""
    infos: list[ModelInfo] = []
    seen: set[str] = set()
    for index, spec in enumerate(config.models):
        alias = spec.folder_name(index)
        if Path(alias).name != alias or alias in (".", ".."):
            raise ValueError(f"model alias is not a folder name: {alias!r}")
        if alias in seen:
            raise ValueError(f"duplicate model alias: {alias!r}")
        seen.add(alias)

        names = spec.names or getattr(spec.detector, "names", {}) or {}
        names = {int(key): str(value) for key, value in dict(names).items()}
        path = spec.path
        if path is None:
            weights = getattr(spec.detector, "weights_path", None)
            path = Path(weights) if weights else None
        sha256 = spec.sha256
        if not sha256 and path is not None and Path(path).is_file():
            sha256 = sha256_file(path)
        infos.append(
            ModelInfo(
                alias=alias,
                path=Path(path) if path is not None else None,
                model_type=spec.model_type or "yolo",
                names=names,
                sha256=sha256,
            )
        )
    return infos


def _calibration_mode(policy: CalibrationPolicy) -> str:
    mode = str(policy.mode or "auto").strip().lower()
    if mode not in CALIBRATION_MODES:
        raise ValueError(
            f"unknown calibration mode: {policy.mode!r}; expected one of {CALIBRATION_MODES}"
        )
    return mode


def _calibration_index(
    policy: CalibrationPolicy,
    mode: str,
    images: list[Path],
) -> dict[str, float]:
    """Load the filename-to-scale map once for the whole run.

    An explicit path is authoritative: it is used alone, even if the file
    happens to be incomplete. Without one, the known index names are searched
    from the image root. That is the run-level half of the agreed order; the
    per-image half (index, then plate, then manual) lives in ``resolver``.
    """
    if mode not in ("auto", "index"):
        return {}
    if policy.index_path is not None:
        return carry_index(policy.index_path)
    return search_index(images[0]) if images else {}


def _resolve_calibration(
    image_path: Path,
    bgr: np.ndarray,
    policy: CalibrationPolicy,
    mode: str,
    index: dict[str, float],
) -> Calibration:
    """Scale of one photo, following the mode's documented source order."""
    diameter_mm = policy.diameter_plate_mm
    if mode == "none":
        return Calibration(origin=ORIGIN_NONE, diameter_mm=diameter_mm)

    if mode == "manual":
        calibration = resolver(
            image_path,
            um_por_px_manual=policy.um_per_px,
            diameter_mm=diameter_mm,
            measure_plate=False,
        )
        if not calibration.valid and not calibration.warning:
            return replace(
                calibration,
                warning="manual calibration needs a positive um_per_px",
            )
        return calibration

    if mode == "index":
        calibration = resolver(
            image_path,
            index=index,
            diameter_mm=diameter_mm,
            measure_plate=False,
        )
        if not calibration.valid and not calibration.warning:
            return replace(
                calibration,
                warning="image is not listed in the calibration index",
            )
        return calibration

    return resolver(
        image_path,
        index=index,
        um_por_px_manual=policy.um_per_px,
        diameter_mm=diameter_mm,
        measure_plate=policy.measure_plate,
        bgr=bgr,
        height_plate_mm=policy.height_plate_mm,
        distance_camera_mm=policy.distance_camera_mm,
    )


def _measure(detections: list[Instance], bgr: np.ndarray, um_per_px: float | None, params) -> int:
    """Box fallback first, then the current shape measurement; returns shapeless count."""
    return measure_detections(
        detections,
        bgr,
        um_per_px if um_per_px and um_per_px > 0 else None,
        measure_shape=params.measure_shape,
        masks=[getattr(det, "mask", None) for det in detections],
    )


def _uses_mask_iou(predictions: list[Instance], gts: list[Instance]) -> bool:
    """Whether either side carries a mask for this image.

    ``match_image`` still falls back to box IoU per pair, so one particle with
    a failed segmentation does not turn mask matching off for the whole image.
    """
    return any(getattr(det, "mask", None) is not None for det in predictions) or any(
        getattr(det, "mask", None) is not None for det in gts
    )


def _filter_by_size(detections: list[Instance], um_per_px: float | None, params) -> list[Instance]:
    """Drop particles outside the configured micrometre range.

    Both bounds at 0 mean no filter. An unmeasured particle is kept: losing it
    from the count is worse than reporting it without a size. The GT is never
    filtered; it is a property of the image, not of the model's threshold.
    """
    if not um_per_px or (params.size_min_um <= 0 and params.size_max_um <= 0):
        return detections
    kept: list[Instance] = []
    for det in detections:
        if det.diam_um is None:
            kept.append(det)
            continue
        if params.size_min_um > 0 and det.diam_um < params.size_min_um:
            continue
        if params.size_max_um > 0 and det.diam_um > params.size_max_um:
            continue
        kept.append(det)
    return kept


def _number_detections(detections: list[Instance]) -> None:
    """Number from top to bottom and left to right, not in model output order."""
    for number, det in enumerate(
        sorted(detections, key=lambda det: (round(det.y1 / 40), det.x1)), 1
    ):
        det.number = number


def _tiling_description(record: dict) -> str:
    """Human-readable tiling decision for one image; empty when it stayed whole."""
    if not record.get("troceado"):
        return ""
    plan = record.get("plan") or {}
    width = int(record.get("ancho", 0) or 0)
    height = int(record.get("alto", 0) or 0)
    tile = int(plan.get("tile", 0) or 0)
    tiles = int(plan.get("n_tiles", 0) or 0)
    overlap = float(plan.get("overlap", 0.0) or 0.0)
    return f"{width}×{height} px → {tiles} tiles of {tile} px, {int(overlap * 100)}% overlap"


def _predict(
    detector,
    image_path: Path,
    params,
    execution: ModelExecution,
) -> tuple[list[Instance], str]:
    """Run the shared tiling entry point and record what it decided."""
    record: dict = {}
    predictions = detector.predict_auto(
        str(image_path),
        conf=params.conf,
        iou=params.iou_nms,
        imgsz=params.imgsz,
        device=params.device,
        tiling=params.tiling,
        threshold_px=params.tiling_threshold_px,
        tile=params.tiling_tile,
        overlap=params.tiling_overlap,
        record=record,
    )
    if record.get("troceado"):
        execution.tiled_images += 1
        plan = record.get("plan") or {}
        # Each tile enters the network at its native tile side, not at imgsz.
        execution.record_imgsz(int(plan.get("tile", 0) or 0))
    else:
        used_imgsz = int(getattr(detector, "last_imgsz", 0) or 0)
        execution.record_imgsz(used_imgsz or int(params.imgsz))
    if getattr(detector, "last_fallback", False):
        execution.memory_fallback.append(image_path.name)
    return list(predictions), _tiling_description(record)


def run_detection(
    config: RunConfig,
    *,
    on_progress: Callable[[int, int, Path], None] | None = None,
    on_result: Callable[[ImageResult], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> RunResult:
    """Run every model over every image and return the canonical result.

    Callbacks: ``on_progress(done, total, image_path)`` counts model/image
    pairs; ``on_result(result)`` fires after that result's files are written.
    ``should_abort`` is consulted at image boundaries, never mid-image, so a
    partial run never contains a half-written image.
    """
    if not config.models:
        raise ValueError("a detection run needs at least one model")
    mode = _calibration_mode(config.calibration)

    images = [Path(path) for path in config.images]
    index = _calibration_index(config.calibration, mode, images)
    output_dir = Path(config.output_dir)
    models = _build_models(config)
    _prepare_output_dir(output_dir, config.force)

    try:
        provenance_data = provenance()
    except Exception:  # noqa: BLE001
        # Provenance is informational; a machine without git still runs.
        provenance_data = {}

    started_at = utcnow()
    result = RunResult(
        run_dir=output_dir,
        images=images,
        models=models,
        params=config.params,
        status=_STATUS_RUNNING,
        started_at=started_at,
        results={index: [] for index in range(len(models))},
        execution={index: ModelExecution() for index in range(len(models))},
    )
    writer = RunWriter(output_dir, result, provenance_data)
    log = logger_for("detector")
    total = len(images) * len(models)
    done = 0
    first_names = models[0].names if models else {}

    disagreeing = [model.alias for model in models[1:] if model.names != first_names]
    if disagreeing:
        log.warning(
            "GT class maps disagree with the first model %s; ground truth uses its names, "
            "differing models: %s",
            models[0].alias,
            ", ".join(disagreeing),
        )

    try:
        for image_index, image_path in enumerate(images):
            if should_abort is not None and should_abort():
                result.status = _STATUS_ABORTED
                break

            bgr = read_image(image_path)
            if bgr is None:
                log.warning("image.undecodable file=%s", image_path)
                result.skipped_images.append(image_path)
                result.calibrations[image_path] = Calibration(
                    origin=ORIGIN_NONE,
                    diameter_mm=config.calibration.diameter_plate_mm,
                    warning="the image could not be read",
                )
                writer.record_skip()
                done += len(models)
                if on_progress is not None:
                    on_progress(done, total, image_path)
                continue

            height, width = bgr.shape[:2]
            calibration = _resolve_calibration(image_path, bgr, config.calibration, mode, index)
            result.calibrations[image_path] = calibration
            um_per_px = calibration.um_por_px if calibration.valid else None

            # GT is a property of the image, not of a model: read and measure
            # it once, with the first model's class names, and reuse it.
            gt_path = find_label_file(image_path, config.gt_folder)
            has_gt = gt_path is not None
            gts: list[Instance] = []
            if gt_path is not None:
                gts, malformed = read_label_file(gt_path, width, height, first_names)
                if malformed:
                    log.warning("GT labels: %d unparseable line(s) in %s", malformed, gt_path.name)
                _measure(gts, bgr, um_per_px, config.params)

            for model_index, spec in enumerate(config.models):
                execution = result.execution[model_index]
                predictions, tiling = _predict(spec.detector, image_path, config.params, execution)
                shapeless = _measure(predictions, bgr, um_per_px, config.params)
                execution.shapeless += shapeless
                # Same order as the GUI runner: measure, filter, abstain, number,
                # then match. Abstention never drops a particle.
                predictions = _filter_by_size(predictions, um_per_px, config.params)
                mark_assignables(predictions, config.params.conf_assignment)
                _number_detections(predictions)

                if has_gt:
                    match = match_image(
                        predictions,
                        gts,
                        iou_thr=config.params.iou_tp,
                        use_mask_iou=_uses_mask_iou(predictions, gts),
                    )
                    tp, fp, fn, miscls = match.tp, match.fp, match.fn, match.miscls
                else:
                    tp = fp = fn = miscls = 0

                views = triptych(
                    bgr,
                    predictions,
                    gts,
                    pred_label_for=lambda det: f"{det.class_name} {det.conf:.2f}",
                    has_gt=has_gt,
                )
                image_result = ImageResult(
                    image_path=image_path,
                    model_idx=model_index,
                    predictions=predictions,
                    gt=gts,
                    has_gt=has_gt,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    miscls=miscls,
                    tiling=tiling,
                )
                result.results.setdefault(model_index, []).append(image_result)
                execution.images += 1
                execution.detections += len(predictions)

                writer.record_result(
                    image_result,
                    calibration,
                    views,
                    image_index,
                    width,
                    height,
                )
                if on_result is not None:
                    on_result(image_result)

                done += 1
                if on_progress is not None:
                    on_progress(done, total, image_path)

        if result.status != _STATUS_ABORTED:
            result.status = _STATUS_COMPLETED
        result.finished_at = utcnow()
        result.metrics = aggregate_metrics(result)
        writer.finish()
        return result
    except Exception as exc:
        result.status = _STATUS_FAILED
        result.error = f"{type(exc).__name__}: {exc}"
        result.finished_at = utcnow()
        result.metrics = aggregate_metrics(result)
        try:
            writer.finish()
        except Exception:
            log.exception("run.finish_failed")
        raise RunFailed(result, result.error) from exc


__all__ = ["RunDirNotEmpty", "RunFailed", "run_detection"]
