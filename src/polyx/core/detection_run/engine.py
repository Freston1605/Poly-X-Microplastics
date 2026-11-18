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
from polyx.core.detect.tiling import TilingDecision
from polyx.core.i18n import _
from polyx.core.image_io import read_image
from polyx.core.labels import find_label_file, read_label_file
from polyx.core.measure.calibration import (
    ORIGIN_INDEX,
    ORIGIN_NONE,
    Calibration,
    read_calibration_index,
    resolver,
)
from polyx.core.measure.shape import measure_detections
from polyx.core.measure.warnings import (
    IMAGE_NOT_IN_INDEX,
    IMAGE_UNREADABLE,
    MANUAL_SCALE_MISSING,
    Warning,
)
from polyx.core.provenance import logger_for, provenance, sha256_file
from polyx.core.quality.assignment import mark_assignables
from polyx.core.quality.metrics import MatchResult, match_image

from .config import CALIBRATION_MODES, CalibrationPolicy, RunConfig
from .identity import resolve_identity
from .result import ImageResult, ModelExecution, ModelInfo, RunResult, RunStatus
from .schema import aggregate_metrics, utcnow
from .writer import RunWriter


class RunOutputError(RuntimeError):
    """The chosen output path cannot safely receive a run folder."""

    def __init__(self, path: Path, message: str):
        self.path = Path(path)
        super().__init__(message)


class RunDirNotEmpty(RunOutputError):
    """The caller chose an output folder that already has content."""

    def __init__(self, path: Path):
        super().__init__(
            path,
            _("output directory is not empty: %(path)s") % {"path": Path(path)},
        )


class RunOutputNotADirectory(RunOutputError):
    """The output path exists but it is a file, not a folder."""

    def __init__(self, path: Path):
        super().__init__(
            path,
            _("output path is not a directory: %(path)s") % {"path": Path(path)},
        )


class RunFailed(RuntimeError):
    """An unexpected error ended the run; ``result`` is the partial run record."""

    def __init__(self, result: RunResult, message: str | None = None):
        self.result = result
        super().__init__(message or result.error or _("detection run failed"))


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    """Create the run folder, refusing content chosen by someone else."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RunOutputNotADirectory(output_dir)
        if any(output_dir.iterdir()) and not force:
            raise RunDirNotEmpty(output_dir)
        return
    output_dir.mkdir(parents=True)


def _build_models(config: RunConfig) -> list[ModelInfo]:
    """Resolve aliases, class maps and fingerprints; one owner for the alias.

    Safety and uniqueness belong to :func:`resolve_identity`; the checks below
    stay as the engine's last line of defence, so a resolver regression fails
    the run instead of writing outside the run folder or into a sibling's.
    """
    infos: list[ModelInfo] = []
    seen: set[str] = set()
    for index, spec in enumerate(config.models):
        path = spec.path
        if path is None:
            weights = getattr(spec.detector, "weights_path", None)
            path = Path(weights) if weights else None
        identity = resolve_identity(typed=spec.alias, weights=path, index=index, taken=seen)
        alias = identity.alias
        if Path(alias).name != alias or alias in (".", ".."):
            raise ValueError(f"model alias is not a folder name: {alias!r}")
        if alias in seen:
            raise ValueError(f"duplicate model alias: {alias!r}")
        seen.add(alias)

        names = spec.names or getattr(spec.detector, "names", {}) or {}
        names = {int(key): str(value) for key, value in dict(names).items()}
        sha256 = spec.sha256
        if not sha256 and path is not None and Path(path).is_file():
            sha256 = sha256_file(path)
        infos.append(
            ModelInfo(
                alias=alias,
                # The GUI keeps its placeholder as a literal ``model_1``, so it
                # must declare it; the resolver only knows when it invented one.
                alias_generated=spec.alias_generated or identity.alias_generated,
                path=Path(path) if path is not None else None,
                model_type=spec.model_type or "yolo",
                names=names,
                sha256=sha256,
            )
        )
    return infos


def _validated_calibration_mode(policy: CalibrationPolicy) -> str:
    mode = str(policy.mode or "auto").strip().lower()
    if mode not in CALIBRATION_MODES:
        raise ValueError(
            f"unknown calibration mode: {policy.mode!r}; expected one of {CALIBRATION_MODES}"
        )
    return mode


def _load_calibration_index(policy: CalibrationPolicy, mode: str) -> dict[str, float]:
    """Load the per-image map once, before the run touches its output folder.

    Only ``index`` mode consults the file, and only at the path the caller
    named: there is no search and no legacy alias. A bad or absent file fails
    here with the reader's actionable message instead of silently scaling a
    crop with a value that belongs to a different photo.
    """
    if mode != "index":
        return {}
    if policy.index_path is None:
        raise ValueError(
            _("index calibration needs a calibration index CSV (--calibration-index PATH)")
        )
    return read_calibration_index(policy.index_path)


def _missing_calibration_warning(calibration: Calibration, code: str) -> Calibration:
    """Attach a mode-specific warning code when the resolver provided none."""
    if calibration.valid or calibration.warning:
        return calibration
    return replace(calibration, warning=Warning(code))


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

    if mode == "index":
        um_per_px = index.get(image_path.name)
        if um_per_px is None:
            # Never fall back to a batch-wide scale here: a crop whose name is
            # absent from the index has no known scale for THIS photo.
            return Calibration(
                origin=ORIGIN_NONE,
                diameter_mm=diameter_mm,
                warning=Warning(IMAGE_NOT_IN_INDEX),
            )
        return Calibration(um_por_px=um_per_px, origin=ORIGIN_INDEX, diameter_mm=diameter_mm)

    if mode == "manual":
        calibration = resolver(
            image_path,
            um_por_px_manual=policy.um_per_px,
            diameter_mm=diameter_mm,
            measure_plate=False,
        )
        if not calibration.valid and not calibration.warning:
            return _missing_calibration_warning(calibration, MANUAL_SCALE_MISSING)
        return calibration

    return resolver(
        image_path,
        um_por_px_manual=policy.um_per_px,
        diameter_mm=diameter_mm,
        measure_plate=policy.measure_plate,
        bgr=bgr,
        height_plate_mm=policy.height_plate_mm,
        distance_camera_mm=policy.distance_camera_mm,
    )


def measure_particles(
    detections: list[Instance], bgr: np.ndarray, um_per_px: float | None, params
) -> int:
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


def decide_match(
    predictions: list[Instance], gts: list[Instance], iou_thr: float
) -> tuple[MatchResult, bool]:
    """The run's matching decision for one model/image, and the rule it used.

    One owner for the criterion: whoever decides again over a set of labels --
    a completed run re-read, for instance -- must decide the way the run did.
    That rule is "mask IoU when either side carries a mask", so the returned
    flag is what the caller records next to the counts.
    """
    use_mask_iou = _uses_mask_iou(predictions, gts)
    match = match_image(predictions, gts, iou_thr=iou_thr, use_mask_iou=use_mask_iou)
    return match, use_mask_iou


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


def _predict(
    detector,
    image_path: Path,
    params,
    execution: ModelExecution,
) -> tuple[list[Instance], TilingDecision]:
    """Run the shared tiling entry point and keep the decision it reported."""
    record: dict = {}
    predictions = detector.predict_auto(
        image_path,
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
    decision = TilingDecision.from_record(record, params.tiling)
    plan = decision.plan
    if plan is not None:
        execution.tiled_images += 1
        # Each tile enters the network at its native tile side, not at imgsz.
        execution.record_imgsz(plan.tile)
    else:
        used_imgsz = int(getattr(detector, "last_imgsz", 0) or 0)
        execution.record_imgsz(used_imgsz or int(params.imgsz))
    if getattr(detector, "last_fallback", False):
        execution.memory_fallback.append(image_path.name)
    return list(predictions), decision


def _write_failed_run(
    output_dir: Path,
    result: RunResult,
    provenance_data: dict,
    log,
) -> None:
    """Best-effort partial ``run.json`` for a run that failed during setup."""
    try:
        RunWriter(output_dir, result, provenance_data).finish()
    except Exception:
        log.exception("run.finish_failed")


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
    mode = _validated_calibration_mode(config.calibration)
    # A missing or malformed index is a usage error: fail before creating the
    # run folder, so no partial run is left behind for a file we never read.
    index = _load_calibration_index(config.calibration, mode)

    images = [Path(path) for path in config.images]
    output_dir = Path(config.output_dir)

    try:
        provenance_data = provenance()
    except Exception:  # noqa: BLE001
        # Provenance is informational; a machine without git still runs.
        provenance_data = {}

    started_at = utcnow()
    log = logger_for("detector")

    # The output folder guard is an expected refusal, not a failed run: it
    # must not touch the folder the caller chose. Everything after it can
    # leave a partial run.json and a domain failure carrying the result.
    try:
        _prepare_output_dir(output_dir, config.force)
        models = _build_models(config)
    except RunOutputError:
        raise
    except Exception as exc:
        result = RunResult(
            run_dir=output_dir,
            images=images,
            params=config.params,
            status=RunStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            finished_at=utcnow(),
        )
        _write_failed_run(output_dir, result, provenance_data, log)
        raise RunFailed(result, result.error) from exc

    result = RunResult(
        run_dir=output_dir,
        images=images,
        models=models,
        params=config.params,
        status=RunStatus.RUNNING,
        started_at=started_at,
        results={index: [] for index in range(len(models))},
        execution={index: ModelExecution() for index in range(len(models))},
    )
    try:
        writer = RunWriter(output_dir, result, provenance_data)
    except Exception as exc:
        result.status = RunStatus.FAILED
        result.error = f"{type(exc).__name__}: {exc}"
        result.finished_at = utcnow()
        result.metrics = aggregate_metrics(result)
        _write_failed_run(output_dir, result, provenance_data, log)
        raise RunFailed(result, result.error) from exc

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
                result.status = RunStatus.ABORTED
                break

            bgr = read_image(image_path)
            if bgr is None:
                log.warning("image.undecodable file=%s", image_path)
                result.skipped_images.append(image_path)
                result.calibrations[image_path] = Calibration(
                    origin=ORIGIN_NONE,
                    diameter_mm=config.calibration.diameter_plate_mm,
                    warning=Warning(IMAGE_UNREADABLE),
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
                measure_particles(gts, bgr, um_per_px, config.params)

            for model_index, spec in enumerate(config.models):
                execution = result.execution[model_index]
                predictions, tiling_decision = _predict(
                    spec.detector, image_path, config.params, execution
                )
                shapeless = measure_particles(predictions, bgr, um_per_px, config.params)
                execution.shapeless += shapeless
                # Same order as the GUI runner: measure, filter, abstain, number,
                # then match. Abstention never drops a particle.
                predictions = _filter_by_size(predictions, um_per_px, config.params)
                mark_assignables(predictions, config.params.conf_assignment)
                _number_detections(predictions)

                if has_gt:
                    match, mask_iou_rule = decide_match(predictions, gts, config.params.iou_tp)
                    tp, fp, fn, miscls = match.tp, match.fp, match.fn, match.miscls
                else:
                    mask_iou_rule = False
                    match = None
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
                    match=match,
                    match_iou_thr=config.params.iou_tp,
                    match_use_mask_iou=mask_iou_rule,
                    tiling=tiling_decision.describe(),
                    tiling_plan=tiling_decision.plan,
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

        if result.status is not RunStatus.ABORTED:
            result.status = RunStatus.COMPLETED
        result.finished_at = utcnow()
        result.metrics = aggregate_metrics(result)
        writer.finish()
        return result
    except Exception as exc:
        result.status = RunStatus.FAILED
        result.error = f"{type(exc).__name__}: {exc}"
        result.finished_at = utcnow()
        result.metrics = aggregate_metrics(result)
        try:
            writer.finish()
        except Exception:
            log.exception("run.finish_failed")
        raise RunFailed(result, result.error) from exc


__all__ = [
    "RunDirNotEmpty",
    "RunFailed",
    "RunOutputError",
    "RunOutputNotADirectory",
    "run_detection",
]
