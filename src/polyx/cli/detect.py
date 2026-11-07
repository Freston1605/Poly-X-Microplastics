"""Headless batch detection CLI for Poly-X.

Runs the canonical detection engine over a folder of images and writes the run
folder to the exact ``--out`` path. ``--model`` and ``--model-type`` are
repeatable: one type applies to every model, one per model must match, and no
type means YOLO. A non-empty ``--out`` is refused unless ``--force`` is given.

Calibration defaults to ``auto``: an explicit ``--index-calibration`` CSV,
then the index files searched from the image root, then the plate measured on
each photo, then ``--um-per-px``. ``--um-per-px`` alone selects ``manual``;
``--calibrate none`` refuses every scale.

Usage:
    polyx-detect --model best.pt --images ./photos --out ./run
    polyx-detect --model det.pt --model seg.pt --model-type yolo --model-type yolo-seg \\
        --images ./photos --out ./run
    polyx-detect --model best.pt --images ./photos --out ./run --gt-folder ./labels
    polyx-detect --model best.pt --images ./photos --out ./run --calibrate index \\
        --index-calibration ./indice.csv
    polyx-detect --model best.pt --images ./photos --out ./run --calibrate none
    polyx-detect --model best.pt --images ./photos --out ./run --tiling auto \\
        --tiling-tile 1024 --size-min-um 10 --conf-assignment 0.5
    polyx-detect --model best.pt --images ./photos --out ./run --full-report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyx.core.detect import make_detector
from polyx.core.detection_run import (
    CALIBRATION_MODES,
    CalibrationPolicy,
    ModelSpec,
    RunConfig,
    RunDirNotEmpty,
    RunFailed,
    RunOutputError,
    RunParams,
    RunResult,
    RunStatus,
    run_detection,
)
from polyx.core.i18n import _
from polyx.core.paths import find_images

# Exit codes: one for a failed run, one for bad usage, one for refusing to
# overwrite, one for an aborted run. The compact report path never touches Qt.
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_OUTPUT_EXISTS = 3
EXIT_ABORTED = 4


class _CalibrationAction(argparse.Action):
    """Store ``--calibrate`` and remember that it was given explicitly.

    ``--um-per-px`` implies ``manual`` only while nobody chose a mode: an
    explicit ``--calibrate auto`` keeps the typed value as its last fallback,
    and ``--calibrate none`` must not be overridden by it.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.calibrate_given = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_("Batch-detect microplastics (headless)"))
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="WEIGHTS",
        help=_("weights path; repeat once per model (aliases default to the file stem)"),
    )
    parser.add_argument(
        "--model-type",
        action="append",
        default=[],
        metavar="TYPE",
        help=_("detector type; repeat once per model, or give one value for all (default: yolo)"),
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True, help=_("exact run folder to write"))
    parser.add_argument("--force", action="store_true", help=_("overwrite a non-empty run folder"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--device",
        default=None,
        help=_("'cpu', a CUDA index, or empty = auto-select (default: auto)"),
    )
    parser.add_argument(
        "--calibrate",
        choices=CALIBRATION_MODES,
        default="auto",
        action=_CalibrationAction,
        help=_("per-photo calibration policy (default: auto)"),
    )
    parser.add_argument(
        "--um-per-px",
        type=float,
        default=0.0,
        help=_("manual µm/px; used alone it selects --calibrate manual"),
    )
    parser.add_argument(
        "--index-calibration",
        default=None,
        metavar="CSV",
        help=_("explicit index CSV mapping image names to µm/px"),
    )
    parser.add_argument("--gt-folder", default=None)
    parser.add_argument("--no-measure-shape", action="store_true")
    parser.add_argument(
        "--tiling",
        choices=("auto", "always", "never"),
        default="auto",
        help=_("tiling policy: auto by photo size, always, or never (default: auto)"),
    )
    parser.add_argument(
        "--tiling-threshold-px",
        type=int,
        default=2000,
        help=_("longest side above which --tiling auto splits the photo (default: 2000)"),
    )
    parser.add_argument(
        "--tiling-tile",
        type=int,
        default=0,
        help=_("tile side in px; 0 derives it from min(threshold, imgsz) (default: 0)"),
    )
    parser.add_argument(
        "--tiling-overlap",
        type=float,
        default=0.25,
        help=_("overlap between tiles, 0..0.9 (default: 0.25)"),
    )
    parser.add_argument("--size-min-um", type=float, default=0.0, help=_("drop particles below it"))
    parser.add_argument("--size-max-um", type=float, default=0.0, help=_("drop particles above it"))
    parser.add_argument(
        "--conf-assignment",
        type=float,
        default=0.0,
        help=_("below this confidence the particle is reported as unassignable (default: 0)"),
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help=_("run-order paper-quality HTML report (requires PySide6, [gui] extra)"),
    )
    return parser


def _resolve_model_types(requested: list[str], model_count: int) -> list[str]:
    """Apply the repeatable-type rule: none, one for all, or one per model."""
    types = [value.strip() or "yolo" for value in requested]
    if not types:
        return ["yolo"] * model_count
    if len(types) == 1:
        return types * model_count
    if len(types) != model_count:
        raise ValueError(
            _(
                "--model-type takes one value for all models or one per model "
                "(%(count)d); got %(got)d"
            )
            % {"count": model_count, "got": len(types)}
        )
    return types


def _load_models(weights: list[str], model_types: list[str]) -> list[ModelSpec]:
    """Build one loaded model per weights path; the engine never calls load()."""
    specs: list[ModelSpec] = []
    for weights_path, model_type in zip(weights, model_types, strict=True):
        path = Path(weights_path)
        try:
            detector = make_detector(model_type, path)
            detector.load()
        except Exception as exc:
            raise RuntimeError(f"{path}: {exc}") from exc
        specs.append(ModelSpec(detector=detector, path=path, model_type=model_type))
    return specs


def _resolve_calibration_mode(args) -> str:
    """Resolve the mode: an explicit ``--calibrate`` wins over the implication."""
    if getattr(args, "calibrate_given", False):
        return str(args.calibrate)
    if args.um_per_px > 0:
        return "manual"
    return str(args.calibrate)


def _backend_tiling(choice: str) -> str:
    """Translate the English CLI choice to the detector backend's values.

    ``predict_auto`` and ``RunParams`` keep the historical ``siempre`` /
    ``nunca`` values; only the developer-facing flag is English. This is a
    translation at the boundary, not a second tiling policy.
    """
    return {"always": "siempre", "never": "nunca"}.get(choice, "auto")


def _build_config(args, specs: list[ModelSpec]) -> RunConfig:
    params = RunParams(
        conf=args.conf,
        iou_nms=args.iou,
        imgsz=args.imgsz,
        measure_shape=not args.no_measure_shape,
        size_min_um=args.size_min_um,
        size_max_um=args.size_max_um,
        conf_assignment=args.conf_assignment,
        tiling=_backend_tiling(args.tiling),
        tiling_threshold_px=args.tiling_threshold_px,
        tiling_tile=args.tiling_tile,
        tiling_overlap=args.tiling_overlap,
    )
    if args.device is not None and str(args.device).strip():
        params.device = str(args.device)
    calibration = CalibrationPolicy(
        mode=_resolve_calibration_mode(args),
        um_per_px=args.um_per_px,
        index_path=Path(args.index_calibration) if args.index_calibration else None,
    )
    return RunConfig(
        images=find_images(Path(args.images)),
        models=specs,
        output_dir=Path(args.out),
        params=params,
        gt_folder=Path(args.gt_folder) if args.gt_folder else None,
        calibration=calibration,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        model_types = _resolve_model_types(args.model_type, len(args.model))
    except ValueError as exc:
        print(_("ERROR: %(error)s") % {"error": exc}, file=sys.stderr)
        return EXIT_USAGE

    try:
        specs = _load_models(args.model, model_types)
    except Exception as exc:  # noqa: BLE001
        print(_("ERROR loading model: %(error)s") % {"error": exc}, file=sys.stderr)
        return EXIT_ERROR

    try:
        result = run_detection(_build_config(args, specs))
    except RunOutputError as exc:
        hint = (
            _("Use --force to overwrite the canonical files.")
            if isinstance(exc, RunDirNotEmpty)
            else _("Choose a folder path for --out.")
        )
        print(_("ERROR: %(error)s. %(hint)s") % {"error": exc, "hint": hint}, file=sys.stderr)
        return EXIT_OUTPUT_EXISTS
    except RunFailed as exc:
        print(_("ERROR: %(error)s") % {"error": exc}, file=sys.stderr)
        print(_("PARTIAL RUN: %(run_dir)s") % {"run_dir": exc.result.run_dir}, file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        print(_("ERROR: %(error)s") % {"error": exc}, file=sys.stderr)
        return EXIT_ERROR

    if result.status is RunStatus.ABORTED:
        print(_("ABORTED: %(run_dir)s") % {"run_dir": result.run_dir}, file=sys.stderr)
        return EXIT_ABORTED

    _write_reports(args, result)
    detections = sum(
        len(image_result.predictions)
        for model_results in result.results.values()
        for image_result in model_results
    )
    print(
        _("IMAGES: %(images)d  DETECTIONS: %(detections)d")
        % {"images": len(result.images), "detections": detections}
    )
    print(_("OUTPUT: %(run_dir)s") % {"run_dir": result.run_dir})
    if result.skipped_images:
        print(_("SKIPPED: %(count)d") % {"count": len(result.skipped_images)}, file=sys.stderr)
    return 0


def _write_reports(args, result: RunResult) -> None:
    report_path = result.run_dir / "report.html"
    if args.full_report:
        try:
            _write_full_report(result, report_path)
        except Exception as exc:  # noqa: BLE001
            print(
                _("Full report unavailable (%(error)s); writing compact report.") % {"error": exc}
            )
            print(_("COMPACT REPORT: %(path)s") % {"path": _write_compact(result.run_dir)})
        else:
            print(_("FULL REPORT: %(path)s") % {"path": report_path})
    else:
        print(_("COMPACT REPORT: %(path)s") % {"path": _write_compact(result.run_dir)})


def _write_full_report(result: RunResult, report_path: Path) -> None:
    """Render the paper-quality report from the engine result.

    Qt and the report package are imported only on this path. The report
    consumes the ``RunResult`` the engine just produced; no ``DetectorState``
    is reconstructed for a new run.
    """
    import importlib

    qt = importlib.import_module("polyx.gui.qt")
    qt.require_qt()
    from polyx.gui.report_html import generate_report

    generate_report(result, report_path)


def _write_compact(run_dir: Path) -> Path:
    from polyx.core.report_simple import generate

    return generate(run_dir, run_dir / "report.html")


if __name__ == "__main__":
    raise SystemExit(main())
