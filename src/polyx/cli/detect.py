"""Headless batch detection CLI for Poly-X.

Usage:
    polyx-detect --model best.pt --images ./photos --out ./run --conf 0.25 --measure-shape
    polyx-detect --model best.pt --images ./photos --out ./run --gt-folder ./labels
    polyx-detect --model best.pt --images ./photos --out ./run --full-report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyx.core.detect import make_detector
from polyx.core.detection_run import run_batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch-detect microplastics (headless)")
    p.add_argument("--model", required=True, help="weights path")
    p.add_argument("--model-type", default="yolo")
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None,
                   help="'cpu', a CUDA index, or empty = auto-select (default: auto)")
    p.add_argument("--um-per-px", type=float, default=0.0)
    p.add_argument("--gt-folder", default=None)
    p.add_argument("--no-measure-shape", action="store_true")
    p.add_argument("--full-report", action="store_true",
                   help="run-order paper-quality HTML report (requires PySide6, [gui] extra)")
    args = p.parse_args(argv)

    try:
        model = make_detector(args.model_type, args.model, alias="model")
        model.load()
    except Exception as exc:
        print(f"ERROR loading model: {exc}", file=sys.stderr)
        return 1

    try:
        summary = run_batch(
            model,
            Path(args.images),
            Path(args.out),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            um_per_px=args.um_per_px,
            measure_shape=not args.no_measure_shape,
            gt_folder=Path(args.gt_folder) if args.gt_folder else None,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    report_path = out_dir / "report.html"
    if args.full_report:
        # Report assembly lives in polyx.gui.detector.run_state; the single
        # deferred import keeps this CLI importable without Qt while the
        # builder itself imports the Qt state module directly.
        try:
            import importlib
            qt = importlib.import_module("polyx.gui.qt")
            qt.require_qt()
            from polyx.gui.report_html import generate_report
            build_state = importlib.import_module(
                "polyx.gui.detector.run_state").build_state_from_run
            state = build_state(out_dir, Path(args.images))
            generate_report(state, report_path)
            print(f"FULL REPORT: {report_path}")
        except Exception as exc:
            print(f"Full report unavailable ({exc}); writing compact report.")
            print(f"COMPACT REPORT: {_write_compact(run_dir=out_dir)}")
    else:
        print(f"COMPACT REPORT: {_write_compact(run_dir=out_dir)}")

    print(f"IMAGES: {summary['images']}  DETECTIONS: {summary['detections']}")
    print(f"OUTPUT: {summary['output_dir']}")
    return 0


def _write_compact(run_dir: Path) -> Path:
    from polyx.core.report_simple import generate
    return generate(run_dir, run_dir / "report.html")


if __name__ == "__main__":
    raise SystemExit(main())
