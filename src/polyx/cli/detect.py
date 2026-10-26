"""Headless batch detection CLI for Poly-X.

Usage:
    polyx-detect --model best.pt --images ./photos --out ./run --conf 0.25 --measure-shape
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
    p.add_argument("--device", default="0")
    p.add_argument("--um-per-px", type=float, default=0.0)
    p.add_argument("--gt-folder", default=None)
    p.add_argument("--no-measure-shape", action="store_true")
    p.add_argument("--full-report", action="store_true", help="also build paper-quality HTML (requires PySide6)")
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
    if args.full_report:
        try:
            import importlib
            qt = importlib.import_module("polyx.gui.qt")
            qt.require_qt()
            state_mod = importlib.import_module("polyx.gui.detector.state")
            DetectorState = state_mod.DetectorState
            ImageResult = state_mod.ImageResult
            InferenceParams = state_mod.InferenceParams
            Detection = importlib.import_module("polyx.core.detect.yolo").Detection
            generate_report = importlib.import_module("polyx.gui.report_html").generate_report
            import csv
            params = InferenceParams(conf=args.conf, iou_nms=args.iou, iou_tp=0.5,
                                    imgsz=args.imgsz, device=args.device, um_per_px=args.um_per_px,
                                    measure_shape=not args.no_measure_shape)
            state = DetectorState()
            state.params = params
            state.run_dir = out_dir
            # Build one ImageResult per image from centroids.csv
            by_image: dict = {}
            cp = out_dir / "centroids.csv"
            if cp.exists():
                with open(cp, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        by_image.setdefault(r["image"], []).append(r)
            results = {}
            for idx, img_path in enumerate(sorted(Path(args.images).rglob("*"))):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
                    continue
                preds = []
                for r in by_image.get(img_path.name, []):
                    preds.append(Detection(
                        class_id=0, class_name=r["class"], conf=float(r["conf"]),
                        x1=float(r["x1"]), y1=float(r["y1"]), x2=float(r["x2"]), y2=float(r["y2"]),
                        diam_um=float(r["diam_um"]) if r.get("diam_um") not in (None, "") else None,
                    ))
                img_result = ImageResult(image_path=img_path, model_idx=0, predictions=preds,
                                 has_gt=False, annotated_path=out_dir / "images" / (img_path.stem + ".png"))
                results.setdefault(0, []).append(img_result)
            state.results = results
            state.images = [p for p in sorted(Path(args.images).rglob("*"))
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")]
            generate_report(state, out_dir / "report.html")
            print(f"FULL REPORT: {out_dir / 'report.html'}")
        except Exception as exc:
            print(f"Full report unavailable ({exc}); writing compact report.")
            from polyx.core.report_simple import generate as _gen
            _gen(out_dir, out_dir / "report.html")
            print(f"COMPACT REPORT: {out_dir / 'report.html'}")
    else:
        from polyx.core.report_simple import generate as _gen
        _gen(out_dir, out_dir / "report.html")
        print(f"COMPACT REPORT: {out_dir / 'report.html'}")

    print(f"IMAGES: {summary['images']}  DETECTIONS: {summary['detections']}")
    print(f"OUTPUT: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())