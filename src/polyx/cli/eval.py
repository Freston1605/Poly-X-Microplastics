"""Headless evaluation CLI for Poly-X models.

Usage:
    polyx-eval --weights best.pt --data data.yaml --split val --output metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyx.core.evaluation import EvalConfig, evaluate, write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a Poly-X YOLO model (headless)")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, dest="data_yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--device", default=None, help="'cpu', a CUDA index, or empty = auto-select (default: auto)"
    )
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--output", default="metrics.json")
    args = p.parse_args(argv)

    cfg = EvalConfig(
        weights=args.weights,
        data_yaml=args.data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        iou=args.iou,
    )

    def on_log(msg):
        print(msg, flush=True)

    try:
        metrics = evaluate(cfg, on_log=on_log)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out = Path(args.output)
    write_report(metrics, out)
    print(f"WROTE: {out}")
    print(f"mAP50: {metrics.get('mAP50')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
