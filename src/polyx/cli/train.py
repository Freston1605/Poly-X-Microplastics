"""Headless training CLI for Poly-X.

Usage:
    polyx-train --weights yolov8m.pt --data data.yaml --epochs 100 --batch 16 --imgsz 640
"""

from __future__ import annotations

import argparse
import sys

from polyx.core.training import TrainingConfig
from polyx.core.training import train as run_training


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a Poly-X YOLO model (headless)")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, dest="data_yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device", default=None, help="'cpu', a CUDA index, or empty = auto-select (default: auto)"
    )
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="train")
    args = p.parse_args(argv)

    cfg = TrainingConfig(
        weights=args.weights,
        data_yaml=args.data_yaml,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
    )

    def on_log(msg):
        print(msg, flush=True)

    def on_epoch(info):
        m = info.get("metrics", {})
        print(f"epoch {info.get('epoch')}: {m}", flush=True)

    try:
        best = run_training(cfg, on_epoch=on_epoch, on_log=on_log)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"BEST: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
