"""Checkpoint metadata helpers for the report."""

from __future__ import annotations

from pathlib import Path


def _training_of(weight: Path) -> dict | None:
    """Reads from the checkpoint with which configuration and result the model
    was trained.

    Ultralytics keeps ``train_args`` and ``train_metrics`` inside the .pt
    itself, so the report can declare imgsz, batch, epochs and metrics
    without requiring the run folder to still exist.
    """
    try:
        import torch

        ck = torch.load(str(weight), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    args = ck.get("train_args") or {}
    met = ck.get("train_metrics") or {}
    if not args and not met:
        return None
    return {
        "base": args.get("model", "—"),
        "imgsz": args.get("imgsz", "—"),
        "batch": args.get("batch", "—"),
        "epocas": args.get("epochs", "—"),
        "optimizador": args.get("optimizer", "—"),
        "lr0": args.get("lr0", "—"),
        "amp": args.get("amp", "—"),
        "precision": met.get("metrics/precision(B)"),
        "recall": met.get("metrics/recall(B)"),
        "map50": met.get("metrics/mAP50(B)"),
        "map": met.get("metrics/mAP50-95(B)"),
    }



