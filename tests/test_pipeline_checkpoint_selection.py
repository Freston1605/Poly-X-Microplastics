"""Validation and best.pt must name the live weights, not a staged EMA."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tests.conftest import build_tiny_dataset


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_staged_ema_is_not_the_validated_or_kept_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A staged .pt with distinct model/EMA must not leak the EMA forward."""
    from ultralytics import YOLO

    from polyx.pipeline.command import main

    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    base = YOLO("yolov8n.yaml")
    ema = deepcopy(base.model)
    with torch.no_grad():
        for parameter in ema.parameters():
            parameter.add_(1.0)
    staged = tmp_path / "staged.pt"
    torch.save({"model": base.model, "ema": ema}, staged)

    output = tmp_path / "executions" / "ema-check"
    code = main(
        [
            f"data_yaml={data_yaml}",
            f"output_dir={output}",
            f"family.weights={staged}",
            "protocol.epochs=1",
            "protocol.batch=2",
            "protocol.image_size=64",
            "protocol.warmup_epochs=0",
            "protocol.amp.enabled=false",
            "protocol.early_stopping.patience=99",
        ]
    )
    assert code == 0, capsys.readouterr().err

    epoch = torch.load(
        output / "checkpoints" / "epoch_000.pt", map_location="cpu", weights_only=False
    )
    best = torch.load(output / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert "ema" not in epoch
    assert "ema" not in best
    assert "model" in epoch and "model" in best

    staged_ema = torch.load(staged, map_location="cpu", weights_only=False)["ema"].state_dict()
    kept = best["model"].state_dict()
    assert any(
        not torch.equal(value.float(), staged_ema[name].float())
        for name, value in kept.items()
        if name in staged_ema and value.is_floating_point()
    ), "best.pt still carries the staged EMA weights"

    metrics = _read(output / "metrics.json")
    assert metrics["selection"]["metric"] == "with_class_score"
    assert metrics["selection"]["confidence_threshold"] == 0.25
