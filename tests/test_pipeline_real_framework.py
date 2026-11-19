"""A real-framework spike, offline, through the same command seam.

The rest of the suite replaces Ultralytics at the import boundary; that proves
the wrapper's contract but not that a real Ultralytics network trains under the
Lightning loop. This test builds `YOLO("yolov8n.yaml")` from the vendored
architecture (no download), rebuilds its head for the fixture's three classes
and runs one real epoch and one real prediction pass on the tiny dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_pipeline_local_tracer import build_tiny_dataset


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_real_ultralytics_network_trains_one_epoch_on_the_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real Ultralytics, real Lightning, one batch of real training."""
    from polyx.pipeline.command import main

    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "real-ultralytics"

    code = main(
        [
            f"data_yaml={data_yaml}",
            f"output_dir={output}",
            # The architecture yaml needs no network: no weights are
            # downloaded and nothing is claimed to be hash-pinned.
            "family.weights=yolov8n.yaml",
            "protocol.epochs=1",
            "protocol.batch=2",
            "protocol.image_size=64",
            "protocol.warmup_epochs=0",
            "protocol.amp.enabled=false",
            "protocol.early_stopping.patience=99",
        ]
    )

    assert code == 0, capsys.readouterr().err
    state = _read(output / "training_state.json")
    assert state["framework"] == "ultralytics"
    assert state["epochs_completed"] == 1
    assert state["weights"] == "yolov8n.yaml"
    assert state["weights_verified"] is False

    # The training log records the shared evaluator's score for the epoch, so
    # the real loop really ran a training step and a prediction pass.
    log = (output / "logs" / "training.log").read_text(encoding="utf-8")
    assert "epoch=0 with_class_score=" in log

    # The exported checkpoint is the real Ultralytics network, sized for the
    # fixture's three classes, not a Lightning state dict and not a stub.
    import torch

    checkpoint = torch.load(
        output / "checkpoints" / "epoch_000.pt", map_location="cpu", weights_only=False
    )
    assert "model" in checkpoint
    network = checkpoint["model"]
    assert getattr(network, "nc", None) == 3 or (getattr(network, "yaml", {}) or {}).get("nc") == 3
    assert (output / "checkpoints" / "best.pt").is_file()
    metrics = _read(output / "metrics.json")
    assert metrics["selection"]["metric"] == "with_class_score"
    assert 0.0 <= metrics["selection"]["value"] <= 1.0
