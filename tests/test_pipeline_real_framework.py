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


def test_a_real_ultralytics_network_trains_two_epochs_after_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validation must not freeze the live network: training continues."""
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
            "protocol.epochs=2",
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
    assert state["epochs_completed"] == 2
    assert state["weights"] == "yolov8n.yaml"
    assert state["weights_verified"] is False

    # Both epochs were scored by the shared evaluator.
    log = (output / "logs" / "training.log").read_text(encoding="utf-8")
    assert "epoch=0 with_class_score=" in log
    assert "epoch=1 with_class_score=" in log

    # The exported checkpoints are real Ultralytics networks, and the second
    # one has different weights: Ultralytics' predictor in-place fusion used to
    # freeze the training network at the first validation, so the second
    # backward pass either failed or left the weights untouched.
    import torch

    first = torch.load(
        output / "checkpoints" / "epoch_000.pt", map_location="cpu", weights_only=False
    )["model"]
    second = torch.load(
        output / "checkpoints" / "epoch_001.pt", map_location="cpu", weights_only=False
    )["model"]
    assert getattr(first, "nc", None) == 3 or (getattr(first, "yaml", {}) or {}).get("nc") == 3
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    assert any(
        not torch.equal(first_state[name].float(), second_state[name].float())
        for name in first_state
        if first_state[name].is_floating_point()
    ), "validation froze the training network"
    assert (output / "checkpoints" / "best.pt").is_file()
    metrics = _read(output / "metrics.json")
    assert metrics["selection"]["metric"] == "with_class_score"
    assert 0.0 <= metrics["selection"]["value"] <= 1.0


def test_real_box_training_reads_same_size_edits_without_touching_stale_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.nn.tasks import DetectionModel

    from polyx.pipeline.command import main

    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    train = data_yaml.parent / "images" / "train"
    # Reproduce a user who trained with the native loader before using Poly-X.
    YOLODataset(img_path=str(train), imgsz=64, data={"names": {0: "PET", 1: "PP", 2: "LDPE"}})
    label_cache = data_yaml.parent / "labels" / "train.cache"
    before = label_cache.read_bytes()
    label = data_yaml.parent / "labels" / "train" / "plate_t1.txt"
    original = label.read_text(encoding="utf-8")
    edited = original.replace("0.375", "0.625", 1)
    assert len(edited) == len(original)
    label.write_text(edited, encoding="utf-8")
    image_cache = train / "plate_t1.npy"
    np.save(image_cache, np.full((48, 64, 3), 255, dtype=np.uint8))
    cached_image = image_cache.read_bytes()
    validation = data_yaml.parent / "images" / "val" / "plate_v1.png"
    validation.rename(validation.with_suffix(".PNG"))

    native_loss = DetectionModel.loss
    pet_centers = []
    source_peaks = []

    def observe_targets(self, batch, *args, **kwargs):
        pet = (batch["cls"].flatten() == 0).nonzero().flatten()
        pet_centers.extend(batch["bboxes"][pet, 0].tolist())
        samples = batch["batch_idx"][pet].long()
        source_peaks.extend(batch["img"][samples].amax(dim=(1, 2, 3)).tolist())
        return native_loss(self, batch, *args, **kwargs)

    monkeypatch.setattr(DetectionModel, "loss", observe_targets)
    output = tmp_path / "execution"
    code = main(
        [
            f"data_yaml={data_yaml}",
            f"output_dir={output}",
            "family.weights=yolov8n.yaml",
            "protocol.epochs=1",
            "protocol.batch=2",
            "protocol.image_size=64",
            "protocol.amp.enabled=false",
            "protocol.augmentation_subset=[]",
        ]
    )

    assert code == 0, capsys.readouterr().err
    assert pet_centers == pytest.approx([0.625])
    assert len(source_peaks) == 1 and source_peaks[0] < 1.0
    assert label_cache.read_bytes() == before
    assert image_cache.read_bytes() == cached_image
    provenance = _read(output / "provenance.json")
    assert "images/val/plate_v1.PNG" in provenance["dataset_hashes"]
    assert provenance["dataset_audit"]["per_class"]["val"]["PET"] == 1
