"""One local training execution driven only through the headless command.

Lightning and Ultralytics are replaced at the import boundary so that the
command, Hydra config resolution, shared evaluator, provenance and artifact
layout are the real code under test. The dataset is a tiny YOLO box dataset
built with the suite's split helper; nothing reaches the network or a GPU.

Every assertion reads a file the command wrote or the value it returned. No
private helper of the pipeline is imported.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from polyx.core.provenance import sha256_file
from tests.conftest import split_dataset

NAMES = {0: "PET", 1: "PP", 2: "LDPE"}


def _boxes_prediction() -> list[dict]:
    """Two correct boxes on each tiny 80x80 image (class ids 0 and 1)."""
    return [
        {"image_id": 0, "category_id": 0, "bbox": [20.0, 20.0, 20.0, 20.0], "score": 0.9},
        {"image_id": 0, "category_id": 1, "bbox": [50.0, 20.0, 20.0, 20.0], "score": 0.8},
        {"image_id": 1, "category_id": 2, "bbox": [20.0, 50.0, 20.0, 20.0], "score": 0.85},
    ]


def _empty_prediction() -> list[dict]:
    return []


class _FakeBoxes:
    def __init__(self, predictions: list[dict]) -> None:
        if predictions:
            self.xyxy = np.array(
                [
                    [
                        p["bbox"][0],
                        p["bbox"][1],
                        p["bbox"][0] + p["bbox"][2],
                        p["bbox"][1] + p["bbox"][3],
                    ]
                    for p in predictions
                ],
                dtype=float,
            )
            self.conf = np.array([p["score"] for p in predictions], dtype=float)
            self.cls = np.array([p["category_id"] for p in predictions], dtype=int)
        else:
            self.xyxy = np.zeros((0, 4), dtype=float)
            self.conf = np.zeros((0,), dtype=float)
            self.cls = np.zeros((0,), dtype=int)
        self.ap_class_index = [0, 1, 2]
        self.p = [0.9, 0.8, 0.85]
        self.r = [0.9, 0.8, 0.85]
        self.ap50 = [0.9, 0.8, 0.85]
        self.map50 = 0.85
        self.map = 0.7


class _FakeResult:
    def __init__(self, predictions: list[dict]) -> None:
        self.boxes = _FakeBoxes(predictions)
        self.names = dict(NAMES)
        self.box = self.boxes


class _FakeYOLO:
    """Ultralytics stand-in: one prediction set per validation call."""

    prediction_sets: list[list[dict]] = []
    instances: list[_FakeYOLO] = []

    def __init__(self, weights: object) -> None:
        self.weights = str(weights)
        self.names = dict(NAMES)
        self.model = types.SimpleNamespace(parameters=lambda: iter(()))
        _FakeYOLO.instances.append(self)

    def val(self, **_kwargs: object) -> list[_FakeResult]:
        per_image = _FakeYOLO.prediction_sets.pop(0)
        return [_FakeResult([p for p in per_image if p["image_id"] == i]) for i in range(2)]

    def save(self, path: object) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"checkpoint:{self.weights}:{target.name}".encode())


class _FakeLightningModule:
    """Base class the real adapter subclasses."""

    def __init__(self) -> None:
        self._epoch = 0

    @property
    def current_epoch(self) -> int:
        return self._epoch


class _FakeTrainer:
    """Runs epochs, invokes the module hook and the registered callback."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.max_epochs = int(kwargs.get("max_epochs", 1))
        self.precision = str(kwargs.get("precision", "32-true"))
        self.callbacks = list(kwargs.get("callbacks", []))
        self.should_stop = False
        self.current_epoch = 0

    def fit(self, module: object) -> None:
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            module._epoch = epoch
            module.on_validation_epoch_end()
            for callback in self.callbacks:
                callback.on_validation_epoch_end(self, module)
            self.current_epoch = epoch + 1
            if self.should_stop:
                break


class _FakeCallback:
    pass


def _install_fakes(monkeypatch: pytest.MonkeyPatch, prediction_sets: list[list[dict]]) -> None:
    """Put minimal Lightning and Ultralytics doubles in ``sys.modules``."""
    _FakeYOLO.prediction_sets = list(prediction_sets)
    _FakeYOLO.instances = []
    lightning = types.ModuleType("lightning")
    lightning.LightningModule = _FakeLightningModule
    lightning.Callback = _FakeCallback
    lightning.Trainer = _FakeTrainer
    lightning.seed_everything = lambda seed, **_kw: seed
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Two train and two val images with box labels, plus a data.yaml."""
    labels = {
        "train_a": "0 0.375 0.375 0.25 0.25\n1 0.75 0.375 0.25 0.25\n",
        "train_b": "2 0.375 0.75 0.25 0.25\n",
        "val_a": "0 0.375 0.375 0.25 0.25\n1 0.75 0.375 0.25 0.25\n",
        "val_b": "2 0.375 0.75 0.25 0.25\n",
    }
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {k: v for k, v in labels.items() if k.startswith("train")},
            "val": {k: v for k, v in labels.items() if k.startswith("val")},
        },
        names=("PET", "PP", "LDPE"),
        data_yaml=True,
    )
    images = root / "images"
    for split in ("train", "val"):
        for image in (images / split).glob("*.png"):
            # split_dataset writes 60x40 images; rewrite them at 80x80 so the
            # prediction boxes above de-normalise to whole pixel coordinates.
            import cv2

            ok, buffer = cv2.imencode(".png", np.full((80, 80, 3), 0, dtype=np.uint8))
            assert ok
            image.write_bytes(buffer.tobytes())
    return root / "data.yaml"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def test_local_tracer_emits_every_artifact_and_names_each_criterion(
    tmp_path: Path, tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command is the only seam; all proof is on disk."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _empty_prediction()])
    output = tmp_path / "executions" / "tracer"

    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=6",
        "protocol.early_stopping.patience=1",
    )

    assert code == 0
    for relative in (
        "config.yaml",
        "resolved_protocol.yaml",
        "provenance.json",
        "training_state.json",
        "metrics.json",
        "report.md",
        "logs/training.log",
        "checkpoints/best.pt",
    ):
        assert (output / relative).is_file(), f"missing artifact: {relative}"

    # The resolved protocol is written, consumed unchanged by the adapter and
    # echoed verbatim into the training state.
    from_command = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((output / "resolved_protocol.yaml").read_text(encoding="utf-8"))
    state = _read(output / "training_state.json")
    assert resolved == from_command["protocol"]
    assert state["protocol"] == resolved

    # Every figure names its criterion; the with-class score drives selection
    # and localisation is reported alongside it.
    metrics = _read(output / "metrics.json")
    assert metrics["selection"]["metric"] == "with_class_score"
    assert metrics["selection"]["value"] == pytest.approx(1.0)
    for name, figure in metrics["figures"].items():
        assert figure["criterion"], f"figure {name} has no criterion"
    assert "class" in metrics["figures"]["with_class_score"]["criterion"]
    assert "localisation" in metrics["figures"]["localisation_score"]["criterion"]
    assert metrics["figures"]["with_class_score"]["value"] == pytest.approx(1.0)
    assert 0.0 <= metrics["figures"]["localisation_score"]["value"] <= 1.0
    assert "adapter_cross_check" in metrics

    # Early stopping: the empty epoch did not improve the with-class score,
    # patience=1 stops after it and the best (first) weights are restored.
    stopping = state["early_stopping"]
    assert stopping["monitor"] == "with_class_score"
    assert stopping["mode"] == "max"
    assert stopping["patience"] == 1
    assert stopping["min_delta"] == pytest.approx(0.001)
    assert stopping["restore_best"] is True
    assert stopping["best_epoch"] == 0
    assert stopping["stopped_early"] is True
    assert state["epochs_completed"] == 2
    best = output / "checkpoints" / "best.pt"
    assert sha256_file(best) == sha256_file(output / "checkpoints" / "epoch_000.pt")
    assert sha256_file(best) != sha256_file(output / "checkpoints" / "epoch_001.pt")

    # FP16 AMP is configured and recorded as active.
    assert state["amp"]["enabled"] is True
    assert state["amp"]["dtype"] == "fp16"
    assert state["precision"] == "16-mixed"
    assert state["amp_active"] is True

    # Provenance: git commit and dataset hashes are recorded.
    provenance = _read(output / "provenance.json")
    assert provenance["git"]["commit"]
    assert "data.yaml" in provenance["dataset_hashes"]
    assert "images/val/val_a.png" in provenance["dataset_hashes"]
    assert provenance["dataset_hashes"]["images/val/val_a.png"] == sha256_file(
        tiny_dataset.parent / "images" / "val" / "val_a.png"
    )
    assert provenance["dataset_audit"]["ok"] is True


def test_patience_is_honoured_and_protocol_defaults_are_stable(
    tmp_path: Path, tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A larger patience keeps training past the first stagnant epoch."""
    _install_fakes(
        monkeypatch,
        [_boxes_prediction(), _empty_prediction(), _empty_prediction(), _boxes_prediction()],
    )
    output = tmp_path / "executions" / "patient"

    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=8",
        "protocol.early_stopping.patience=2",
    )

    assert code == 0
    state = _read(output / "training_state.json")
    assert state["early_stopping"]["patience"] == 2
    assert state["early_stopping"]["best_epoch"] == 0
    assert state["epochs_completed"] == 3
    assert state["early_stopping"]["stopped_early"] is True
    # Untouched defaults are the documented ones.
    protocol = yaml.safe_load((output / "resolved_protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["epochs"] == 8  # overridden above
    assert protocol["batch"] == 8
    assert protocol["image_size"] == 1280
    assert protocol["optimizer"] == "SGD"
    assert protocol["learning_rate"] == pytest.approx(0.01)
    assert protocol["momentum"] == pytest.approx(0.937)
    assert protocol["weight_decay"] == pytest.approx(0.0005)
    assert protocol["schedule"] == "cosine"
    assert protocol["warmup_epochs"] == 5
