"""One Mask R-CNN execution driven only through the headless command seam.

The CVAT-shaped COCO export is imported as the canonical segmentation training
dataset (ticket #196), then the same command trains a torchvision Mask R-CNN
family from that dataset for one epoch (ticket #198). The torchvision model
factory is replaced at the framework boundary; the command, Hydra config
resolution, dataset reads, shared evaluator, provenance and artifact layout are
the real code under test.

Every assertion reads a file the command wrote or a value it returned. The two
segmentation families are compared through their artifacts, never through
pipeline internals.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.test_pipeline_yolo_seg import (
    NAMES,
    _imported_store,
    _perfect_validation,
    _shifted_validation,
)

# ── torchvision / Lightning doubles ───────────────────────────────────────


def _mask_for(polygon: list[float], width: int = 64, height: int = 48) -> np.ndarray:
    """Rasterize a fixture polygon the same way the canonical audit does."""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(polygon, dtype=float).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


class _FakeDetector:
    """torchvision Mask R-CNN stand-in with a real ``nn.Module`` base."""

    prediction_sets: list[list[list[dict]]] = []
    instances: list[object] = []
    seen_targets: list[list[dict]] = []
    inference_calls = 0
    fail_on_inference: int | None = None

    @classmethod
    def build(cls) -> object:
        import torch

        class _Detector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))
                self.first_scale: float | None = None
                # The adapter sizes the pretrained heads for the dataset's
                # classes. A real torchvision model exposes these attributes.
                self.roi_heads = types.SimpleNamespace(
                    box_predictor=types.SimpleNamespace(
                        cls_score=types.SimpleNamespace(in_features=8)
                    ),
                    mask_predictor=types.SimpleNamespace(
                        conv5_mask=types.SimpleNamespace(in_channels=8)
                    ),
                )
                _FakeDetector.instances.append(self)

            def forward(self, images: list, targets: list | None = None) -> object:
                if targets is not None:
                    if self.first_scale is None:
                        self.first_scale = float(self.scale)
                    _FakeDetector.seen_targets.append(
                        [
                            {
                                "masks": tuple(target["masks"].shape),
                                "boxes": tuple(target["boxes"].shape),
                                "labels": [int(label) for label in target["labels"]],
                            }
                            for target in targets
                        ]
                    )
                    total = self.scale.sum() * 0.0
                    for image in images:
                        total = total + image.float().sum() * 0.0
                    return {
                        "loss_classifier": total,
                        "loss_box_reg": total,
                        "loss_mask": total,
                        "loss_objectness": total,
                        "loss_rpn_box_reg": total,
                    }
                _FakeDetector.inference_calls += 1
                if _FakeDetector.fail_on_inference == _FakeDetector.inference_calls:
                    # A preemption after the previous epoch's checkpoint.
                    raise RuntimeError("simulated preemption during inference")
                return [
                    _outputs(detections, image)
                    for detections, image in zip(
                        _FakeDetector.prediction_sets.pop(0), images, strict=False
                    )
                ]

        return _Detector()


def _outputs(detections: list[dict], image: object) -> dict:
    """One image's torchvision inference dict, shaped like Mask R-CNN's."""
    import torch

    height, width = int(image.shape[-2]), int(image.shape[-1])  # type: ignore[attr-defined]
    boxes = torch.tensor([item["xyxy"] for item in detections], dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor([item["class_id"] + 1 for item in detections], dtype=torch.int64)
    scores = torch.tensor([item["score"] for item in detections], dtype=torch.float32)
    masks = [
        torch.from_numpy(_mask_for(item["polygon"], width, height)).float() for item in detections
    ]
    stacked = torch.stack(masks).unsqueeze(1) if masks else torch.zeros((0, 1, height, width))
    return {"boxes": boxes, "labels": labels, "scores": scores, "masks": stacked}


class _FakeLightningModule:
    def __init__(self) -> None:
        self._epoch = 0

    @property
    def current_epoch(self) -> int:
        return self._epoch


class _FakeTrainer:
    """Runs the validation loop, the epoch hook and the registered callback."""

    interrupt_after: int | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.max_epochs = int(kwargs.get("max_epochs", 1))
        self.precision = str(kwargs.get("precision", "32-true"))
        self.callbacks = list(kwargs.get("callbacks", []))
        self.should_stop = False
        self.current_epoch = 0

    def fit(self, module: object, ckpt_path: object = None) -> None:
        module.trainer = self
        start_epoch = 0
        if ckpt_path is not None:
            start_epoch = int(Path(ckpt_path).read_text(encoding="utf-8").split(":")[-1]) + 1
        self.current_epoch = start_epoch
        for completed, epoch in enumerate(
            range(start_epoch, self.max_epochs), start=start_epoch + 1
        ):
            self.current_epoch = epoch
            module._epoch = epoch
            for index, batch in enumerate(module.val_dataloader()):
                module.validation_step(batch, index)
            module.on_validation_epoch_end()
            for callback in self.callbacks:
                callback.on_validation_epoch_end(self, module)
            self.current_epoch = epoch + 1
            if self.should_stop:
                break
            if self.interrupt_after is not None and completed >= self.interrupt_after:
                raise RuntimeError("simulated preemption after the epoch checkpoint")

    def save_checkpoint(self, path: object) -> None:
        """Stand-in for Lightning's checkpoint writer; names the epoch it holds."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"lightning-checkpoint:{self.current_epoch}", encoding="utf-8")


class _FakeCallback:
    pass


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    prediction_sets: list[list[list[dict]]],
    *,
    interrupt_after: int | None = None,
) -> None:
    """Fake the framework boundary: Lightning and the torchvision factory."""
    _FakeDetector.prediction_sets = list(prediction_sets)
    _FakeDetector.instances = []
    _FakeDetector.seen_targets = []
    _FakeDetector.factory_calls = []
    _FakeDetector.inference_calls = 0
    _FakeDetector.fail_on_inference = None
    _FakeTrainer.interrupt_after = interrupt_after
    lightning = types.ModuleType("lightning")
    lightning.LightningModule = _FakeLightningModule
    lightning.Callback = _FakeCallback
    lightning.Trainer = _FakeTrainer
    lightning.seed_everything = lambda seed, **_kw: seed
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    _install_torchvision_factory(monkeypatch)


def _install_torchvision_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.models.detection

    _FakeDetector.factory_calls = []

    def factory(**kwargs: object) -> object:
        _FakeDetector.factory_calls.append(kwargs)
        return _FakeDetector.build()

    monkeypatch.setattr(torchvision.models.detection, "maskrcnn_resnet50_fpn_v2", factory)


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _train_mask_rcnn(
    tmp_path: Path,
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
    prediction_sets: list[list[list[dict]]],
    *,
    label: str = "mask-rcnn",
    epochs: int = 1,
) -> tuple[int, Path]:
    _install_fakes(monkeypatch, prediction_sets)
    output = tmp_path / "executions" / label
    code = _command(
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        "family=mask_rcnn",
        f"protocol.epochs={epochs}",
    )
    return code, output


def _train_yolo_seg(
    tmp_path: Path,
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
    prediction_sets: list[list[list[dict]]],
    *,
    label: str = "yolo-seg",
    epochs: int = 1,
) -> tuple[int, Path]:
    from tests.test_pipeline_yolo_seg import _install_fakes as install_yolo

    install_yolo(monkeypatch, prediction_sets)
    output = tmp_path / "executions" / label
    code = _command(
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        "family=yolo_segmentation",
        f"protocol.epochs={epochs}",
    )
    return code, output


# ── Acceptance criteria ───────────────────────────────────────────────────


def test_mask_rcnn_trains_from_coco_masks_for_one_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole path is visible from the command: artifacts and metrics."""
    store = _imported_store(tmp_path)

    code, output = _train_mask_rcnn(tmp_path, store, monkeypatch, [_perfect_validation()])

    assert code == 0, capsys.readouterr().err
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
    assert (output / "checkpoints" / "epoch_000.pt").is_file()

    state = _read(output / "training_state.json")
    assert state["family"]["name"] == "mask_rcnn"
    assert state["family"]["framework"] == "torchvision"
    assert state["family"]["task"] == "segment"
    assert state["epochs_completed"] == 1
    assert state["early_stopping"]["stopped_early"] is False
    # AMP is a protocol request, recorded identically for every family.
    assert state["precision"] == "16-mixed"
    assert state["amp_active"] is True

    # The canonical COCO store is the only dataset; the run folder holds no
    # masks, no YOLO labels and no data.yaml.
    canonical = _read(store / "dataset.json")
    assert canonical["images"][0]["file_name"].startswith("images/")
    assert not (output / "labels").exists()
    assert not (output / "data.yaml").exists()
    assert not list(output.rglob("*.txt"))
    assert not list(output.rglob("*.json")) or {path.name for path in output.rglob("*.json")} <= {
        "metrics.json",
        "provenance.json",
        "training_state.json",
        "progress.json",
    }

    # Mask R-CNN consumes the canonical COCO JSON directly: no second
    # framework-specific conversion is written for it.
    provenance = _read(output / "provenance.json")
    assert provenance["dataset_source"]["kind"] == "coco_instances"
    assert provenance["dataset_source"]["path"] == str(store)
    assert "conversion" not in provenance or not provenance["conversion"]

    # The pretrained detector was requested at the protocol's image size, and
    # the shared evaluator scored the predictions with the segmentation
    # criterion.
    call = _FakeDetector.factory_calls[-1]
    assert call["weights"] == "DEFAULT"
    assert call["min_size"] == 1280 and call["max_size"] == 1280
    printed = capsys.readouterr().out
    assert "SELECTION: with_class_score=1.000000" in printed
    assert "CRITERION: COCO mask mAP" in printed
    assert "LOCALISATION: 1.000000" in printed

    from polyx.core.provenance import sha256_file

    assert sha256_file(output / "checkpoints" / "best.pt") == sha256_file(
        output / "checkpoints" / "epoch_000.pt"
    )


def test_the_shared_evaluator_scores_both_segmentation_families_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same geometry in, same metric shape, criteria and operating point out."""
    store = _imported_store(tmp_path)
    geometry = _shifted_validation(dy=2.0)

    seg_code, seg_output = _train_yolo_seg(
        tmp_path, store, monkeypatch, [geometry], label="parity-yolo"
    )
    mask_code, mask_output = _train_mask_rcnn(
        tmp_path, store, monkeypatch, [geometry], label="parity-mask"
    )

    assert seg_code == 0 and mask_code == 0
    seg = _read(seg_output / "metrics.json")
    mask = _read(mask_output / "metrics.json")

    assert set(mask["figures"]) == set(seg["figures"])
    assert set(mask["per_class"]) == set(seg["per_class"])
    assert mask["selection"]["metric"] == seg["selection"]["metric"] == "with_class_score"
    assert mask["selection"]["criterion"] == seg["selection"]["criterion"]
    assert mask["selection"]["value"] == pytest.approx(seg["selection"]["value"])

    for name in seg["figures"]:
        assert mask["figures"][name]["criterion"] == seg["figures"][name]["criterion"], name
        assert mask["figures"][name]["value"] == pytest.approx(seg["figures"][name]["value"]), name
    for class_name in seg["per_class"]:
        assert set(mask["per_class"][class_name]) == set(seg["per_class"][class_name])
        for name in seg["per_class"][class_name]:
            assert mask["per_class"][class_name][name]["value"] == pytest.approx(
                seg["per_class"][class_name][name]["value"]
            ), (class_name, name)

    # The operating point is a real number, not a degenerate perfect score.
    assert mask["selection"]["value"] == pytest.approx(0.7)
    assert mask["figures"]["mask_iou"]["value"] == pytest.approx(19.0 / 23.0)


def test_both_segmentation_families_resolve_the_same_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model is the only variable: the resolved protocol is identical."""
    store = _imported_store(tmp_path)
    seg_code, seg_output = _train_yolo_seg(
        tmp_path, store, monkeypatch, [_perfect_validation()], label="protocol-yolo"
    )
    mask_code, mask_output = _train_mask_rcnn(
        tmp_path, store, monkeypatch, [_perfect_validation()], label="protocol-mask"
    )
    assert seg_code == 0 and mask_code == 0

    def resolved(path: Path) -> dict:
        return yaml.safe_load((path / "resolved_protocol.yaml").read_text(encoding="utf-8"))

    seg_protocol = resolved(seg_output)
    mask_protocol = resolved(mask_output)
    assert mask_protocol == seg_protocol
    assert _read(mask_output / "training_state.json")["protocol"] == mask_protocol
    # The family is what changed, and only the family.
    mask_config = yaml.safe_load((mask_output / "config.yaml").read_text(encoding="utf-8"))
    seg_config = yaml.safe_load((seg_output / "config.yaml").read_text(encoding="utf-8"))
    assert mask_config["family"]["name"] == "mask_rcnn"
    assert seg_config["family"]["name"] == "yolo_segmentation"
    assert mask_config["family"] != seg_config["family"]


def test_benchmark_augmentation_is_the_protocol_subset_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No family-specific augmentation enters the benchmark campaign."""
    store = _imported_store(tmp_path)
    mask_code, mask_output = _train_mask_rcnn(
        tmp_path, store, monkeypatch, [_perfect_validation()], label="aug-mask"
    )
    seg_code, seg_output = _train_yolo_seg(
        tmp_path, store, monkeypatch, [_perfect_validation()], label="aug-yolo"
    )
    assert mask_code == 0 and seg_code == 0

    mask_augmentation = _read(mask_output / "training_state.json")["augmentation"]
    seg_augmentation = _read(seg_output / "training_state.json")["augmentation"]
    declared = ["hsv", "flip_lr", "mosaic"]

    for augmentation in (mask_augmentation, seg_augmentation):
        assert augmentation["subset"] == declared
        assert augmentation["family_specific"] is False
        assert set(augmentation["applied"]) <= set(declared)
        assert not set(augmentation["applied"]) & set(augmentation["skipped"])

    # Mask R-CNN implements the two framework-neutral transforms and never
    # substitutes a torchvision-specific one for mosaic.
    assert mask_augmentation["applied"] == ["hsv", "flip_lr"]
    assert mask_augmentation["skipped"] == ["mosaic"]
    assert seg_augmentation["applied"] == declared
    # The declared subset is the same in both families, so the benchmark
    # campaign's augmentation is not a per-family variable.
    assert mask_augmentation["subset"] == seg_augmentation["subset"]


def test_one_epoch_completes_under_the_real_lightning_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter must be a Lightning module that trains on COCO masks."""
    store = _imported_store(tmp_path)
    _FakeDetector.prediction_sets = [_perfect_validation()]
    _FakeDetector.instances = []
    _FakeDetector.seen_targets = []
    _install_torchvision_factory(monkeypatch)
    output = tmp_path / "executions" / "real-loop"

    code = _command(
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        "family=mask_rcnn",
        "protocol.epochs=1",
        "protocol.amp.enabled=false",
        "protocol.warmup_epochs=0",
    )

    assert code == 0, "the real Lightning loop rejected the adapter"
    assert _FakeDetector.seen_targets, "Lightning never ran a training step"
    first_batch = _FakeDetector.seen_targets[0]
    assert first_batch, "the training batch carried no targets"
    for target in first_batch:
        masks, boxes, labels = target["masks"], target["boxes"], target["labels"]
        assert len(masks) == 3 and masks[0] > 0  # N, H, W per instance
        assert len(boxes) == 2  # N, 4
        # torchvision reserves label 0 for background; the dataset must shift.
        assert labels and min(labels) >= 1 and max(labels) <= len(NAMES)

    state = _read(output / "training_state.json")
    assert state["epochs_completed"] == 1
    assert (output / "checkpoints" / "epoch_000.pt").is_file()
    assert _read(output / "metrics.json")["selection"]["value"] == pytest.approx(1.0)


def test_a_preempted_mask_rcnn_run_resumes_from_the_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights and the engine's selection state survive a preemption."""
    store = _imported_store(tmp_path)
    output = tmp_path / "executions" / "resumable-mask"
    overrides = (
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        "family=mask_rcnn",
        "protocol.epochs=2",
        "protocol.early_stopping.patience=99",
    )

    _install_fakes(
        monkeypatch,
        [_perfect_validation(), _perfect_validation()],
        interrupt_after=1,
    )
    assert _command(*overrides) == 1

    progress = _read(output / "progress.json")
    assert progress["epochs_completed"] == 1
    resumed_from = progress["last_checkpoint"]
    assert Path(resumed_from).is_file()
    assert Path(progress["resume_checkpoint"]).is_file()
    first_best = (output / "checkpoints" / "epoch_000.pt").read_bytes()
    assert not (output / "training_state.json").exists()

    _install_fakes(monkeypatch, [_perfect_validation(), _perfect_validation()])
    assert _command(*overrides, "resume=true") == 0

    state = _read(output / "training_state.json")
    assert state["resumed"] is True
    assert state["resumed_from_epoch"] == 1
    assert state["resumed_from"] == resumed_from
    assert state["epochs_completed"] == 2
    assert (output / "checkpoints" / "epoch_001.pt").is_file()
    assert (output / "checkpoints" / "epoch_000.pt").read_bytes() == first_best


def test_mask_rcnn_requires_the_coco_mask_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A torchvision segmenter is never trained from box labels."""
    from tests.test_pipeline_local_tracer import build_tiny_dataset

    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    code = _command(
        f"data_yaml={data_yaml}",
        f"output_dir={tmp_path / 'executions' / 'wrong-source'}",
        "family=mask_rcnn",
    )

    assert code == 1
    assert "coco mask dataset" in capsys.readouterr().err.lower()
    assert not (tmp_path / "executions" / "wrong-source" / "training_state.json").exists()
