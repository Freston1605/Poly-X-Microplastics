"""Real-Lightning resume for every family, through the command seam.

The bug these tests pin down is framework-level: the Ultralytics adapters used
to hand an Ultralytics ``.pt`` to ``Trainer.fit(ckpt_path=...)``, which real
Lightning rejects because the file has no ``pytorch-lightning_version``; Mask
R-CNN restored its state dict by hand and restarted the epoch loop. Here the
Lightning ``Trainer``, its checkpoint writer, its optimizer/scheduler restore
and the epoch loop are the real code under test. Only the model framework at
the boundary (Ultralytics, the torchvision factory) is a double, so no weights
are downloaded and the suite stays offline.

The proof is external: the command's exit status, the artifacts under the
output directory, the real checkpoint's contents and the epochs each resumed
process actually ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.test_pipeline_local_tracer import (
    _boxes_prediction,
    _FakeResult,
    build_tiny_dataset,
)
from tests.test_pipeline_mask_rcnn import (
    _FakeDetector,
    _install_torchvision_factory,
)
from tests.test_pipeline_yolo_seg import _imported_store, _perfect_validation


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


# ── Ultralytics boundary double with a real torch module ──────────────────


class _TinyNet(torch.nn.Module):
    """Minimal stand-in for the Ultralytics network: parameters and a loss."""

    def __init__(self, owner: object) -> None:
        super().__init__()
        self.owner = owner
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """A differentiable (loss, items) pair, the shape Ultralytics returns."""
        owner = self.owner
        if owner.first_weight is None:  # type: ignore[attr-defined]
            owner.first_weight = float(self.weight)  # type: ignore[attr-defined]
        return self.weight * 1.0, {}


class _RealLoopYOLO:
    """Ultralytics stand-in for a real Lightning fit; no native loop."""

    prediction_sets: list[list[dict]] = []
    fail_on_validation: int | None = None
    instances: list[_RealLoopYOLO] = []

    def __init__(self, weights: object) -> None:
        self.requested_weights = str(weights)
        self.val_calls = 0
        self.first_weight: float | None = None
        self.model = _TinyNet(self)
        self.weights = self.model.weight
        _RealLoopYOLO.instances.append(self)

    def val(self, **_kwargs: object) -> list[_FakeResult]:
        self.val_calls += 1
        if _RealLoopYOLO.fail_on_validation == self.val_calls:
            # A time limit or preemption after the previous epoch's checkpoint
            # and progress snapshot were written.
            raise RuntimeError("simulated preemption during validation")
        per_image = _RealLoopYOLO.prediction_sets.pop(0)
        return [_FakeResult([p for p in per_image if p["image_id"] == i]) for i in range(2)]

    def save(self, path: object) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"ultralytics-export:{target.name}".encode())


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """The suite's tiny YOLO box dataset, rooted in this test's tmp dir."""
    return build_tiny_dataset(tmp_path / "dataset")


@pytest.fixture
def yolo_double(monkeypatch: pytest.MonkeyPatch) -> type[_RealLoopYOLO]:
    """Patch only the Ultralytics wrapper; Lightning stays real."""
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _RealLoopYOLO)
    _RealLoopYOLO.prediction_sets = []
    _RealLoopYOLO.fail_on_validation = None
    _RealLoopYOLO.instances = []
    return _RealLoopYOLO


def _fit_detection(output: Path, data_yaml: Path, *extra: str, resume: bool = False) -> int:
    overrides = [
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        "protocol.epochs=2",
        "protocol.batch=2",
        "protocol.image_size=64",
        "protocol.warmup_epochs=0",
        "protocol.amp.enabled=false",
        "protocol.early_stopping.patience=99",
        *extra,
    ]
    if resume:
        overrides.append("resume=true")
    return _command(*overrides)


def test_a_preempted_yolo_run_resumes_with_a_real_lightning_checkpoint(
    tmp_path: Path,
    tiny_dataset: Path,
    yolo_double: type[_RealLoopYOLO],
) -> None:
    """The interrupted run leaves a Lightning checkpoint; the resumed one uses it."""
    output = tmp_path / "executions" / "resume-yolo"
    yolo_double.prediction_sets = [_boxes_prediction(), _boxes_prediction()]
    # Fail on the second validation call: epoch 0 was completed normally.
    yolo_double.fail_on_validation = 2

    assert _fit_detection(output, tiny_dataset) == 1
    first = yolo_double.instances[-1]
    assert first.val_calls == 2, "the first epoch must complete before the preemption"

    progress = _read(output / "progress.json")
    assert progress["epochs_completed"] == 1
    # Compatibility export still names the family-native file...
    assert progress["last_checkpoint"] == str(output / "checkpoints" / "epoch_000.pt")
    assert Path(progress["last_checkpoint"]).is_file()
    # ...while resume names a real Lightning checkpoint.
    lightning = Path(progress["resume_checkpoint"])
    assert lightning == output / "checkpoints" / "lightning" / "last.ckpt"
    assert lightning.is_file()
    saved = torch.load(lightning, map_location="cpu", weights_only=False)
    assert "pytorch-lightning_version" in saved
    assert saved["optimizer_states"], "the optimizer state must be in the checkpoint"
    assert saved["lr_schedulers"], "the scheduler state must be in the checkpoint"
    assert saved["epoch"] == 0
    saved_weight = float(saved["state_dict"]["net.weight"])

    yolo_double.prediction_sets = [_boxes_prediction()]
    yolo_double.fail_on_validation = None
    yolo_double.instances = []
    assert _fit_detection(output, tiny_dataset, resume=True) == 0
    resumed = yolo_double.instances[-1]

    # Exactly the remaining epoch ran: a restart would validate twice.
    assert resumed.val_calls == 1
    state = _read(output / "training_state.json")
    assert state["epochs_completed"] == 2
    assert state["resumed"] is True
    assert state["resumed_from_epoch"] == 1
    assert (output / "checkpoints" / "epoch_001.pt").is_file()

    # The restored weights were the ones in the Lightning checkpoint, not a
    # fresh pretrained model: the first loss of the resumed run saw them.
    assert resumed.first_weight == pytest.approx(saved_weight)


def test_a_resumed_run_that_never_improves_keeps_the_snapshot_best_metrics(
    tmp_path: Path,
    tiny_dataset: Path,
    yolo_double: type[_RealLoopYOLO],
) -> None:
    """The remaining epoch does not beat the snapshot: no zero is written.

    Before the fix a resumed execution started its ``best`` bookkeeping from
    empty, so a non-improving remaining epoch evaluated an empty prediction
    list and overwrote ``metrics.json`` with 0.0 and ``selected_epoch = -1``
    even though ``best.pt`` and the early-stopping state named epoch 0.
    """
    output = tmp_path / "executions" / "resume-no-improvement"
    yolo_double.prediction_sets = [_boxes_prediction(), _boxes_prediction()]
    yolo_double.fail_on_validation = 2
    assert _fit_detection(output, tiny_dataset) == 1

    progress = _read(output / "progress.json")
    assert progress["best_epoch"] == 0
    assert progress["best_score"] == pytest.approx(1.0)

    # The resumed epoch finds nothing: the with-class score drops to zero and
    # cannot beat the snapshot's best.
    yolo_double.prediction_sets = [[]]
    yolo_double.fail_on_validation = None
    yolo_double.instances = []
    assert _fit_detection(output, tiny_dataset, resume=True) == 0

    metrics = _read(output / "metrics.json")
    state = _read(output / "training_state.json")
    assert metrics["selection"]["value"] == pytest.approx(state["early_stopping"]["best_score"])
    assert metrics["selection"]["value"] == pytest.approx(1.0)
    assert metrics["selected_epoch"] == state["early_stopping"]["best_epoch"] == 0
    assert metrics["figures"]["with_class_score"]["value"] == pytest.approx(1.0)


def test_a_legacy_snapshot_without_a_lightning_checkpoint_is_refused(
    tmp_path: Path,
    tiny_dataset: Path,
    yolo_double: type[_RealLoopYOLO],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A snapshot that predates Lightning checkpoints must not restart silently."""
    output = tmp_path / "executions" / "legacy-snapshot"
    yolo_double.prediction_sets = [_boxes_prediction(), _boxes_prediction()]
    yolo_double.fail_on_validation = 2
    assert _fit_detection(output, tiny_dataset) == 1

    progress = _read(output / "progress.json")
    progress["resume_checkpoint"] = None
    (output / "progress.json").write_text(json.dumps(progress), encoding="utf-8")

    yolo_double.prediction_sets = [_boxes_prediction()]
    yolo_double.fail_on_validation = None
    code = _fit_detection(output, tiny_dataset, resume=True)

    assert code == 1
    assert "no Lightning checkpoint" in capsys.readouterr().err
    assert not (output / "training_state.json").exists()


# ── Mask R-CNN under the same real loop ───────────────────────────────────


def _fit_mask_rcnn(output: Path, store: Path, scratch: Path, *, resume: bool = False) -> int:
    overrides = [
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={scratch}",
        "family=mask_rcnn",
        "family.weights=none",
        "protocol.epochs=2",
        "protocol.batch=2",
        "protocol.image_size=64",
        "protocol.warmup_epochs=0",
        "protocol.amp.enabled=false",
        "protocol.early_stopping.patience=99",
    ]
    if resume:
        overrides.append("resume=true")
    return _command(*overrides)


def test_a_preempted_mask_rcnn_run_continues_the_real_epoch_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The epoch loop, optimizer and scheduler continue from Lightning state."""
    store = _imported_store(tmp_path)
    output = tmp_path / "executions" / "resume-mask"
    scratch = tmp_path / "scratch"

    _FakeDetector.prediction_sets = [_perfect_validation(), _perfect_validation()]
    _FakeDetector.instances = []
    _FakeDetector.seen_targets = []
    _FakeDetector.inference_calls = 0
    # Fail on the second inference epoch, after epoch 0 completed.
    _FakeDetector.fail_on_inference = 2
    _install_torchvision_factory(monkeypatch)
    assert _fit_mask_rcnn(output, store, scratch) == 1

    progress = _read(output / "progress.json")
    assert progress["epochs_completed"] == 1
    assert progress["last_checkpoint"] == str(output / "checkpoints" / "epoch_000.pt")
    lightning = Path(progress["resume_checkpoint"])
    assert lightning.is_file()
    saved = torch.load(lightning, map_location="cpu", weights_only=False)
    assert "pytorch-lightning_version" in saved
    assert saved["optimizer_states"]
    assert saved["lr_schedulers"]
    assert "net.scale" in saved["state_dict"]
    saved_scale = float(saved["state_dict"]["net.scale"])

    _FakeDetector.prediction_sets = [_perfect_validation()]
    _FakeDetector.instances = []
    _FakeDetector.inference_calls = 0
    _FakeDetector.fail_on_inference = None
    _install_torchvision_factory(monkeypatch)
    assert _fit_mask_rcnn(output, store, scratch, resume=True) == 0

    state = _read(output / "training_state.json")
    assert state["epochs_completed"] == 2
    assert state["resumed"] is True
    assert state["resumed_from_epoch"] == 1
    # Exactly one new epoch ran, and one new per-epoch export exists.
    assert _FakeDetector.inference_calls == 1
    assert (output / "checkpoints" / "epoch_001.pt").is_file()
    assert _FakeDetector.instances[-1].first_scale is not None
    assert _FakeDetector.instances[-1].first_scale == pytest.approx(saved_scale)
