"""Hash-verified weights on the training path.

The adapters used to hand a bare framework identifier to Ultralytics and
torchvision, so an offline compute node would try to download a weight and the
``fetch_weights`` manifest was test-only. These tests drive the command seam:
a pinned weight that is absent or does not match its digest must stop the
execution before any training, and a staged, verified file must be the one the
framework receives.

The model frameworks are doubles; the resolver, the manifest parsing, the
digest check and the artifact writes are the real code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from polyx.core.provenance import sha256_file
from tests.test_pipeline_local_tracer import (
    _boxes_prediction,
    _install_fakes,
    build_tiny_dataset,
)
from tests.test_pipeline_mask_rcnn import _FakeDetector, _install_torchvision_factory
from tests.test_pipeline_yolo_seg import _imported_store, _perfect_validation


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(directory: Path, name: str, payload: bytes, digest: str | None = None) -> Path:
    """Write one weight file and a manifest entry for it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    manifest = directory / "SHA256SUMS"
    line = f"{digest or sha256_file(path)}  {name}\n"
    previous = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
    manifest.write_text(previous + line, encoding="utf-8")
    return path


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def test_training_fails_closed_when_a_pinned_weight_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An offline node must stop, not silently download."""
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "SHA256SUMS").write_text(f"{'0' * 64}  yolov8n.pt\n", encoding="utf-8")
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    _install_fakes(monkeypatch, [_boxes_prediction()])
    output = tmp_path / "executions" / "absent-weight"

    code = _command(
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        f"weights_dir={weights}",
        f"weights_manifest={weights / 'SHA256SUMS'}",
        "protocol.epochs=1",
    )

    assert code == 1
    error = capsys.readouterr().err
    assert "not staged" in error
    assert not (output / "training_state.json").exists()


def test_training_fails_closed_when_the_digest_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A present file with the wrong digest is not a trusted weight."""
    weights = tmp_path / "weights"
    _stage(weights, "yolov8n.pt", b"tampered", digest="1" * 64)
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    _install_fakes(monkeypatch, [_boxes_prediction()])
    output = tmp_path / "executions" / "bad-digest"

    code = _command(
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        f"weights_dir={weights}",
        "protocol.epochs=1",
    )

    assert code == 1
    assert "not verified" in capsys.readouterr().err
    assert not (output / "training_state.json").exists()


def test_a_staged_verified_weight_is_the_one_the_framework_receives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The absolute mounted path reaches YOLO, and the artifact records the proof."""
    weights = tmp_path / "weights"
    staged = _stage(weights, "yolov8n.pt", b"a trusted weight placeholder")
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    _install_fakes(monkeypatch, [_boxes_prediction()])
    output = tmp_path / "executions" / "verified-weight"

    code = _command(
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        f"weights_dir={weights}",
        "protocol.epochs=1",
    )

    assert code == 0
    from tests.test_pipeline_local_tracer import _FakeYOLO

    assert _FakeYOLO.instances[0].weights == str(staged.resolve())
    state = _read(output / "training_state.json")
    assert state["weights_verified"] is True
    assert state["weights"] == str(staged.resolve())


def test_a_torchvision_pinned_checkpoint_is_loaded_instead_of_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sibling family reads the same verified file, never ``weights=DEFAULT``."""
    store = _imported_store(tmp_path)
    weights = tmp_path / "weights"
    checkpoint = weights / "maskrcnn_resnet50_fpn_v2.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"scale": torch.ones(1)}, checkpoint)
    (weights / "SHA256SUMS").write_text(
        f"{sha256_file(checkpoint)}  {checkpoint.name}\n", encoding="utf-8"
    )
    _FakeDetector.prediction_sets = [_perfect_validation()]
    _FakeDetector.instances = []
    _FakeDetector.seen_targets = []
    _FakeDetector.inference_calls = 0
    _install_torchvision_factory(monkeypatch)
    output = tmp_path / "executions" / "pinned-mask"

    code = _command(
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        f"weights_dir={weights}",
        "family=mask_rcnn",
        "protocol.epochs=1",
        "protocol.amp.enabled=false",
        "protocol.warmup_epochs=0",
    )

    assert code == 0
    call = _FakeDetector.factory_calls[-1]
    assert call["weights"] is None, "a pinned checkpoint must not ask torchvision to download"
    assert call["weights_backbone"] is None
    state = _read(output / "training_state.json")
    assert state["weights_verified"] is True
    assert state["weights"] == str(checkpoint.resolve())


def test_an_architecture_declaration_and_an_untrained_model_need_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``yolov8n.yaml`` builds offline and ``weights: none`` has nothing to verify."""
    from polyx.pipeline.config import Family
    from polyx.pipeline.weights import resolve_weights

    assert resolve_weights(Family(weights="yolov8n.yaml")).path is None
    untrained = resolve_weights(Family(weights="none"))
    assert untrained.path is None
    assert untrained.verified is False
    # A configured, empty manifest still refuses a claim of verification.
    empty = tmp_path / "weights"
    empty.mkdir()
    (empty / "SHA256SUMS").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not pinned"):
        resolve_weights(Family(weights="yolov8n.pt"), weights_dir=empty)
