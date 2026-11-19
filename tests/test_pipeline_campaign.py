"""One local Hydra multirun campaign across every first-wave family.

The campaign is the campaign command with the local backend: Hydra launches one
job per model family in a single multirun, and each job is a normal training
execution under the identical protocol. Hydra, the composition, the campaign
runner, the engine, the shared evaluator, the artifact layout and the real
Lightning loop are the code under test. Only the model frameworks at the
boundary (Ultralytics, the torchvision factory) are doubles, so the campaign
runs offline and on CPU.

Every assertion reads a file a job wrote. The cluster campaign itself is a
manual acceptance step on the HPC cluster and is not faked here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_pipeline_local_tracer import build_tiny_dataset
from tests.test_pipeline_mask_rcnn import _FakeDetector, _install_torchvision_factory
from tests.test_pipeline_yolo_seg import (
    _FakeYOLO,
    _imported_store,
    _perfect_validation,
)

FAMILIES = ("yolo_detection", "yolo_segmentation", "mask_rcnn")


class _CampaignNet(torch.nn.Module):
    """A real torch module with the Ultralytics loss shape; no native loop."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def loss(self, batch: dict) -> tuple[torch.Tensor, None]:
        return self.weight * 1.0, None


def _install_framework_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Double the model frameworks; Hydra and Lightning stay real."""
    import ultralytics

    _FakeYOLO.network = _CampaignNet()
    # One validation call per YOLO family, one epoch each.
    _FakeYOLO.prediction_sets = [_perfect_validation(), _perfect_validation()]
    _FakeYOLO.instances = []
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    _FakeDetector.prediction_sets = [_perfect_validation()]
    _FakeDetector.instances = []
    _FakeDetector.seen_targets = []
    _FakeDetector.inference_calls = 0
    _FakeDetector.fail_on_inference = None
    _install_torchvision_factory(monkeypatch)


def _run_campaign(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    data_yaml: Path,
    store: Path,
    scratch: Path,
    *,
    families: tuple[str, ...] = FAMILIES,
    resume: bool = False,
) -> None:
    from polyx.pipeline import campaign_app

    overrides = [
        "campaign_app",
        "--multirun",
        f"campaign_root={root}",
        f"data_yaml={data_yaml}",
        f"coco_dataset={store}",
        f"scratch_dir={scratch}",
        f"family={','.join(families)}",
        "protocol.epochs=1",
        "protocol.batch=2",
        "protocol.image_size=64",
        "protocol.warmup_epochs=0",
        "protocol.amp.enabled=false",
        "protocol.early_stopping.patience=99",
    ]
    if resume:
        overrides.append("resume=true")
    monkeypatch.setattr(sys, "argv", overrides)
    campaign_app.campaign()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_one_multirun_trains_every_family_under_the_same_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign is one command; each family lands in its own execution."""
    data_yaml = build_tiny_dataset(tmp_path / "detection")
    store = _imported_store(tmp_path)
    root = tmp_path / "campaign"
    _install_framework_doubles(monkeypatch)

    _run_campaign(monkeypatch, root, data_yaml, store, tmp_path / "scratch")

    protocols = {}
    for family in FAMILIES:
        output = root / family
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
            assert (output / relative).is_file(), f"{family} is missing {relative}"
        state = _read(output / "training_state.json")
        assert state["family"]["name"] == family
        assert state["epochs_completed"] == 1
        assert _read(output / "metrics.json")["selection"]["metric"] == "with_class_score"
        provenance = _read(output / "provenance.json")
        if family == "yolo_detection":
            assert provenance["data_yaml"] == str(data_yaml)
            assert provenance["dataset_source"]["kind"] == "yolo_boxes"
        else:
            # The unused detection source is never reported as this family's
            # dataset, even though the campaign job carries both sources.
            assert provenance["data_yaml"] == ""
            assert provenance["dataset_source"]["kind"] == "coco_instances"
        protocols[family] = yaml.safe_load(
            (output / "resolved_protocol.yaml").read_text(encoding="utf-8")
        )

    # The model is the only variable: one protocol, three families.
    assert protocols["yolo_detection"] == protocols["yolo_segmentation"]
    assert protocols["yolo_segmentation"] == protocols["mask_rcnn"]
    # Hydra launched the three jobs from one multirun, one job directory each.
    for family in FAMILIES:
        assert (root / "hydra" / family).is_dir(), f"no multirun job directory for {family}"

    # The campaign's artifacts sync back per family, checkpoints included.
    from polyx.hpc.sync_back import sync_campaign

    synced = sync_campaign(root, tmp_path / "local")
    assert set(synced) == set(FAMILIES)
    for family in FAMILIES:
        for relative in (
            "config.yaml",
            "resolved_protocol.yaml",
            "metrics.json",
            "report.md",
            "checkpoints/best.pt",
        ):
            assert (tmp_path / "local" / family / relative).is_file(), f"{family}: {relative}"


def test_resuming_a_campaign_skips_finished_families_and_starts_the_missing_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A preempted campaign resumes only the families that need epochs.

    The bug: ``resume`` was passed to every family, so a family that never
    started aborted the whole job on a missing progress snapshot, and a family
    that had finished was rerun from its last state and could overwrite its
    metrics.
    """
    data_yaml = build_tiny_dataset(tmp_path / "detection")
    store = _imported_store(tmp_path)
    root = tmp_path / "campaign"
    scratch = tmp_path / "scratch"

    _install_framework_doubles(monkeypatch)
    _run_campaign(monkeypatch, root, data_yaml, store, scratch, families=("yolo_detection",))

    finished = root / "yolo_detection" / "metrics.json"
    finished_before = finished.read_bytes()
    finished_time = finished.stat().st_mtime_ns

    _install_framework_doubles(monkeypatch)
    _run_campaign(
        monkeypatch,
        root,
        data_yaml,
        store,
        scratch,
        families=FAMILIES,
        resume=True,
    )

    assert finished.read_bytes() == finished_before, "a finished family was overwritten"
    assert finished.stat().st_mtime_ns == finished_time
    printed = capsys.readouterr().out
    assert "SKIPPED" in printed and "yolo_detection" in printed
    for family in FAMILIES:
        state = _read(root / family / "training_state.json")
        assert state["epochs_completed"] == 1, family
    # The families that never started resumed nothing; they ran normally.
    assert _read(root / "yolo_segmentation" / "training_state.json").get("resumed") is None
    assert _read(root / "mask_rcnn" / "training_state.json").get("resumed") is None
    # The campaign still syncs every family's artifacts.
    from polyx.hpc.sync_back import sync_campaign

    assert set(sync_campaign(root, tmp_path / "local")) == set(FAMILIES)


def test_the_campaign_requires_a_root_and_a_dataset_for_each_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A campaign without a destination, or without a dataset, fails loudly."""
    data_yaml = build_tiny_dataset(tmp_path / "detection")
    store = _imported_store(tmp_path)
    _install_framework_doubles(monkeypatch)

    from polyx.pipeline import campaign_app

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_app",
            "--multirun",
            f"data_yaml={data_yaml}",
            f"coco_dataset={store}",
            f"family={','.join(FAMILIES)}",
            f"hydra.sweep.dir={tmp_path / 'nowhere'}",
            "protocol.epochs=1",
        ],
    )
    with pytest.raises(SystemExit):
        campaign_app.campaign()
    assert "campaign_root" in capsys.readouterr().err.lower()
