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
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from polyx.core.provenance import sha256_file
from tests.conftest import build_tiny_dataset as build_tiny_dataset  # re-export for sibling suites

NAMES = {0: "PET", 1: "PP", 2: "LDPE"}


def _boxes_prediction() -> list[dict]:
    """Correct boxes on the tiny 64x48 images, in validation order."""
    return [
        {"image_id": 0, "category_id": 0, "bbox": [16.0, 12.0, 16.0, 12.0], "score": 0.9},
        {"image_id": 0, "category_id": 1, "bbox": [40.0, 12.0, 16.0, 12.0], "score": 0.8},
        {"image_id": 1, "category_id": 2, "bbox": [16.0, 30.0, 16.0, 12.0], "score": 0.85},
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


class _FakeNetwork(torch.nn.Module):
    """Picklable stand-in for the Ultralytics network saved in a snapshot."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """A differentiable (loss, items) pair, the shape Ultralytics returns."""
        return self.weight * 1.0, {}


class _FakeYOLO:
    """Ultralytics stand-in: one prediction set per validation call.

    The Lightning loop is the real one; only the framework at the boundary is
    a double, so no weights are downloaded and the suite stays offline.
    """

    prediction_sets: list[list[dict]] = []
    instances: list[_FakeYOLO] = []
    validation_calls = 0
    fail_on_validation: int | None = None

    def __init__(self, weights: object) -> None:
        self.weights = str(weights)
        self.names = dict(NAMES)
        self.model = _FakeNetwork()
        _FakeYOLO.instances.append(self)

    def predict(self, source: object = None, **_kwargs: object) -> list[_FakeResult]:
        _FakeYOLO.validation_calls += 1
        if _FakeYOLO.fail_on_validation == _FakeYOLO.validation_calls:
            # A time limit or preemption during this epoch's validation.
            raise RuntimeError("simulated preemption during validation")
        paths = list(source) if isinstance(source, (list, tuple)) else []
        per_image = _FakeYOLO.prediction_sets.pop(0)
        return [
            _FakeResult([p for p in per_image if p["image_id"] == image_id])
            for image_id in range(len(paths))
        ]

    def save(self, path: object) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"checkpoint:{self.weights}:{target.name}".encode())


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    prediction_sets: list[list[dict]],
    *,
    interrupt_after: int | None = None,
) -> None:
    """Replace only the Ultralytics wrapper; Lightning runs for real."""
    import ultralytics

    _FakeYOLO.prediction_sets = list(prediction_sets)
    _FakeYOLO.instances = []
    _FakeYOLO.validation_calls = 0
    _FakeYOLO.fail_on_validation = interrupt_after
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)


def test_resume_finishes_publishing_after_metrics_before_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed epoch with missing report is published, never retrained."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction(), _boxes_prediction()])
    output = tmp_path / "executions" / "publication"
    overrides = [
        f"data_yaml={tiny_dataset_path(tmp_path)}",
        f"output_dir={output}",
        "protocol.epochs=1",
    ]
    assert _command(*overrides) == 0
    metrics_before = (output / "metrics.json").read_bytes()
    (output / "report.md").unlink()
    (output / "completion.json").unlink()

    assert _command(*overrides, "resume=true") == 0
    assert (output / "completion.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "metrics.json").read_bytes() == metrics_before
    assert "recovered_publication" not in _read(output / "training_state.json")


def test_completion_is_not_written_when_the_final_log_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed log line after completion would mark a broken run complete."""
    from polyx.pipeline import execution as engine

    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    output = tmp_path / "executions" / "log-failure"
    real_append = engine.append_log

    def failing_append(path: Path, message: str) -> None:
        if message.startswith("done "):
            raise OSError("simulated log failure")
        real_append(path, message)

    monkeypatch.setattr(engine, "append_log", failing_append)
    code = _command(
        f"data_yaml={build_tiny_dataset(tmp_path / 'dataset')}",
        f"output_dir={output}",
        "protocol.epochs=1",
    )
    assert code == 1
    assert not (output / "completion.json").exists()


def test_a_fresh_run_refuses_an_existing_execution_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """A reused directory is resumed explicitly, never inherited by a new run."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    output = tmp_path / "executions" / "reused"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = [f"data_yaml={data_yaml}", f"output_dir={output}", "protocol.epochs=1"]
    assert _command(*overrides) == 0
    assert (output / "completion.json").is_file()

    selected_language("en")
    assert _command(*overrides) == 1
    assert "already holds an execution" in capsys.readouterr().err
    # The finished execution is untouched: its marker and metrics survive.
    assert (output / "completion.json").is_file()
    assert _read(output / "metrics.json")["selection"]["value"] > 0.0


def test_resume_rebuilds_corrupt_metrics_from_the_progress_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated metrics file is recovered from the intact progress record."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction(), _boxes_prediction()])
    output = tmp_path / "executions" / "truncated"
    overrides = [
        f"data_yaml={build_tiny_dataset(tmp_path / 'dataset')}",
        f"output_dir={output}",
        "protocol.epochs=1",
    ]
    assert _command(*overrides) == 0
    (output / "metrics.json").write_text('{"selection":', encoding="utf-8")
    (output / "completion.json").unlink()
    (output / "report.md").unlink()

    assert _command(*overrides, "resume=true") == 0
    assert (output / "completion.json").is_file()
    assert (output / "report.md").is_file()
    assert _read(output / "metrics.json")["selection"]["metric"] == "with_class_score"


def _same_weights(first: Path, second: Path) -> bool:
    """Whether two native exports carry the same tensors.

    File bytes are not the contract: the criterion object a checkpoint pickles
    changes between training and a restored validation pass, while the weights
    are what ``best.pt`` promises.
    """
    import torch

    left = torch.load(first, map_location="cpu", weights_only=False)["model"].state_dict()
    right = torch.load(second, map_location="cpu", weights_only=False)["model"].state_dict()
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def tiny_dataset_path(tmp_path: Path) -> Path:
    """The shared fixture path used by the publication test."""
    return build_tiny_dataset(tmp_path / "dataset")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def test_local_tracer_emits_every_artifact_and_names_each_criterion(
    tmp_path: Path, tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command is the only seam; all proof is on disk."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _empty_prediction(), _boxes_prediction()])
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
    assert "adapter_cross_check" not in metrics, "the shared evaluator is the only scorer"

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
    assert _same_weights(best, output / "checkpoints" / "epoch_000.pt")
    assert not _same_weights(best, output / "checkpoints" / "epoch_001.pt")

    # FP16 AMP is configured and recorded as active.
    assert state["amp"]["enabled"] is True
    assert state["amp"]["dtype"] == "fp16"
    # Lightning substitutes bf16-mixed when the device has no fp16 support
    # (CPU here); the request and the precision actually run are both recorded.
    assert state["amp_requested"] == "16-mixed"
    assert state["precision"] in {"16-mixed", "bf16-mixed"}
    assert state["amp_active"] is True

    # Provenance: git commit and dataset hashes are recorded.
    provenance = _read(output / "provenance.json")
    assert provenance["git"]["commit"]
    assert "data.yaml" in provenance["dataset_hashes"]
    assert "images/val/plate_v1.png" in provenance["dataset_hashes"]
    assert provenance["dataset_hashes"]["images/val/plate_v1.png"] == sha256_file(
        tiny_dataset.parent / "images" / "val" / "plate_v1.png"
    )
    assert provenance["dataset_audit"]["ok"] is True


def test_one_epoch_execution_completes_without_early_stop(
    tmp_path: Path, tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal one-epoch execution finishes and writes one checkpoint."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    output = tmp_path / "executions" / "one_epoch"

    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=1",
    )

    assert code == 0
    state = _read(output / "training_state.json")
    assert state["epochs_completed"] == 1
    assert state["early_stopping"]["stopped_early"] is False
    assert (output / "checkpoints" / "epoch_000.pt").is_file()
    assert _read(output / "metrics.json")["selection"]["value"] == pytest.approx(1.0)
    # Untouched early-stopping defaults: the callbacks get the documented rule.
    protocol = yaml.safe_load((output / "resolved_protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["early_stopping"]["patience"] == 30
    assert protocol["early_stopping"]["min_delta"] == pytest.approx(0.001)


def test_patience_is_honoured_and_protocol_defaults_are_stable(
    tmp_path: Path, tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A larger patience keeps training past the first stagnant epoch."""
    _install_fakes(
        monkeypatch,
        [
            _boxes_prediction(),
            _empty_prediction(),
            _empty_prediction(),
            # The publication pass re-scores the kept (first) epoch.
            _boxes_prediction(),
        ],
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
    # The stopping rule the callbacks receive is the documented one.
    assert protocol["early_stopping"]["monitor"] == "with_class_score"
    assert protocol["early_stopping"]["mode"] == "max"
    assert protocol["early_stopping"]["patience"] == 2  # overridden above
    assert protocol["early_stopping"]["min_delta"] == pytest.approx(0.001)


def test_interrupted_execution_resumes_from_the_last_checkpoint(
    tmp_path: Path, tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preemption leaves checkpoints and progress; a resumed run continues.

    The whole proof is read from the output directory: the number of epochs the
    second run reports, the checkpoint it names as the resume point and the
    fact that the first run's best weights survived unchanged.
    """
    output = tmp_path / "executions" / "resumable"
    _install_fakes(
        monkeypatch,
        [_boxes_prediction(), _boxes_prediction()],
        interrupt_after=2,
    )

    first = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=6",
        "protocol.early_stopping.patience=99",
    )

    assert first == 1  # the simulated preemption is reported as a failure
    progress = _read(output / "progress.json")
    # An epoch is checkpointed only after its validation scored it, so the
    # interrupted validation left epoch 0 as the last completed one.
    assert progress["epochs_completed"] == 1
    assert progress["best_epoch"] == 0
    # The native snapshot for epoch 1 was written before its validation ran;
    # what an interrupted validation must not leave is a completed-epoch
    # Lightning checkpoint.
    assert not (output / "checkpoints" / "lightning" / "checkpoint_001.ckpt").exists()
    assert not (output / "training_state.json").exists()
    first_best = (output / "checkpoints" / "epoch_000.pt").read_bytes()
    assert Path(progress["resume_checkpoint"]).name == "last.ckpt"
    assert Path(progress["resume_checkpoint"]).is_file()

    _install_fakes(monkeypatch, [_boxes_prediction()] * 7)
    second = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=6",
        "protocol.early_stopping.patience=99",
        "resume=true",
    )

    assert second == 0
    state = _read(output / "training_state.json")
    assert state["epochs_completed"] == 6
    assert state["early_stopping"]["best_epoch"] == 0
    assert state["resumed_from_epoch"] == 1
    assert state["resumed_from"] == str(output / "checkpoints" / "lightning" / "last.ckpt")
    assert (output / "checkpoints" / "epoch_000.pt").read_bytes() == first_best
    assert (output / "checkpoints" / "epoch_001.pt").is_file()
    assert (output / "checkpoints" / "epoch_005.pt").is_file()
    assert _same_weights(
        output / "checkpoints" / "best.pt", output / "checkpoints" / "epoch_000.pt"
    )
    assert "resume from epoch=1" in (output / "logs" / "training.log").read_text(encoding="utf-8")


def test_restore_best_false_is_refused_instead_of_crashing_after_training(
    tmp_path: Path,
    tiny_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The unsupported flag is validated before the first epoch, not after."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    output = tmp_path / "executions" / "no-restore"

    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=1",
        "protocol.early_stopping.restore_best=false",
    )

    assert code == 1
    assert "restore_best" in capsys.readouterr().err
    assert not (output / "training_state.json").exists()


def test_resume_without_progress_fails_loudly(
    tmp_path: Path,
    tiny_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A resume request without a progress snapshot must not restart silently."""
    _install_fakes(monkeypatch, [_boxes_prediction()])
    output = tmp_path / "executions" / "never-started"

    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=1",
        "resume=true",
    )

    assert code == 1
    assert "no progress" in capsys.readouterr().err
    assert not (output / "training_state.json").exists()


def test_resume_with_a_different_protocol_is_refused(
    tmp_path: Path,
    tiny_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """The protocol that produced a checkpoint is part of the resume contract."""
    output = tmp_path / "executions" / "protocol-drift"
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()], interrupt_after=2)
    assert (
        _command(
            f"data_yaml={tiny_dataset}",
            f"output_dir={output}",
            "protocol.epochs=6",
            "protocol.early_stopping.patience=99",
        )
        == 1
    )

    _install_fakes(monkeypatch, [_boxes_prediction()])
    capsys.readouterr()
    selected_language("en")
    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=6",
        "protocol.batch=16",
        "protocol.early_stopping.patience=99",
        "resume=true",
    )

    assert code == 1
    assert "protocol changed" in capsys.readouterr().err
    assert not (output / "training_state.json").exists()


@pytest.mark.parametrize("changed", ["label", "image", "test", "unrelated"])
def test_resume_fingerprints_the_resolved_dataset_outside_the_yaml_directory(
    tmp_path: Path,
    tiny_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
    changed: str,
) -> None:
    """The weights and best figures belong to the bytes that trained them."""
    declaration = tmp_path / "declaration"
    declaration.mkdir()
    data_yaml = declaration / "data.yaml"
    test_images = tiny_dataset.parent / "images" / "test"
    test_images.mkdir()
    test_image = test_images / "plate_test.png"
    test_image.write_bytes((tiny_dataset.parent / "images" / "val" / "plate_v1.png").read_bytes())
    source = tiny_dataset.read_text(encoding="utf-8").replace("path: .", "path: ../dataset")
    data_yaml.write_text(source + "test: images/test\n", encoding="utf-8")
    output = tmp_path / "executions" / "dataset-drift"
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()], interrupt_after=2)
    assert (
        _command(
            f"data_yaml={data_yaml}",
            f"output_dir={output}",
            "protocol.epochs=2",
            "protocol.early_stopping.patience=99",
        )
        == 1
    )
    progress = _read(output / "progress.json")
    assert progress["dataset_fingerprint"]

    if changed == "label":
        label = tiny_dataset.parent / "labels" / "val" / "plate_v1.txt"
        original = label.read_text(encoding="utf-8")
        label.write_text(original.replace("0.375", "0.380", 1), encoding="utf-8")
    elif changed == "image":
        image = tiny_dataset.parent / "images" / "val" / "plate_v1.png"
        image.write_bytes(image.read_bytes() + b"\0")
    elif changed == "test":
        test_image.write_bytes(test_image.read_bytes() + b"\0")
    else:
        (declaration / "notes.txt").write_text("Not a dataset input.\n", encoding="utf-8")

    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    capsys.readouterr()
    selected_language("en")
    code = _command(
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        "protocol.epochs=2",
        "protocol.early_stopping.patience=99",
        "resume=true",
    )

    if changed == "unrelated":
        assert code == 0, capsys.readouterr().err
        assert (output / "completion.json").is_file()
    else:
        assert code == 1
        assert "dataset behind" in capsys.readouterr().err
        assert not (output / "training_state.json").exists()


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("split-list", "must name an image directory"),
        ("missing-split", "needs a directory with images"),
        ("legacy-label", "the two readers do not agree"),
        ("prediction-label", "five-column boxes"),
        ("corrupt-image", "cannot use"),
        ("unsupported-image", "unsupported headless box image format"),
    ],
)
def test_headless_box_preflight_names_incompatible_dataset_inputs(
    tmp_path: Path,
    tiny_dataset: Path,
    capsys: pytest.CaptureFixture[str],
    selected_language,
    problem: str,
    message: str,
) -> None:
    selected_language("en")
    root = tiny_dataset.parent
    image = root / "images" / "val" / "plate_v1.png"
    label = root / "labels" / "val" / "plate_v1.txt"
    overrides = []
    if problem == "split-list":
        offending = root / "val.txt"
        offending.write_text(f"{image}\n", encoding="utf-8")
        source = tiny_dataset.read_text(encoding="utf-8")
        tiny_dataset.write_text(source.replace("val: images/val", "val: val.txt"), encoding="utf-8")
    elif problem == "missing-split":
        overrides = ["protocol.split=holdout"]
        offending = tiny_dataset
    elif problem == "legacy-label":
        offending = image
        label.rename(image.with_suffix(".txt"))
    elif problem == "prediction-label":
        offending = label
        label.write_text("0 0.375 0.375 0.25 0.25 0.99\n", encoding="utf-8")
    elif problem == "unsupported-image":
        offending = image.with_suffix(".heic")
        image.rename(offending)
    else:
        offending = image
        image.write_bytes(b"not an image")

    output = tmp_path / "rejected"
    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "family.weights=yolov8n.yaml",
        "protocol.epochs=1",
        *overrides,
    )

    error = capsys.readouterr().err
    assert code == 1
    assert message in error
    assert str(offending) in error
    assert not (output / "progress.json").exists()


def test_preflight_refuses_an_image_declared_in_two_folds(
    tmp_path: Path,
    tiny_dataset: Path,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """A file in both folds would be trained on and then scored."""
    selected_language("en")
    # Pointing val at the train folder is the common way this happens.
    source = tiny_dataset.read_text(encoding="utf-8")
    tiny_dataset.write_text(
        source.replace("val: images/val", "val: images/train"), encoding="utf-8"
    )
    output = tmp_path / "fold-leak"

    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "family.weights=yolov8n.yaml",
        "protocol.epochs=1",
    )

    error = capsys.readouterr().err
    assert code == 1
    assert "declared in both the train and the val splits" in error
    assert str(tiny_dataset.parent / "images" / "train") in error
    assert not (output / "progress.json").exists()
