"""One YOLO-seg execution driven only through the headless command seam.

The whole path is exercised: a CVAT-shaped COCO export is imported as the
canonical segmentation training dataset (ticket #196), then the same command
trains a YOLO-seg family from that dataset for one epoch. Lightning and
Ultralytics are replaced at the import boundary; the command, Hydra config
resolution, job-time COCO conversion, shared evaluator, provenance and artifact
layout are the real code under test.

Every assertion reads a file the command wrote or a value it returned. No
private helper of the pipeline is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tests.conftest import write_image

NAMES = ("PET", "PP", "LDPE")
#: A 20x20 axis-aligned square on the 64x48 fixture images.
SQUARE = [10.0, 8.0, 30.0, 8.0, 30.0, 28.0, 10.0, 28.0]
#: Pixel offsets of the two annotations each validation image carries.
OFFSETS = (0.0, 32.0)


def _square(offset: float = 0.0, dy: float = 0.0) -> list[float]:
    return [value + (offset if index % 2 == 0 else dy) for index, value in enumerate(SQUARE)]


def _build_export(root: Path) -> Path:
    """A CVAT-shaped COCO export: four split images, every class in both folds."""
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images = [
        {"id": 1, "file_name": "train/plate_t1.png", "width": 64, "height": 48},
        {"id": 2, "file_name": "train/plate_t2.png", "width": 64, "height": 48},
        {"id": 3, "file_name": "val/plate_v1.png", "width": 64, "height": 48},
        {"id": 4, "file_name": "val/plate_v2.png", "width": 64, "height": 48},
    ]
    for image in images:
        image["split"] = "train" if image["file_name"].startswith("train") else "val"
        write_image(root / "images" / image["file_name"], width=64, height=48)

    # (image_id, category_id, x-offset): the validation geometry below is what
    # the fake segmenter predicts, so a perfect run scores exactly 1.0.
    planned = [
        (1, 1, OFFSETS[0]),
        (1, 2, OFFSETS[1]),
        (2, 3, OFFSETS[0]),
        (2, 1, OFFSETS[1]),
        (3, 1, OFFSETS[0]),
        (3, 2, OFFSETS[1]),
        (4, 3, OFFSETS[0]),
        (4, 2, OFFSETS[1]),
    ]
    annotations = [
        {
            "id": index,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [_square(offset)],
            "bbox": [10.0 + offset, 8.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        }
        for index, (image_id, category_id, offset) in enumerate(planned, start=1)
    ]
    export = {
        "info": {"description": "CVAT export of the mask fixture", "domain": "real"},
        "images": images,
        "categories": [{"id": index, "name": name} for index, name in enumerate(NAMES, start=1)],
        "annotations": annotations,
    }
    (annotations_dir / "instances_default.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8"
    )
    return root


# ── Ultralytics / Lightning doubles ───────────────────────────────────────


class _FakeBoxes:
    """The four arrays the adapter reads out of an Ultralytics result."""

    def __init__(self, detections: list[dict]) -> None:
        self.xyxy = np.array([item["xyxy"] for item in detections], dtype=float).reshape(-1, 4)
        self.conf = np.array([item["score"] for item in detections], dtype=float).reshape(-1)
        self.cls = np.array([item["class_id"] for item in detections], dtype=int).reshape(-1)
        # Adapter cross-check figures, shaped like Ultralytics' DetMetrics.
        self.ap_class_index = list(range(len(NAMES)))
        self.p = [0.9] * len(NAMES)
        self.r = [0.9] * len(NAMES)
        self.ap50 = [0.9] * len(NAMES)
        self.map50 = 0.9
        self.map = 0.7


class _FakeMasks:
    def __init__(self, detections: list[dict]) -> None:
        self.xy = [np.array(item["polygon"], dtype=float).reshape(-1, 2) for item in detections]


class _FakeSegResult:
    def __init__(self, detections: list[dict]) -> None:
        self.boxes = _FakeBoxes(detections)
        self.box = self.boxes
        self.masks = _FakeMasks(detections)
        self.names = dict(enumerate(NAMES))


class _FakeNetwork(torch.nn.Module):
    """Picklable stand-in for a network saved into a validation snapshot."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """A differentiable (loss, items) pair, the shape Ultralytics returns."""
        return self.weight * 1.0, {}


class _FakeYOLO:
    """Ultralytics stand-in: one per-image prediction set per validation call."""

    prediction_sets: list[list[list[dict]]] = []
    instances: list[_FakeYOLO] = []
    network: object | None = None

    def __init__(self, weights: object) -> None:
        self.weights = str(weights)
        self.names = dict(enumerate(NAMES))
        self.model = _FakeYOLO.network or _FakeNetwork()
        _FakeYOLO.instances.append(self)

    def predict(self, **_kwargs: object) -> list[_FakeSegResult]:
        per_image = _FakeYOLO.prediction_sets.pop(0)
        return [_FakeSegResult(detections) for detections in per_image]

    def save(self, path: object) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"checkpoint:{self.weights}:{target.name}".encode())


def _with_publication(
    prediction_sets: list[list[list[dict]]],
) -> list[list[list[dict]]]:
    """One scripted validation per epoch, plus the publication re-score.

    A run validates every epoch and validates the kept epoch once more before
    it publishes, so the last scripted outcome is the one that produces the
    published metrics.
    """
    if not prediction_sets:
        return []
    return [*prediction_sets, prediction_sets[0]]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, prediction_sets: list[list[list[dict]]]
) -> None:
    """Replace only the Ultralytics wrapper; Lightning runs for real."""
    import ultralytics

    _FakeYOLO.prediction_sets = list(prediction_sets)
    _FakeYOLO.instances = []
    _FakeYOLO.network = None
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)


class _TinySegNet(torch.nn.Module):
    """A real, picklable network with an Ultralytics-shaped ``loss``."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.seen_batches = 0
        self.mask_shapes: list[tuple[int, ...]] = []

    def loss(self, batch: object) -> tuple[object, None]:
        self.seen_batches += 1
        masks = batch.get("masks")  # type: ignore[union-attr]
        self.mask_shapes.append(tuple(masks.shape))
        images = batch["img"]  # type: ignore[index]
        zero = (self.scale * torch.ones(1)).sum() + images.float().sum()
        return zero * 0.0, None


def _tiny_seg_net() -> object:
    """A real ``torch`` network that records the batches it received."""
    return _TinySegNet()


def _install_real_lightning_fakes(
    monkeypatch: pytest.MonkeyPatch,
    network: object,
    prediction_sets: list[list[list[dict]]],
) -> None:
    """Fake only Ultralytics, so the real Lightning loop drives the adapter."""
    import ultralytics

    _FakeYOLO.prediction_sets = list(prediction_sets)
    _FakeYOLO.instances = []
    _FakeYOLO.network = network
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)


def _detection(class_index: int, offset: float, dy: float = 0.0) -> dict:
    polygon = _square(offset, dy)
    return {
        "class_id": class_index,
        "polygon": polygon,
        "xyxy": [polygon[0], polygon[1], polygon[4], polygon[5]],
        "score": 0.9,
    }


def _perfect_validation() -> list[list[dict]]:
    """The two validation images, each with the masks the export declares."""
    return [
        [_detection(0, OFFSETS[0]), _detection(1, OFFSETS[1])],  # PET, PP
        [_detection(2, OFFSETS[0]), _detection(1, OFFSETS[1])],  # LDPE, PP
    ]


def _shifted_validation(dy: float) -> list[list[dict]]:
    """The same masks moved two pixels down: mask IoU 0.9, not 1.0."""
    return [
        [_detection(0, OFFSETS[0], dy), _detection(1, OFFSETS[1], dy)],
        [_detection(2, OFFSETS[0], dy), _detection(1, OFFSETS[1], dy)],
    ]


# ── The command seam ──────────────────────────────────────────────────────


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _imported_store(tmp_path: Path) -> Path:
    """Import the COCO export as a versioned tool output; prove v1 landed."""
    export = _build_export(tmp_path / "export")
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    code = _command(
        f"coco_import={export}",
        f"dataset_root={dataset_root}",
        f"output_dir={tmp_path / 'executions' / 'import'}",
        "dataset_name=mask_batch",
    )
    assert code == 0
    store = dataset_root / "mask_batch" / "v1"
    assert (store / "dataset.json").is_file()
    return store


def test_duplicate_class_and_box_are_rejected_before_conversion(
    tmp_path: Path,
) -> None:
    """Two annotations the loader would collapse cannot become two targets."""
    from polyx.pipeline.seg_conversion import convert_coco_dataset

    store = _imported_store(tmp_path)
    dataset = _read(store / "dataset.json")
    first, second = dataset["annotations"][:2]
    second["image_id"] = first["image_id"]
    second["category_id"] = first["category_id"]
    # Declared boxes differ, but the serialized polygons collapse to one box.
    second["bbox"] = [value + 1.0 for value in first["bbox"]]
    second["segmentation"] = [list(first["segmentation"][0])]
    (store / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"(same YOLO box|misma caja YOLO)") as caught:
        convert_coco_dataset(store, tmp_path / "scratch-duplicate")
    message = str(caught.value)
    assert str(first["id"]) in message and str(second["id"]) in message
    assert dataset["images"][0]["file_name"] in message


def test_polygons_with_the_same_extrema_but_different_vertices_are_rejected(
    tmp_path: Path,
) -> None:
    """The loader deduplicates the class/box row, not the polygon line.

    A triangle and a square of the same class and extent produce two different
    label lines and still become one training target, because
    ``segments2boxes`` collapses both to the same ``xywh`` before ``np.unique``.
    """
    from polyx.pipeline.seg_conversion import convert_coco_dataset

    store = _imported_store(tmp_path)
    dataset = _read(store / "dataset.json")
    first, second = dataset["annotations"][:2]
    second["image_id"] = first["image_id"]
    second["category_id"] = first["category_id"]
    first["segmentation"] = [[10.0, 8.0, 30.0, 8.0, 30.0, 28.0, 10.0, 28.0]]
    second["segmentation"] = [[10.0, 8.0, 30.0, 28.0, 10.0, 28.0]]
    (store / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"(same YOLO box|misma caja YOLO)"):
        convert_coco_dataset(store, tmp_path / "scratch-extrema")


def test_a_square_image_annotation_is_quantized_like_the_label_file(tmp_path: Path) -> None:
    """Both families train on the six-decimal polygon the loader reads back.

    ``2.00001`` px survives a raw comparison against the loader's two-pixel
    retention threshold, but serializes to ``0.031250`` on a 64-pixel image and
    comes back as exactly ``2`` px, which the loader drops. Mask R-CNN must see
    the serialized number, or the two families keep different instances.
    """
    from polyx.pipeline.adapters import _filter_tiny_instances, _serialized_ring

    ring = [10.0, 8.0, 30.0, 8.0, 30.0, 10.00001, 10.0, 10.00001]
    quantized = _serialized_ring({"segmentation": [ring]}, 64, 64)
    assert quantized is not None
    height = float(quantized[:, 1].max()) - float(quantized[:, 1].min())
    assert height == 2.0
    raw = np.asarray(ring, dtype=float).reshape(-1, 2)
    assert float(raw[:, 1].max()) - float(raw[:, 1].min()) > 2.0

    def kept(points: np.ndarray) -> bool:
        box = [
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ]
        mask = np.zeros((64, 64), dtype=bool)
        _, _, labels = _filter_tiny_instances(
            [box],
            [mask],
            [1],
            image_width=64,
            image_height=64,
            image_size=64,
            segments=[points],
        )
        return bool(labels)

    assert kept(raw) is True
    assert kept(quantized) is False


def test_multipart_masks_are_rejected_instead_of_becoming_two_targets(
    tmp_path: Path,
) -> None:
    """One COCO instance must remain one YOLO instance at conversion time."""
    from polyx.pipeline.seg_conversion import convert_coco_dataset

    store = _imported_store(tmp_path)
    dataset = _read(store / "dataset.json")
    annotation = dataset["annotations"][0]
    annotation["segmentation"] = [
        _square(),
        [40.0, 8.0, 50.0, 8.0, 50.0, 18.0, 40.0, 18.0],
    ]
    (store / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"(exactly one polygon ring|exactamente un anillo de polígono)"
    ):
        convert_coco_dataset(store, tmp_path / "scratch-multipart")


def _train_segmentation(
    tmp_path: Path,
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
    prediction_sets: list[list[list[dict]]],
    *,
    epochs: int = 1,
) -> tuple[int, Path]:
    _install_fakes(monkeypatch, _with_publication(prediction_sets))
    output = tmp_path / "executions" / "segmentation"
    scratch = tmp_path / "scratch"
    code = _command(
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={scratch}",
        "family=yolo_segmentation",
        f"protocol.epochs={epochs}",
    )
    return code, output


def test_yolo_seg_trains_from_coco_masks_for_one_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The import and the training run are both observable from the command."""
    store = _imported_store(tmp_path)

    code, output = _train_segmentation(tmp_path, store, monkeypatch, [_perfect_validation()])

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
    assert state["family"]["name"] == "yolo_segmentation"
    assert state["family"]["task"] == "segment"
    assert state["epochs_completed"] == 1
    assert state["early_stopping"]["stopped_early"] is False

    # The canonical dataset is still the COCO store, untouched: the run holds
    # no YOLO labels and no second canonical format.
    canonical = _read(store / "dataset.json")
    assert canonical["images"][0]["file_name"].startswith("images/")
    assert not (output / "labels").exists()
    assert not (output / "data.yaml").exists()
    assert not list(output.rglob("*.txt"))

    # The job-time conversion lives in scratch only and is named in provenance.
    provenance = _read(output / "provenance.json")
    assert provenance["dataset_source"]["kind"] == "coco_instances"
    assert provenance["dataset_source"]["path"] == str(store)
    conversion = provenance["conversion"]
    assert conversion["format"] == "yolo-seg-polygons"
    assert conversion["cached"] is False
    converted = Path(conversion["root"])
    assert converted.is_relative_to(tmp_path / "scratch")
    assert converted != output and not converted.is_relative_to(output)
    assert Path(conversion["data_yaml"]).is_file()
    labels = sorted((converted / "labels" / "val").glob("*.txt"))
    assert len(labels) == 2
    first = labels[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(first) == 2
    for line in first:
        parts = line.split()
        assert len(parts) == 1 + len(SQUARE)  # class plus polygon coordinates
        assert 0 <= int(parts[0]) < len(NAMES)
        assert all(0.0 <= float(value) <= 1.0 for value in parts[1:])
    # PET is COCO category 1 but the family expects the sorted position 0.
    assert first[0].split()[0] == "0"

    # The adapter received the scratch declaration, not a canonical path.
    # The first wrapper is the training model; every later one is the
    # per-epoch validation snapshot loaded from disk.
    assert _FakeYOLO.instances[0].weights == "yolov8n-seg.pt"
    # Every later wrapper validates an immutable snapshot written to disk: one
    # per epoch, plus the scratch snapshot of the restored best weights that
    # the publication re-score predicts from.
    assert all(
        Path(instance.weights).name.startswith(("epoch_", ".validation"))
        for instance in _FakeYOLO.instances[1:]
    )

    # Checkpoint selection uses the with-class score; localisation is printed
    # beside it and the best weights are the kept epoch's own weights.
    from tests.test_pipeline_local_tracer import _same_weights

    printed = capsys.readouterr().out
    assert "SELECTION: with_class_score=1.000000" in printed
    assert "CRITERION: with-class operating-point F1" in printed
    assert "LOCALISATION: 1.000000" in printed
    assert "mask IoU" in printed
    assert _same_weights(
        output / "checkpoints" / "best.pt", output / "checkpoints" / "epoch_000.pt"
    )


def test_mask_iou_and_map_are_reported_per_class_with_the_criterion_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A perfect mask run scores 1.0 and every figure names mask IoU."""
    store = _imported_store(tmp_path)

    code, output = _train_segmentation(tmp_path, store, monkeypatch, [_shifted_validation(dy=2.0)])

    assert code == 0
    metrics = _read(output / "metrics.json")
    assert metrics["selection"]["metric"] == "with_class_score"
    criterion = metrics["selection"]["criterion"]
    assert "mask" in criterion.lower() and "iou" in criterion.lower()
    # A two-pixel vertical shift of a 20x20 square: exact polygon overlap is
    # 18x20 over a 440-pixel union, so IoU is 18/22. Every instance clears the
    # 0.50 operating point, so with-class F1 is 1.0 even though mAP@[0.50:0.95]
    # is lower.
    overlap = pytest.approx(18.0 / 22.0)
    assert metrics["selection"]["value"] == pytest.approx(1.0)
    assert metrics["selected_epoch"] == 0

    figures = metrics["figures"]
    for name in ("with_class_score", "localisation_score", "mask_iou", "mask_mAP50-95"):
        assert name in figures, f"missing figure: {name}"
        assert figures[name]["criterion"], f"figure {name} has no criterion"
    assert figures["mask_iou"]["value"] == overlap
    assert "mask" in figures["mask_iou"]["criterion"].lower()
    assert "localisation" in figures["localisation_score"]["criterion"].lower()
    assert figures["localisation_score"]["value"] == pytest.approx(1.0)
    assert figures["mask_mAP50-95"]["value"] == pytest.approx(0.7)
    assert figures["with_class_score"]["value"] == pytest.approx(1.0)

    per_class = metrics["per_class"]
    assert set(per_class) == set(NAMES)
    for name, class_figures in per_class.items():
        assert "mask_iou" in class_figures, name
        assert class_figures["mask_iou"]["value"] == overlap
        assert "mask" in class_figures["mask_iou"]["criterion"].lower()
        assert name in class_figures["mask_iou"]["criterion"]
        assert class_figures["mask_AP50"]["value"] == pytest.approx(1.0)

    # The report repeats the selection and localisation criteria side by side.
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "with_class_score" in report
    assert "localisation_score" in report
    assert "mask" in report.lower()
    assert "PET" in report and "PP" in report and "LDPE" in report


def test_segmentation_and_detection_share_the_same_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model is the only variable: the resolved protocol must not change."""
    from tests.test_pipeline_local_tracer import build_tiny_dataset

    store = _imported_store(tmp_path)
    _install_fakes(monkeypatch, _with_publication([_perfect_validation()]))
    seg_output = tmp_path / "executions" / "seg-protocol"
    assert (
        _command(
            f"coco_dataset={store}",
            f"output_dir={seg_output}",
            f"scratch_dir={tmp_path / 'scratch'}",
            "family=yolo_segmentation",
            "protocol.epochs=1",
        )
        == 0
    )

    data_yaml = build_tiny_dataset(tmp_path / "detection")
    from tests.test_pipeline_local_tracer import _boxes_prediction
    from tests.test_pipeline_local_tracer import _install_fakes as install

    install(monkeypatch, _with_publication([_boxes_prediction()]))
    det_output = tmp_path / "executions" / "det-protocol"
    assert (
        _command(
            f"data_yaml={data_yaml}",
            f"output_dir={det_output}",
            "protocol.epochs=1",
        )
        == 0
    )

    def protocol(path: Path) -> dict:
        return yaml.safe_load((path / "resolved_protocol.yaml").read_text(encoding="utf-8"))

    assert protocol(seg_output) == protocol(det_output)


def test_amp_records_the_precision_that_actually_ran_not_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lightning substitutes bf16-mixed for 16-mixed on CPU; say so honestly.

    The engine used to record the requested ``16-mixed`` as the precision that
    ran and derived ``amp_active`` from that request, so a CPU execution
    claimed FP16 AMP it never had.
    """
    store = _imported_store(tmp_path)
    network = _tiny_seg_net()
    _install_real_lightning_fakes(monkeypatch, network, _with_publication([_perfect_validation()]))
    output = tmp_path / "executions" / "amp-cpu"

    code = _command(
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        "family=yolo_segmentation",
        "protocol.epochs=1",
        "protocol.amp.enabled=true",
        "protocol.amp.dtype=fp16",
        "protocol.warmup_epochs=0",
    )

    assert code == 0
    state = _read(output / "training_state.json")
    # What Lightning says it ran, on this CPU host.
    assert state["precision"] == "bf16-mixed"
    assert state["amp_requested"] == "16-mixed"
    assert state["amp_active"] is True, "bf16-mixed is mixed precision, but not fp16"


def test_a_second_run_reuses_the_scratch_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conversion is a cache keyed by the canonical dataset, not a rewrite."""
    store = _imported_store(tmp_path)
    first_code, output = _train_segmentation(tmp_path, store, monkeypatch, [_perfect_validation()])
    assert first_code == 0
    first = _read(output / "provenance.json")["conversion"]
    assert first["cached"] is False

    _install_fakes(monkeypatch, _with_publication([_perfect_validation()]))
    second_output = tmp_path / "executions" / "segmentation-second"
    second_code = _command(
        f"coco_dataset={store}",
        f"output_dir={second_output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        "family=yolo_segmentation",
        "protocol.epochs=1",
    )

    assert second_code == 0
    second = _read(second_output / "provenance.json")["conversion"]
    assert second["cached"] is True
    assert second["cache_key"] == first["cache_key"]
    assert second["root"] == first["root"]


def test_changed_image_bytes_do_not_reuse_the_conversion_cache(tmp_path: Path) -> None:
    """The cache key includes pixels, not only dataset.json."""
    from polyx.pipeline.seg_conversion import convert_coco_dataset

    store = _imported_store(tmp_path)
    first = convert_coco_dataset(store, tmp_path / "scratch-bytes")
    image = store / "images" / "val" / "plate_v1.png"
    image.write_bytes(image.read_bytes() + b"\0")
    second = convert_coco_dataset(store, tmp_path / "scratch-bytes")
    assert first.cache_key != second.cache_key
    assert first.root != second.root


def test_provenance_records_the_pixels_the_conversion_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store modified under the same version must not report the old hashes."""
    import hashlib

    store = _imported_store(tmp_path)
    image = store / "images" / "val" / "plate_v1.png"
    image.write_bytes(image.read_bytes() + b"\0")
    expected = hashlib.sha256(image.read_bytes()).hexdigest()

    code, output = _train_segmentation(tmp_path, store, monkeypatch, [_perfect_validation()])
    assert code == 0
    hashes = _read(output / "provenance.json")["dataset_hashes"]
    assert expected in hashes.values()


def test_masks_are_matched_within_the_same_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical geometry in another image is a false positive, not a match."""
    store = _imported_store(tmp_path)
    duplicated = [
        # The first image only shows PET: its PP mask is not predicted.
        [_detection(0, OFFSETS[0])],
        # Both PP predictions belong to the second image; the first image's PP
        # truth has the same geometry and must not absorb the duplicate.
        [_detection(2, OFFSETS[0]), _detection(1, OFFSETS[1]), _detection(1, OFFSETS[1])],
    ]

    code, output = _train_segmentation(tmp_path, store, monkeypatch, [duplicated])

    assert code == 0
    metrics = _read(output / "metrics.json")
    # Three true positives, one false positive and one false negative survive
    # image-scoped matching; the micro operating-point F1 is 0.75. A
    # cross-image match would have hidden the duplicate and scored 1.0.
    assert metrics["figures"]["mask_iou"]["value"] == pytest.approx(0.75)
    assert metrics["selection"]["value"] == pytest.approx(0.75)


def test_mask_iou_is_class_aware_like_its_criterion_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identical mask with the wrong polymer is not an overlap.

    ``_best_iou`` ignored ``category_id``, so a perfect PP prediction against
    a PET ground truth scored ``mask_iou = 1.0`` while the with-class selection
    correctly scored zero, contradicting the figure's own class-aware
    criterion.
    """
    store = _imported_store(tmp_path)
    wrong_class = [
        [{**detection, "class_id": (detection["class_id"] + 1) % len(NAMES)} for detection in image]
        for image in _perfect_validation()
    ]

    code, output = _train_segmentation(tmp_path, store, monkeypatch, [wrong_class])

    assert code == 0
    metrics = _read(output / "metrics.json")
    assert metrics["selection"]["value"] == pytest.approx(0.0)
    # The core definition keeps a misclassified box out of both criteria's
    # true-positive numerator; localisation cannot claim it either.
    assert metrics["figures"]["localisation_score"]["value"] == pytest.approx(0.0)
    assert metrics["figures"]["mask_iou"]["value"] == pytest.approx(0.0)
    assert "class-aware" in metrics["figures"]["mask_iou"]["criterion"]
    for name, class_figures in metrics["per_class"].items():
        assert class_figures["mask_iou"]["value"] == pytest.approx(0.0), name


def test_segmentation_family_requires_the_coco_mask_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A segmenter is never trained from a box dataset; the refusal is loud."""
    from tests.test_pipeline_local_tracer import build_tiny_dataset

    _install_fakes(monkeypatch, [])
    data_yaml = build_tiny_dataset(tmp_path / "dataset")

    code = _command(
        f"data_yaml={data_yaml}",
        f"output_dir={tmp_path / 'executions' / 'wrong-source'}",
        "family=yolo_segmentation",
    )

    assert code == 1
    assert "coco mask dataset" in capsys.readouterr().err.lower()
    assert not (tmp_path / "executions" / "wrong-source" / "training_state.json").exists()


def test_detection_family_refuses_the_coco_mask_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The symmetric mistake is refused too: detection stays box-only."""
    store = _imported_store(tmp_path)

    code = _command(
        f"coco_dataset={store}",
        f"output_dir={tmp_path / 'executions' / 'wrong-family'}",
        "family=yolo_detection",
    )

    assert code == 1
    assert "segmentation family" in capsys.readouterr().err.lower()


def test_one_epoch_completes_under_the_real_lightning_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper must be a Lightning module, not merely import like one.

    Only Ultralytics is faked here: the real Lightning ``Trainer`` runs the
    epoch, builds the converted dataset, calls ``training_step``/the optimizer
    and reaches the validation hook that Ultralytics-backed evaluation uses.
    A wrapper without those hooks fails before writing any artifact.
    """
    store = _imported_store(tmp_path)
    network = _tiny_seg_net()
    _install_real_lightning_fakes(monkeypatch, network, _with_publication([_perfect_validation()]))
    output = tmp_path / "executions" / "real-loop"

    code = _command(
        f"coco_dataset={store}",
        f"output_dir={output}",
        f"scratch_dir={tmp_path / 'scratch'}",
        "family=yolo_segmentation",
        "protocol.epochs=1",
        "protocol.amp.enabled=false",
        "protocol.warmup_epochs=0",
    )

    assert code == 0, "the real Lightning loop rejected the adapter"
    assert network.seen_batches >= 1, "Lightning never ran a training step"
    assert network.mask_shapes, "the converted dataset produced no masks"
    assert all(len(shape) == 3 for shape in network.mask_shapes)
    state = _read(output / "training_state.json")
    assert state["epochs_completed"] == 1
    assert (output / "checkpoints" / "epoch_000.pt").is_file()
    assert _read(output / "metrics.json")["selection"]["value"] == pytest.approx(1.0)
