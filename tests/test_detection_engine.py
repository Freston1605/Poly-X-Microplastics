"""Contract tests for the canonical detection run engine and run folder.

Ticket #58: ``run_detection`` executes N already-loaded models over a set of
images, writes the canonical run folder as each result is produced, and
``load_run`` reconstructs the same Qt-free result from that folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from polyx.core.detect.protocol import Instance
from polyx.core.detect.tiling import TilingDecision, TilingMode, TilingPlan, describe_tiling
from polyx.core.detection_run import (
    CalibrationPolicy,
    ModelSpec,
    RunConfig,
    RunDirNotEmpty,
    RunFailed,
    RunOutputError,
    RunOutputNotADirectory,
    RunParams,
    RunStatus,
    load_run,
    run_detection,
)
from polyx.core.detection_run import engine as engine_module
from polyx.core.measure.calibration import ORIGIN_MANUAL, ORIGIN_NONE
from polyx.core.quality.metrics import MatchResult

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

TOP_BOX = (20.0, 20.0, 60.0, 60.0)
BOTTOM_BOX = (90.0, 90.0, 130.0, 130.0)


def write_plate(path: Path, width: int = 160, height: int = 160) -> Path:
    """Write a plate with two bright squares, so boxes have something to measure."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in (TOP_BOX, BOTTOM_BOX):
        img[int(y1) + 5 : int(y2) - 5, int(x1) + 5 : int(x2) - 5] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path


class FakeDetector:
    """A loaded detection source: no weights, no tiling, fixed boxes."""

    def __init__(
        self,
        boxes: tuple[tuple[float, float, float, float], ...] = (TOP_BOX, BOTTOM_BOX),
        names: dict[int, str] | None = None,
    ):
        self.boxes = boxes
        self._names = {0: "PET", 1: "PP"} if names is None else names
        self.calls: list[tuple[Path, dict]] = []
        self.last_imgsz = 0
        self.last_fallback = False

    @property
    def names(self) -> dict[int, str]:
        return self._names

    def predict_auto(self, image_path, **kwargs) -> list[Instance]:
        self.calls.append((Path(image_path), dict(kwargs)))
        self.last_imgsz = int(kwargs.get("imgsz", 0) or 0)
        return [
            Instance(
                class_id=index % 2,
                class_name=self._names.get(index % 2, str(index % 2)),
                conf=0.9 - 0.1 * index,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            for index, (x1, y1, x2, y2) in enumerate(self.boxes)
        ]


def make_config(
    images: list[Path],
    models: list[FakeDetector],
    output_dir: Path,
    *,
    gt_folder: Path | None = None,
    measure_shape: bool = False,
    calibration: CalibrationPolicy | None = None,
    force: bool = False,
    iou_tp: float = 0.5,
) -> RunConfig:
    return RunConfig(
        images=images,
        models=[
            ModelSpec(detector=model, alias=f"model_{chr(97 + i)}")
            for i, model in enumerate(models)
        ],
        output_dir=output_dir,
        params=RunParams(measure_shape=measure_shape, iou_tp=iou_tp),
        gt_folder=gt_folder,
        calibration=calibration or CalibrationPolicy(mode="manual", um_per_px=2.0),
        force=force,
    )


def test_two_models_write_the_canonical_folder_and_round_trip(tmp_path):
    images = [
        write_plate(tmp_path / "photos" / "plate1.png"),
        write_plate(tmp_path / "photos" / "plate2.png"),
    ]
    model_a = FakeDetector()
    model_b = FakeDetector(boxes=(TOP_BOX,))
    run_dir = tmp_path / "run"

    result = run_detection(make_config(images, [model_a, model_b], run_dir))

    assert result.status is RunStatus.COMPLETED
    assert result.run_dir == run_dir
    assert result.images == images
    assert [model.alias for model in result.models] == ["model_a", "model_b"]
    assert set(result.results) == {0, 1}
    assert [item.image_path for item in result.results[0]] == images
    assert [item.image_path for item in result.results[1]] == images
    assert [len(item.predictions) for item in result.results[0]] == [2, 2]
    assert [len(item.predictions) for item in result.results[1]] == [1, 1]
    assert result.calibrations[images[0]].origin == ORIGIN_MANUAL
    assert result.calibrations[images[0]].um_por_px == pytest.approx(2.0)

    # Each model has its own folder with the three views that exist and the
    # prediction labels (6 columns: the confidence is the extra one).
    for model in ("model_a", "model_b"):
        folder = run_dir / model
        first_annot = folder / "0000_plate1_annot.png"
        first_pred = folder / "0000_plate1_pred.png"
        assert first_annot.is_file() and first_pred.is_file()
        assert not (folder / "0000_plate1_gt.png").exists()
        labels = folder / "labels" / "0000_plate1.txt"
        lines = labels.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == (2 if model == "model_a" else 1)
        assert all(len(line.split()) == 6 for line in lines)

    # run.json and detections.csv are the canonical machine surface.
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert [model["alias"] for model in payload["models"]] == ["model_a", "model_b"]
    assert [entry["path"] for entry in payload["images"]] == [str(path) for path in images]
    assert payload["error"] is None

    with (run_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.results[0]) * 2 + len(result.results[1]) * 1
    assert set(rows[0]) == {
        "model",
        "image",
        "number",
        "class_id",
        "class",
        "conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "um_per_px",
        "calibration_origin",
        "length_um",
        "method_length",
        "feret_um",
        "geodesic_um",
        "equivalent_rect_length_um",
        "width_um",
        "area_um2",
        "diam_um",
        "aspect",
        "curvature",
        "morphotype",
        "size_source",
        "has_mask",
        "assignable",
    }
    assert {row["model"] for row in rows} == {"model_a", "model_b"}
    assert {row["assignable"] for row in rows} == {"True"}
    assert {row["image"] for row in rows} == {str(path) for path in images}

    # load_run rebuilds the same result from the folder alone.
    loaded = load_run(run_dir)
    assert loaded.status == result.status
    assert loaded.images == result.images
    assert [model.alias for model in loaded.models] == [model.alias for model in result.models]
    assert loaded.params.conf == result.params.conf
    assert loaded.params.imgsz == result.params.imgsz
    assert set(loaded.results) == set(result.results)
    for model_idx in result.results:
        assert len(loaded.results[model_idx]) == len(result.results[model_idx])
        for original, rebuilt in zip(
            result.results[model_idx], loaded.results[model_idx], strict=True
        ):
            assert rebuilt.image_path == original.image_path
            assert rebuilt.model_idx == original.model_idx
            assert rebuilt.has_gt == original.has_gt
            assert len(rebuilt.predictions) == len(original.predictions)
            assert (rebuilt.tp, rebuilt.fp, rebuilt.fn, rebuilt.miscls) == (
                original.tp,
                original.fp,
                original.fn,
                original.miscls,
            )
            assert rebuilt.annotated_path == original.annotated_path
            assert rebuilt.pred_path == original.pred_path
            assert rebuilt.gt_path == original.gt_path
            for before, after in zip(original.predictions, rebuilt.predictions, strict=True):
                assert after.class_id == before.class_id
                assert after.class_name == before.class_name
                assert after.conf == pytest.approx(before.conf)
                assert (after.x1, after.y1, after.x2, after.y2) == pytest.approx(
                    (before.x1, before.y1, before.x2, before.y2)
                )
    assert loaded.calibrations[images[0]].um_por_px == pytest.approx(2.0)


def test_ground_truth_is_read_once_and_matched_per_model(tmp_path, monkeypatch):
    image = write_plate(tmp_path / "photos" / "plate.png")
    width = height = 160
    x1, y1, x2, y2 = TOP_BOX
    (image.parent / "plate.txt").write_text(
        f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n",
        encoding="utf-8",
    )
    model_a = FakeDetector(boxes=(TOP_BOX, BOTTOM_BOX))
    model_b = FakeDetector(boxes=(BOTTOM_BOX,))

    calls: list[Path] = []
    original = engine_module.read_label_file

    def counted(path, *args, **kwargs):
        calls.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(engine_module, "read_label_file", counted)
    result = run_detection(make_config([image], [model_a, model_b], tmp_path / "run"))

    # Two models, one image: ground truth is a property of the image.
    assert calls == [image.parent / "plate.txt"]
    first, second = result.results[0][0], result.results[1][0]
    assert first.has_gt and second.has_gt
    assert (first.tp, first.fp, first.fn, first.miscls) == (1, 1, 0, 0)
    assert (second.tp, second.fp, second.fn, second.miscls) == (0, 1, 1, 0)
    assert first.gt_path and first.gt_path.name == "0000_plate_gt.png"
    assert first.gt_path.is_file()
    assert second.gt_path and second.gt_path.is_file()

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["images"][0]["has_gt"] is True
    assert len(payload["images"][0]["gt"]) == 1
    assert payload["images"][0]["models"][0]["tp"] == 1

    loaded = load_run(result.run_dir)
    assert loaded.has_gt
    assert len(loaded.results[0][0].gt) == 1
    assert loaded.results[0][0].gt[0].class_id == 0
    assert loaded.results[1][0].gt_path and loaded.results[1][0].gt_path.is_file()


def test_split_ground_truth_is_found_through_the_conventional_layout(tmp_path, monkeypatch):
    """Ticket #92: a run over images/<split> reads labels/<split>."""
    dataset = tmp_path / "dataset"
    image = write_plate(dataset / "images" / "train" / "plate.png")
    width = height = 160

    def label_line(box: tuple[float, float, float, float]) -> str:
        x1, y1, x2, y2 = box
        return (
            f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n"
        )

    label = dataset / "labels" / "train" / "plate.txt"
    label.parent.mkdir(parents=True)
    label.write_text(label_line(TOP_BOX), encoding="utf-8")
    # A legacy file next to the image must lose to the conventional label.
    (image.parent / "plate.txt").write_text(label_line(BOTTOM_BOX), encoding="utf-8")

    read_paths: list[Path] = []
    original = engine_module.read_label_file

    def counted(path, *args, **kwargs):
        read_paths.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(engine_module, "read_label_file", counted)
    result = run_detection(
        make_config([image], [FakeDetector(boxes=(TOP_BOX, BOTTOM_BOX))], tmp_path / "run")
    )

    item = result.results[0][0]
    assert read_paths == [label]
    assert item.has_gt is True
    assert [gt.class_id for gt in item.gt] == [0]
    assert (item.tp, item.fp, item.fn, item.miscls) == (1, 1, 0, 0)
    # The GT view is rendered from the split-layout label, not the decoy.
    assert item.gt_path is not None and item.gt_path.name == "0000_plate_gt.png"
    assert item.gt_path.is_file()

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["images"][0]["has_gt"] is True
    assert len(payload["images"][0]["gt"]) == 1
    assert payload["images"][0]["models"][0]["tp"] == 1

    loaded = load_run(result.run_dir)
    assert loaded.results[0][0].has_gt is True
    assert len(loaded.results[0][0].gt) == 1
    assert loaded.results[0][0].gt_path is not None
    assert loaded.results[0][0].gt_path.is_file()


def test_callbacks_report_each_result_after_its_files_exist(tmp_path):
    images = [
        write_plate(tmp_path / "photos" / "a.png"),
        write_plate(tmp_path / "photos" / "b.png"),
    ]
    models = [FakeDetector(), FakeDetector(boxes=(TOP_BOX,))]
    progress: list[tuple[int, int, Path]] = []
    results: list[object] = []

    def on_result(image_result):
        # The callback contract: the files of this result already exist.
        assert image_result.annotated_path and image_result.annotated_path.is_file()
        assert image_result.pred_path and image_result.pred_path.is_file()
        results.append(image_result)

    run_detection(
        make_config(images, models, tmp_path / "run"),
        on_progress=lambda done, total, path: progress.append((done, total, path)),
        on_result=on_result,
    )

    assert len(results) == len(images) * len(models)
    assert progress[0] == (1, 4, images[0])
    assert progress[-1] == (4, 4, images[1])
    # Progress is monotonic and never doubles a model/image pair.
    assert [entry[0] for entry in progress] == [1, 2, 3, 4]


def test_abort_is_honoured_at_the_image_boundary(tmp_path):
    images = [
        write_plate(tmp_path / "photos" / "a.png"),
        write_plate(tmp_path / "photos" / "b.png"),
    ]
    models = [FakeDetector(), FakeDetector(boxes=(TOP_BOX,))]
    calls = {"abort": 0}

    def should_abort():
        calls["abort"] += 1
        return calls["abort"] > 1

    result = run_detection(make_config(images, models, tmp_path / "run"), should_abort=should_abort)

    assert result.status == "aborted"
    assert result.error is None
    assert result.finished_at is not None
    assert [item.image_path for item in result.results[0]] == [images[0]]
    assert [item.image_path for item in result.results[1]] == [images[0]]
    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "aborted"
    assert [entry["models"] for entry in payload["images"]] == [
        [
            {
                "model": 0,
                "has_gt": False,
                "predictions": 2,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "miscls": 0,
                "match_iou_thr": 0.5,
                "match_use_mask_iou": False,
                "tiling_plan": None,
                "tiling": "",
            },
            {
                "model": 1,
                "has_gt": False,
                "predictions": 1,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "miscls": 0,
                "match_iou_thr": 0.5,
                "match_use_mask_iou": False,
                "tiling_plan": None,
                "tiling": "",
            },
        ],
        [],
    ]
    loaded = load_run(result.run_dir)
    assert loaded.status == "aborted"
    assert {index: len(rows) for index, rows in loaded.results.items()} == {0: 1, 1: 1}


def test_unreadable_images_are_skipped_and_advance_progress(tmp_path):
    good = write_plate(tmp_path / "photos" / "good.png")
    broken = tmp_path / "photos" / "broken.png"
    broken.write_bytes(b"not an image")
    models = [FakeDetector(), FakeDetector(boxes=(TOP_BOX,))]
    progress: list[tuple[int, int, Path]] = []

    result = run_detection(
        make_config([good, broken], models, tmp_path / "run"),
        on_progress=lambda done, total, path: progress.append((done, total, path)),
    )

    assert result.status == "completed"
    assert result.skipped_images == [broken]
    assert [item.image_path for item in result.results[0]] == [good]
    assert [entry[0] for entry in progress] == [1, 2, 4]
    assert progress[-1] == (4, 4, broken)
    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["skipped_images"] == [str(broken)]
    assert payload["images"][1]["skipped"] is True
    assert payload["images"][1]["models"] == []
    loaded = load_run(result.run_dir)
    assert loaded.skipped_images == [broken]
    assert [item.image_path for item in loaded.results[0]] == [good]


def test_non_empty_output_folder_is_refused_unless_forced(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "someone-else.txt").write_text("keep me", encoding="utf-8")
    config = make_config([image], [FakeDetector()], run_dir)

    with pytest.raises(RunDirNotEmpty) as error:
        run_detection(config)
    assert Path(error.value.path) == run_dir
    assert not (run_dir / "run.json").exists()
    assert (run_dir / "someone-else.txt").read_text(encoding="utf-8") == "keep me"

    forced = make_config([image], [FakeDetector()], run_dir, force=True)
    result = run_detection(forced)
    assert result.status is RunStatus.COMPLETED
    assert (run_dir / "run.json").is_file()
    # Force overwrites the canonical files but never deletes unrelated content.
    assert (run_dir / "someone-else.txt").read_text(encoding="utf-8") == "keep me"


def test_output_path_that_is_a_file_is_refused_with_its_own_error(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    output = tmp_path / "not-a-folder.txt"
    output.write_text("someone else's file", encoding="utf-8")
    config = make_config([image], [FakeDetector()], output)

    with pytest.raises(RunOutputNotADirectory) as error:
        run_detection(config)

    assert isinstance(error.value, RunOutputError)
    assert Path(error.value.path) == output
    assert not hasattr(error.value, "run_dir")
    assert output.read_text(encoding="utf-8") == "someone else's file"


def test_unexpected_error_writes_a_failed_run_and_raises_run_failed(tmp_path):
    images = [
        write_plate(tmp_path / "photos" / "ok.png"),
        write_plate(tmp_path / "photos" / "boom.png"),
    ]

    class ExplodingDetector(FakeDetector):
        def predict_auto(self, image_path, **kwargs):
            if Path(image_path).name == "boom.png":
                raise RuntimeError("inference exploded")
            return super().predict_auto(image_path, **kwargs)

    config = make_config(images, [ExplodingDetector()], tmp_path / "run")

    with pytest.raises(RunFailed) as error:
        run_detection(config)

    partial = error.value.result
    assert not hasattr(error.value, "run_result")
    assert partial.status is RunStatus.FAILED
    assert partial.error and "inference exploded" in partial.error
    assert [item.image_path for item in partial.results[0]] == [images[0]]
    payload = json.loads((partial.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "inference exploded" in payload["error"]
    assert len(payload["images"][0]["models"]) == 1
    assert payload["images"][1]["models"] == []
    loaded = load_run(partial.run_dir)
    assert loaded.status == "failed"
    assert loaded.error == partial.error
    assert [item.image_path for item in loaded.results[0]] == [images[0]]


def test_setup_error_writes_a_failed_run_and_raises_run_failed(tmp_path, monkeypatch):
    """A setup error before the loop still leaves a partial record."""
    image = write_plate(tmp_path / "photos" / "plate.png")

    def _explode(_path):
        raise RuntimeError("fingerprint exploded")

    monkeypatch.setattr(engine_module, "sha256_file", _explode)
    config = RunConfig(
        images=[image],
        models=[ModelSpec(detector=FakeDetector(), alias="same", path=image)],
        output_dir=tmp_path / "run",
        params=RunParams(),
        calibration=CalibrationPolicy(mode="none"),
    )

    with pytest.raises(RunFailed) as error:
        run_detection(config)

    assert error.value.result.status is RunStatus.FAILED
    assert "fingerprint exploded" in (error.value.result.error or "")
    payload = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["models"] == []


def test_duplicate_model_aliases_disambiguate_instead_of_failing(tmp_path):
    """Two models with the same name get sibling folders, not an error (#124)."""
    image = write_plate(tmp_path / "photos" / "plate.png")
    config = RunConfig(
        images=[image],
        models=[
            ModelSpec(detector=FakeDetector(), alias="same"),
            ModelSpec(detector=FakeDetector(), alias="same"),
        ],
        output_dir=tmp_path / "run",
        params=RunParams(),
        calibration=CalibrationPolicy(mode="none"),
    )

    result = run_detection(config)

    assert result.status is RunStatus.COMPLETED
    assert [model.alias for model in result.models] == ["same", "same_2"]
    assert (tmp_path / "run" / "same" / "0000_plate_annot.png").is_file()
    assert (tmp_path / "run" / "same_2" / "0000_plate_annot.png").is_file()
    loaded = load_run(tmp_path / "run")
    assert [model.alias for model in loaded.models] == ["same", "same_2"]


def test_none_calibration_invents_no_micrometre_sizes(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    result = run_detection(
        make_config(
            [image],
            [FakeDetector()],
            tmp_path / "run",
            calibration=CalibrationPolicy(mode="none", um_per_px=9.0),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_NONE
    assert calibration.um_por_px == 0.0
    assert all(det.diam_um is None for det in result.results[0][0].predictions)
    with (result.run_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["um_per_px"] for row in rows} == {""}
    assert {row["calibration_origin"] for row in rows} == {ORIGIN_NONE}


def test_predict_auto_is_called_with_the_run_tiling_policy(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    detector = FakeDetector()
    result = run_detection(make_config([image], [detector], tmp_path / "run"))

    call_path, kwargs = detector.calls[0]
    assert call_path == image
    assert kwargs["tiling"] is result.params.tiling is TilingMode.AUTO
    assert kwargs["threshold_px"] == result.params.tiling_threshold_px
    assert kwargs["tile"] == result.params.tiling_tile
    assert kwargs["overlap"] == result.params.tiling_overlap
    assert kwargs["conf"] == result.params.conf
    assert kwargs["iou"] == result.params.iou_nms
    assert kwargs["imgsz"] == result.params.imgsz
    assert kwargs["record"] == {}
    assert result.results[0][0].tiling == ""
    assert result.results[0][0].tiling_plan is None


class TilingDetector(FakeDetector):
    """Reports the decision it took, the way a tiling backend does."""

    plan = TilingPlan(tile=2048, overlap=0.25, n_tiles=6, side=4096, threshold_px=2000)

    def predict_auto(self, image_path, **kwargs):
        record = kwargs.get("record")
        if record is not None:
            record.update(
                {
                    "width": 4096,
                    "height": 3072,
                    "tiled": True,
                    "plan": self.plan,
                }
            )
        return super().predict_auto(image_path, **kwargs)


def test_the_engine_records_the_plan_and_describes_it_once(tmp_path):
    """#125: the record holds the decision; the sentence is derived from it."""
    image = write_plate(tmp_path / "photos" / "plate.png")
    detector = TilingDetector()
    result = run_detection(make_config([image], [detector], tmp_path / "run"))

    image_result = result.results[0][0]
    assert image_result.tiling_plan == TilingDetector.plan
    expected_sentence = describe_tiling(
        TilingDecision(
            mode=TilingMode.AUTO,
            width=4096,
            height=3072,
            plan=TilingDetector.plan,
        )
    )
    assert image_result.tiling == expected_sentence
    assert result.execution[0].tiled_images == 1
    assert result.execution[0].imgsz_used == [2048]

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    row = payload["images"][0]["models"][0]
    assert row["tiling_plan"] == {
        "tile": 2048,
        "overlap": 0.25,
        "n_tiles": 6,
        "side": 4096,
        "threshold_px": 2000,
    }
    assert row["tiling"] == image_result.tiling

    loaded = load_run(result.run_dir).results[0][0]
    assert loaded.tiling_plan == TilingDetector.plan
    assert loaded.tiling == image_result.tiling


def test_measure_shape_false_keeps_the_box_fallback(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    result = run_detection(
        make_config([image], [FakeDetector()], tmp_path / "run", measure_shape=False)
    )

    for det in result.results[0][0].predictions:
        assert det.size_source == "box"
        assert det.diam_um is not None
        assert det.area_um2 is not None


def test_manual_calibration_records_origin_and_sizes(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    result = run_detection(
        make_config(
            [image],
            [FakeDetector()],
            tmp_path / "run",
            calibration=CalibrationPolicy(mode="manual", um_per_px=1.5),
        )
    )

    assert result.calibrations[image].origin == ORIGIN_MANUAL
    assert all(det.diam_um is not None for det in result.results[0][0].predictions)
    with (result.run_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["um_per_px"] for row in rows} == {"1.5"}
    assert {row["calibration_origin"] for row in rows} == {ORIGIN_MANUAL}
    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["calibrations"][0]["um_por_px"] == pytest.approx(1.5)
    assert payload["calibrations"][0]["origin"] == ORIGIN_MANUAL
    assert payload["calibrations"][0]["valid"] is True


def test_model_metadata_and_fingerprint_are_recorded(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    detector = FakeDetector(names={0: "PET", 1: "PP"})
    config = RunConfig(
        images=[image],
        models=[
            ModelSpec(
                detector=detector,
                alias="seg_model",
                path=weights,
                model_type="yolo-seg",
                names={0: "CUSTOM"},
            )
        ],
        output_dir=tmp_path / "run",
        params=RunParams(measure_shape=False),
        calibration=CalibrationPolicy(mode="manual", um_per_px=1.0),
    )

    result = run_detection(config)
    expected_sha = hashlib.sha256(b"weights").hexdigest()
    assert result.models[0].names == {0: "CUSTOM"}
    assert result.models[0].sha256 == expected_sha

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    model = payload["models"][0]
    assert model["alias"] == "seg_model"
    assert model["path"] == str(weights)
    assert model["model_type"] == "yolo-seg"
    assert model["names"] == {"0": "CUSTOM"}
    assert model["sha256"] == expected_sha
    assert model["execution"]["images"] == 1
    assert model["execution"]["detections"] == 2

    loaded = load_run(result.run_dir)
    assert loaded.models[0].alias == "seg_model"
    assert loaded.models[0].path == weights
    assert loaded.models[0].names == {0: "CUSTOM"}
    assert loaded.models[0].sha256 == expected_sha
    assert loaded.execution[0].images == 1
    assert loaded.execution[0].detections == 2


def test_a_copied_run_folder_loads_without_the_source_sidecars(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    width = height = 160
    x1, y1, x2, y2 = TOP_BOX
    (image.parent / "plate.txt").write_text(
        f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n",
        encoding="utf-8",
    )
    original = run_detection(make_config([image], [FakeDetector()], tmp_path / "run"))

    copied = tmp_path / "moved-run"
    shutil.copytree(original.run_dir, copied)
    for sidecar in image.parent.glob("*.txt"):
        sidecar.unlink()

    loaded = load_run(copied)
    assert loaded.status == "completed"
    assert loaded.has_gt
    assert len(loaded.results[0][0].gt) == 1
    assert len(loaded.results[0][0].predictions) == 2
    assert loaded.results[0][0].annotated_path == copied / "model_a" / "0000_plate_annot.png"
    assert loaded.results[0][0].annotated_path.is_file()
    assert loaded.results[0][0].gt_path == copied / "model_a" / "0000_plate_gt.png"
    assert loaded.results[0][0].gt_path.is_file()


def test_engine_executes_and_loads_without_importing_qt(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    out = tmp_path / "run"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from polyx.core.detect.protocol import Instance\n"
        "from polyx.core.detection_run import (\n"
        "    CalibrationPolicy, ModelSpec, RunConfig, RunParams, load_run, run_detection,\n"
        ")\n"
        "class Detector:\n"
        "    names = {0: 'PET'}\n"
        "    last_imgsz = 640\n"
        "    last_fallback = False\n"
        "    def predict_auto(self, image_path, **kwargs):\n"
        "        return [Instance(class_id=0, class_name='PET', conf=0.9,\n"
        "                         x1=20, y1=20, x2=60, y2=60)]\n"
        "image, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "result = run_detection(RunConfig(\n"
        "    images=[image],\n"
        "    models=[ModelSpec(detector=Detector(), alias='model')],\n"
        "    output_dir=out,\n"
        "    params=RunParams(measure_shape=False),\n"
        "    calibration=CalibrationPolicy(mode='manual', um_per_px=1.0),\n"
        "))\n"
        "loaded = load_run(out)\n"
        "assert result.status == 'completed' and loaded.status == 'completed'\n"
        "assert len(loaded.results[0][0].predictions) == 1\n"
        "bad = sorted(m for m in sys.modules if m == 'PySide6' or m.startswith('PySide6.'))\n"
        "print('QT_LOADED=' + repr(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    env["POLYX_LOG_DIR"] = str(tmp_path / "logs")
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, "-c", script, str(image), str(out)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        "executing/loading a run dragged Qt or failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "QT_LOADED=[]" in result.stdout


# ── The match the engine decided (ticket #120) ─────────────────────────
#
# The engine has always decided which prediction is a true positive, a false
# positive, a missed label or a misclassified pair. These tests pin that the
# decision itself crosses the seam, instead of only its four totals: readers
# used to rebuild their own match and could disagree with the run.

_DISJOINT_BOXES = {
    "top": (20.0, 20.0, 60.0, 60.0),
    "bottom": (90.0, 90.0, 130.0, 130.0),
    "left-low": (20.0, 90.0, 60.0, 130.0),
    "right-high": (90.0, 20.0, 130.0, 60.0),
}


class ScriptedDetector:
    """A loaded detection source that returns exactly the instances it was given.

    The engine measures, filters and numbers in place, so every call hands back
    a fresh deep copy; masks included, because ``_uses_mask_iou`` reads them.
    """

    def __init__(self, detections: list[Instance], names: dict[int, str] | None = None):
        self._detections = detections
        self._names = names or {0: "PET", 1: "PP"}
        self.calls: list[Path] = []
        self.last_imgsz = 0
        self.last_fallback = False

    @property
    def names(self) -> dict[int, str]:
        return self._names

    def predict_auto(self, image_path, **kwargs) -> list[Instance]:
        self.calls.append(Path(image_path))
        self.last_imgsz = int(kwargs.get("imgsz", 0) or 0)
        return [deepcopy(det) for det in self._detections]


def _detection(class_id: int, conf: float, box: tuple[float, float, float, float], **extra):
    x1, y1, x2, y2 = box
    return Instance(
        class_id=class_id,
        class_name="PET" if class_id == 0 else "PP",
        conf=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        **extra,
    )


def _label_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    """One YOLO label line for a 160x160 plate."""
    x1, y1, x2, y2 = box
    return (
        f"{class_id} {(x1 + x2) / 2 / 160:.6f} {(y1 + y2) / 2 / 160:.6f} "
        f"{(x2 - x1) / 160:.6f} {(y2 - y1) / 160:.6f}\n"
    )


def _matching_plate(tmp_path: Path) -> Path:
    """A plate with one label of every outcome: TP, FP, misclassified, missed."""
    image = write_plate(tmp_path / "photos" / "plate.png")
    (image.parent / "plate.txt").write_text(
        _label_line(0, _DISJOINT_BOXES["top"])
        + _label_line(1, _DISJOINT_BOXES["left-low"])
        + _label_line(1, _DISJOINT_BOXES["right-high"]),
        encoding="utf-8",
    )
    return image


def _matching_detector() -> ScriptedDetector:
    """Predictions: a PET on the PET label, a stray, and a PET on a PP label."""
    return ScriptedDetector(
        [
            _detection(0, 0.90, _DISJOINT_BOXES["top"]),
            _detection(0, 0.80, _DISJOINT_BOXES["bottom"]),
            _detection(0, 0.70, _DISJOINT_BOXES["left-low"]),
        ]
    )


def test_engine_attaches_the_match_it_performed(tmp_path):
    """The engine keeps the decision, not just its four totals."""
    image = _matching_plate(tmp_path)
    result = run_detection(make_config([image], [_matching_detector()], tmp_path / "run"))
    item = result.results[0][0]

    match = item.match
    assert isinstance(match, MatchResult)
    assert (item.tp, item.fp, item.fn, item.miscls) == (1, 1, 1, 1)
    assert (match.tp, match.fp, match.fn, match.miscls) == (1, 1, 1, 1)
    # Prediction 0 is the TP on GT 0; prediction 1 has no GT; prediction 2 sits
    # on the PP label while claiming PET; GT 2 is never found.
    assert match.tp_pairs == [(0, 0)]
    assert match.fp_idx == [1]
    assert match.miscls_pairs == [(2, 1)]
    assert match.fn_idx == [2]
    assert item.match_iou_thr == pytest.approx(result.params.iou_tp)
    assert item.match_use_mask_iou is False


def test_run_json_records_the_match_rule_and_load_rehydrates_the_match(tmp_path):
    """The rule survives a reload; the pair lists do not need to be written.

    GT and predictions are already in the folder, so one ``match_image`` call
    with the recorded rule rebuilds the decision. Persisting the indices would
    grow the document to say what it already says.
    """
    image = _matching_plate(tmp_path)
    # A non-default threshold, so the round-trip proves the recorded value is
    # read back instead of the dataclass default standing in for it.
    result = run_detection(
        make_config([image], [_matching_detector()], tmp_path / "run", iou_tp=0.7)
    )

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    row = payload["images"][0]["models"][0]
    assert row["match_iou_thr"] == pytest.approx(0.7)
    assert row["match_use_mask_iou"] is False
    assert "tp_pairs" not in row and "fp_idx" not in row

    loaded = load_run(result.run_dir)
    item = loaded.results[0][0]
    assert isinstance(item.match, MatchResult)
    assert item.match.tp_pairs == [(0, 0)]
    assert item.match.fp_idx == [1]
    assert item.match.miscls_pairs == [(2, 1)]
    assert item.match.fn_idx == [2]
    assert (item.match.tp, item.match.fp, item.match.fn, item.match.miscls) == (
        item.tp,
        item.fp,
        item.fn,
        item.miscls,
    )
    assert item.match_iou_thr == pytest.approx(0.7)
    assert loaded.params.iou_tp == pytest.approx(0.7)
    assert item.match_use_mask_iou is False


def _polygon_line(class_id: int, points: list[tuple[float, float]]) -> str:
    """One YOLO-seg label line (normalized polygon) for a 160x160 plate."""
    coords = " ".join(f"{x / 160:.6f} {y / 160:.6f}" for x, y in points)
    return f"{class_id} {coords}\n"


def _solid_mask(box: tuple[float, float, float, float]) -> np.ndarray:
    """A full-frame boolean mask covering ``box``."""
    mask = np.zeros((160, 160), dtype=bool)
    x1, y1, x2, y2 = (int(value) for value in box)
    mask[y1:y2, x1:x2] = True
    return mask


def _masked_plate(tmp_path: Path) -> Path:
    """A plate whose GT carries a polygon mask, so mask IoU is available."""
    image = write_plate(tmp_path / "photos" / "plate.png")
    x1, y1, x2, y2 = _DISJOINT_BOXES["top"]
    (image.parent / "plate.txt").write_text(
        _polygon_line(0, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]),
        encoding="utf-8",
    )
    return image


def test_masked_run_records_the_mask_rule_and_a_box_run_records_the_box_rule(tmp_path):
    """The same plate, decided twice: the recorded rule says which one ran.

    The prediction's box is twice as tall as the particle but its mask covers
    exactly the GT mask, so mask matching finds the true positive (IoU 1.0)
    and box matching does not (IoU 0.125): the two rules disagree on purpose.
    """
    image = _masked_plate(tmp_path)
    mask = _solid_mask(_DISJOINT_BOXES["top"])
    masked = ScriptedDetector(
        [_detection(0, 0.90, (0.0, 0.0, 160.0, 80.0), mask=mask, mask_source="yolo-seg")]
    )
    result = run_detection(make_config([image], [masked], tmp_path / "run"))
    item = result.results[0][0]

    assert (item.tp, item.fp, item.fn) == (1, 0, 0)
    assert item.match_use_mask_iou is True
    assert item.match is not None and item.match.tp_pairs == [(0, 0)]

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["images"][0]["models"][0]["match_use_mask_iou"] is True

    box_run = run_detection(
        make_config(
            [_matching_plate(tmp_path / "plain")], [_matching_detector()], tmp_path / "box-run"
        )
    )
    box_payload = json.loads((box_run.run_dir / "run.json").read_text(encoding="utf-8"))
    assert box_payload["images"][0]["models"][0]["match_use_mask_iou"] is False


def test_a_masked_run_reopened_from_disk_keeps_its_counters_and_its_rule(tmp_path):
    """Masks are not persisted, so the reopen must not invent a second match.

    The record says the run matched by mask IoU and says what the counts were.
    Recomputing that identity from a folder without masks is impossible; the
    loader keeps the counters as the truth and reports no pair indices rather
    than a box-matched decision the run never made.
    """
    image = _masked_plate(tmp_path)
    masked = ScriptedDetector(
        [_detection(0, 0.90, (0.0, 0.0, 160.0, 80.0), mask=_solid_mask(_DISJOINT_BOXES["top"]))]
    )
    result = run_detection(make_config([image], [masked], tmp_path / "run"))

    loaded = load_run(result.run_dir)
    item = loaded.results[0][0]
    assert item.match_use_mask_iou is True
    assert item.match_iou_thr == pytest.approx(result.params.iou_tp)
    assert (item.tp, item.fp, item.fn, item.miscls) == (1, 0, 0, 0)
    assert item.match is not None
    assert (item.match.tp, item.match.fp, item.match.fn) == (1, 0, 0)
    # No box-matched identity is passed off as the run's own.
    assert item.match.tp_pairs == []
    assert item.match.fp_idx == []


def test_a_run_json_without_the_match_rule_still_loads(tmp_path):
    """Folders written before #120 keep loading, with the documented fallback.

    Those runs matched at ``params.iou_tp`` and never persisted masks, so box
    IoU at that threshold is the only rule their counters can have come from.
    With the boxes still in the folder the decision is reproducible.
    """
    image = _matching_plate(tmp_path)
    result = run_detection(make_config([image], [_matching_detector()], tmp_path / "run"))
    run_json = result.run_dir / "run.json"
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    for entry in payload["images"]:
        for row in entry["models"]:
            row.pop("match_iou_thr", None)
            row.pop("match_use_mask_iou", None)
    run_json.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_run(result.run_dir)
    item = loaded.results[0][0]

    assert (item.tp, item.fp, item.fn, item.miscls) == (1, 1, 1, 1)
    assert item.match_iou_thr == pytest.approx(loaded.params.iou_tp)
    assert item.match_use_mask_iou is False
    assert isinstance(item.match, MatchResult)
    assert item.match.tp_pairs == [(0, 0)]
    assert item.match.fp_idx == [1]
    assert item.match.miscls_pairs == [(2, 1)]
    assert item.match.fn_idx == [2]
