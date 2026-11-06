"""Detection-semantics parity between the canonical engine and the GUI runner.

Ticket #61: the engine decides tiling through ``RunParams`` and records it,
measures masks for predictions and ground truth, matches with mask IoU when a
mask is present, filters by size, numbers particles deterministically,
abstains below ``conf_assignment``, writes per-model ``classes.txt`` and reads
GT once with the first model's names while warning about class-map drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from polyx.cli import detect as cli_detect
from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import (
    CalibrationPolicy,
    ModelSpec,
    RunConfig,
    RunParams,
    load_run,
    run_detection,
)
from polyx.core.detection_run import engine as engine_module

TOP_BOX = (20.0, 20.0, 60.0, 60.0)
BOTTOM_BOX = (90.0, 90.0, 130.0, 130.0)


def write_plate(path: Path, width: int = 160, height: int = 160) -> Path:
    """Write a plate with two bright squares, so boxes have something to measure."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in (TOP_BOX, BOTTOM_BOX):
        image[int(y1) + 5 : int(y2) - 5, int(x1) + 5 : int(x2) - 5] = 255
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buffer))
    return path


class FakeDetector:
    """A loaded detection source with fixed boxes, names and an optional mask."""

    def __init__(
        self,
        boxes: tuple[tuple[float, float, float, float], ...] = (TOP_BOX, BOTTOM_BOX),
        names: dict[int, str] | None = None,
        masks: tuple[np.ndarray | None, ...] | None = None,
        confidences: tuple[float, ...] | None = None,
    ):
        self.boxes = boxes
        self._names = {0: "PET", 1: "PP"} if names is None else names
        self.masks = masks
        self.confidences = confidences
        self.calls: list[tuple[Path, dict]] = []
        self.last_imgsz = 0
        self.last_fallback = False

    @property
    def names(self) -> dict[int, str]:
        return self._names

    def load(self):
        return self

    def predict_auto(self, image_path, **kwargs) -> list[Instance]:
        self.calls.append((Path(image_path), dict(kwargs)))
        self.last_imgsz = int(kwargs.get("imgsz", 0) or 0)
        detections: list[Instance] = []
        for index, (x1, y1, x2, y2) in enumerate(self.boxes):
            confidence = (
                self.confidences[index] if self.confidences is not None else 0.9 - 0.1 * index
            )
            detections.append(
                Instance(
                    class_id=index % 2,
                    class_name=self._names.get(index % 2, str(index % 2)),
                    conf=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    mask=None if self.masks is None else self.masks[index],
                )
            )
        return detections


def make_config(
    images: list[Path],
    models: list[FakeDetector],
    output_dir: Path,
    *,
    params: RunParams | None = None,
    gt_folder: Path | None = None,
    calibration: CalibrationPolicy | None = None,
) -> RunConfig:
    return RunConfig(
        images=images,
        models=[
            ModelSpec(detector=model, alias=f"model_{chr(97 + i)}")
            for i, model in enumerate(models)
        ],
        output_dir=output_dir,
        params=params or RunParams(measure_shape=False),
        gt_folder=gt_folder,
        calibration=calibration or CalibrationPolicy(mode="manual", um_per_px=1.0),
    )


class TilingDetector(FakeDetector):
    """Fills the shared tiling record the way ``YoloModel.predict_auto`` does."""

    def __init__(self, tiled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tiled = tiled

    def predict_auto(self, image_path, **kwargs):
        record = kwargs.get("record")
        if record is not None:
            record.clear()
            record.update(
                {
                    "ancho": 160,
                    "alto": 160,
                    "troceado": self.tiled,
                    "plan": (
                        {
                            "tile": 64,
                            "overlap": float(kwargs.get("overlap", 0.25)),
                            "n_tiles": 4,
                            "lado": 160,
                            "umbral_px": 0,
                        }
                        if self.tiled
                        else None
                    ),
                }
            )
        return super().predict_auto(image_path, **kwargs)


def test_tiling_policy_from_run_params_is_passed_and_recorded(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    detector = TilingDetector(tiled=True)
    params = RunParams(
        measure_shape=False,
        tiling="siempre",
        tiling_threshold_px=1234,
        tiling_tile=64,
        tiling_overlap=0.5,
    )

    result = run_detection(make_config([image], [detector], tmp_path / "run", params=params))

    _path, call = detector.calls[0]
    assert call["tiling"] == "siempre"
    assert call["threshold_px"] == 1234
    assert call["tile"] == 64
    assert call["overlap"] == 0.5
    assert isinstance(call["record"], dict)

    image_result = result.results[0][0]
    assert "4 tiles of 64 px" in image_result.tiling
    assert "50% overlap" in image_result.tiling
    assert result.execution[0].tiled_images == 1
    assert result.execution[0].imgsz_used == [64]

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["images"][0]["models"][0]["tiling"] == image_result.tiling
    assert load_run(result.run_dir).results[0][0].tiling == image_result.tiling


def test_untiled_photo_records_no_tiling_description(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    detector = TilingDetector(tiled=False, boxes=(TOP_BOX,))

    result = run_detection(make_config([image], [detector], tmp_path / "run"))

    _path, call = detector.calls[0]
    assert call["tiling"] == "auto"
    assert call["threshold_px"] == 2000
    assert result.results[0][0].tiling == ""
    assert result.execution[0].tiled_images == 0
    assert result.execution[0].imgsz_used == [result.params.imgsz]


# ── Masks ───────────────────────────────────────────────────────────────


def mask_square(x1: int, y1: int, x2: int, y2: int, width: int = 160, height: int = 160):
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def polygon_line(
    class_id: int, points: list[tuple[float, float]], width: int = 160, height: int = 160
):
    values = [f"{class_id}"]
    for x, y in points:
        values.extend([f"{x / width:.6f}", f"{y / height:.6f}"])
    return " ".join(values) + "\n"


def test_prediction_masks_are_used_for_measurement(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    small_mask = mask_square(25, 25, 35, 35)
    detector = FakeDetector(boxes=(TOP_BOX,), masks=(small_mask,))
    params = RunParams(measure_shape=True)

    result = run_detection(
        make_config(
            [image],
            [detector],
            tmp_path / "run",
            params=params,
            calibration=CalibrationPolicy(mode="manual", um_per_px=1.0),
        )
    )

    detection = result.results[0][0].predictions[0]
    assert detection.size_source == "mask"
    # A 10x10 px mask, not the 40x40 box nor the 30x30 bright square.
    assert detection.area_um2 == pytest.approx(100.0)
    assert detection.diam_um == pytest.approx(2.0 * np.sqrt(100.0 / np.pi))


def test_ground_truth_masks_are_used_for_measurement(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    (image.parent / "plate.txt").write_text(
        polygon_line(0, [(25, 25), (35, 25), (35, 35), (25, 35)]),
        encoding="utf-8",
    )
    detector = FakeDetector(boxes=(TOP_BOX,))
    params = RunParams(measure_shape=True)

    result = run_detection(
        make_config(
            [image],
            [detector],
            tmp_path / "run",
            params=params,
            calibration=CalibrationPolicy(mode="manual", um_per_px=1.0),
        )
    )

    gt = result.results[0][0].gt[0]
    assert gt.size_source == "mask"
    # The polygon mask, not the 40x40 GT box nor the 30x30 bright square.
    assert gt.area_um2 is not None and gt.area_um2 < 200.0
    assert gt.diam_um is not None and gt.diam_um < 20.0


def test_mask_iou_drives_matching_when_a_mask_is_present(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    (image.parent / "plate.txt").write_text(
        polygon_line(0, [(20, 20), (60, 20), (60, 60), (20, 60)]),
        encoding="utf-8",
    )
    # The box is identical to the GT box, but the prediction mask covers only
    # its left quarter: mask IoU is 0.25, so it must not be a true positive.
    quarter_mask = mask_square(20, 20, 30, 60)
    masked = FakeDetector(boxes=(TOP_BOX,), masks=(quarter_mask,))
    unmasked = FakeDetector(boxes=(TOP_BOX,))
    params = RunParams(measure_shape=False)

    without_mask = run_detection(
        make_config([image], [unmasked], tmp_path / "run-plain", params=params)
    )
    with_mask = run_detection(make_config([image], [masked], tmp_path / "run-mask", params=params))

    plain = without_mask.results[0][0]
    assert (plain.tp, plain.fp, plain.fn) == (1, 0, 0)

    masked_result = with_mask.results[0][0]
    assert (masked_result.tp, masked_result.fp, masked_result.fn) == (0, 1, 1)


# ── Size filter, numbering and abstention ───────────────────────────────


def test_size_filter_drops_predictions_outside_the_micrometre_range(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    detector = FakeDetector(
        boxes=(
            (20.0, 20.0, 30.0, 30.0),  # 10x10 px -> ~11.3 um diameter
            (60.0, 60.0, 80.0, 80.0),  # 20x20 px -> ~22.6 um
            (100.0, 100.0, 140.0, 140.0),  # 40x40 px -> ~45.1 um
        )
    )
    params = RunParams(measure_shape=False, size_min_um=15.0, size_max_um=30.0)

    result = run_detection(
        make_config(
            [image],
            [detector],
            tmp_path / "run",
            params=params,
            calibration=CalibrationPolicy(mode="manual", um_per_px=1.0),
        )
    )

    predictions = result.results[0][0].predictions
    assert len(predictions) == 1
    assert predictions[0].x1 == 60.0
    assert predictions[0].number == 1
    assert result.execution[0].detections == 1


def test_particles_are_numbered_top_to_bottom_then_left_to_right(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    detector = FakeDetector(boxes=(BOTTOM_BOX, TOP_BOX))

    result = run_detection(
        make_config(
            [image],
            [detector],
            tmp_path / "run",
            calibration=CalibrationPolicy(mode="manual", um_per_px=1.0),
        )
    )

    predictions = result.results[0][0].predictions
    numbers = {round(det.y1): det.number for det in predictions}
    assert numbers == {20: 1, 90: 2}
    # Numbering does not reorder the model's own output.
    assert [det.y1 for det in predictions] == [90.0, 20.0]


def test_abstention_marks_low_confidence_particles_without_dropping_them(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    detector = FakeDetector(boxes=(TOP_BOX, BOTTOM_BOX), confidences=(0.9, 0.7))
    params = RunParams(measure_shape=False, conf_assignment=0.8)

    result = run_detection(make_config([image], [detector], tmp_path / "run", params=params))

    predictions = result.results[0][0].predictions
    assert len(predictions) == 2
    assert predictions[0].assignable is True
    assert predictions[1].assignable is False
    assert [det.number for det in predictions] == [1, 2]


# ── Classes and GT naming ───────────────────────────────────────────────


class RecordingLog:
    """Minimal logger that keeps the warnings the engine emits."""

    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message % args if args else str(message))

    def exception(self, message, *args, **kwargs):
        self.warnings.append(message % args if args else str(message))


def test_each_model_folder_gets_its_classes_txt(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    model_a = FakeDetector(names={0: "PET", 1: "PP"})
    model_b = FakeDetector(names={0: "PET", 1: "LDPE"})

    run_detection(make_config([image], [model_a, model_b], tmp_path / "run"))

    assert (tmp_path / "run" / "model_a" / "classes.txt").read_text(encoding="utf-8") == "PET\nPP\n"
    assert (tmp_path / "run" / "model_b" / "classes.txt").read_text(
        encoding="utf-8"
    ) == "PET\nLDPE\n"


def test_gt_uses_the_first_model_names_and_warns_on_disagreement(tmp_path, monkeypatch):
    image = write_plate(tmp_path / "photos" / "plate.png")
    width = height = 160
    x1, y1, x2, y2 = TOP_BOX
    (image.parent / "plate.txt").write_text(
        f"1 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n",
        encoding="utf-8",
    )
    model_a = FakeDetector(boxes=(TOP_BOX,), names={0: "PET", 1: "PP"})
    model_b = FakeDetector(boxes=(TOP_BOX,), names={0: "PET", 1: "LDPE"})
    log = RecordingLog()
    monkeypatch.setattr(engine_module, "logger_for", lambda _name: log)

    result = run_detection(make_config([image], [model_a, model_b], tmp_path / "run"))

    first, second = result.results[0][0], result.results[1][0]
    assert first.gt[0].class_name == "PP"
    assert second.gt[0].class_name == "PP"

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["models"][0]["names"] == {"0": "PET", "1": "PP"}
    assert payload["models"][1]["names"] == {"0": "PET", "1": "LDPE"}
    assert any(
        "GT class maps disagree" in warning and "model_b" in warning for warning in log.warnings
    )


# ── CLI surface ─────────────────────────────────────────────────────────


def test_cli_exposes_tiling_and_detection_semantics_flags(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    detector = FakeDetector(boxes=(TOP_BOX,), names={0: "PET"})
    monkeypatch.setattr(cli_detect, "make_detector", lambda *_args, **_kwargs: detector)
    images = tmp_path / "photos"
    write_plate(images / "plate.png")
    output = tmp_path / "run"

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(images),
            "--out",
            str(output),
            "--tiling",
            "siempre",
            "--tiling-threshold-px",
            "1234",
            "--tiling-tile",
            "64",
            "--tiling-overlap",
            "0.5",
            "--size-min-um",
            "10",
            "--size-max-um",
            "500",
            "--conf-assignment",
            "0.8",
        ]
    )

    assert code == 0
    run = load_run(output)
    assert run.params.tiling == "siempre"
    assert run.params.tiling_threshold_px == 1234
    assert run.params.tiling_tile == 64
    assert run.params.tiling_overlap == pytest.approx(0.5)
    assert run.params.size_min_um == pytest.approx(10.0)
    assert run.params.size_max_um == pytest.approx(500.0)
    assert run.params.conf_assignment == pytest.approx(0.8)

    _path, call = detector.calls[0]
    assert call["tiling"] == "siempre"
    assert call["threshold_px"] == 1234
    assert call["tile"] == 64
    assert call["overlap"] == pytest.approx(0.5)
