"""Contract: English shape and length-method values in run artifacts.

Ticket #187. The public enums and the bytes written to ``detections.csv`` and
``run.json`` must agree on the same English tokens, so a script that reads a
run needs no knowledge of the writer's locale.
"""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np

from polyx import Method, Morphotype
from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import (
    CalibrationPolicy,
    ModelSpec,
    RunConfig,
    RunParams,
    run_detection,
)
from polyx.core.measure import Method as MeasureMethod
from polyx.core.measure import Morphotype as MeasureMorphotype

ENGLISH_MORPHOTYPES = frozenset({"fibre", "fragment", "indeterminate"})
ENGLISH_METHODS = frozenset({"geodesic", "maximum_feret"})
SPANISH_TOKENS = frozenset({"fibra", "fragmento", "indeterminado", "geodesico", "Feret maximo"})

TOP_BOX = (20.0, 20.0, 60.0, 60.0)
BOTTOM_BOX = (90.0, 90.0, 130.0, 130.0)
SIDE = 160


def _write_plate(path):
    image = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in (TOP_BOX, BOTTOM_BOX):
        image[int(y1) + 5 : int(y2) - 5, int(x1) + 5 : int(x2) - 5] = 255
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())
    return path


def _box_mask(x1, y1, x2, y2):
    mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _polygon(class_id, points):
    values = [str(class_id)]
    for x, y in points:
        values += [f"{x / SIDE:.6f}", f"{y / SIDE:.6f}"]
    return " ".join(values) + "\n"


class _MaskedDetector:
    """A loaded detector with one compact and one thin, elongated particle."""

    def __init__(self, masks):
        self._masks = masks

    @property
    def names(self):
        return {0: "PET"}

    def predict_auto(self, image_path, **kwargs):
        return [
            Instance(
                class_id=0,
                class_name="PET",
                conf=0.9,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                mask=mask,
            )
            for (x1, y1, x2, y2), mask in zip((TOP_BOX, BOTTOM_BOX), self._masks, strict=True)
        ]


def test_public_enums_expose_the_english_values():
    assert [m.value for m in Morphotype] == ["fibre", "fragment", "indeterminate"]
    assert [m.value for m in Method] == ["geodesic", "maximum_feret"]
    # The root and the measure package expose the very same enums.
    assert Morphotype is MeasureMorphotype
    assert Method is MeasureMethod


def test_detection_table_and_run_json_carry_the_english_values(tmp_path):
    image = _write_plate(tmp_path / "photos" / "plate.png")
    # Ground truth with a compact and a thin particle: run.json stores measured
    # shape fields for GT records, while detections.csv stores them for
    # predictions.
    (image.parent / "plate.txt").write_text(
        _polygon(0, [(25, 25), (35, 25), (35, 35), (25, 35)])
        + _polygon(0, [(95, 100), (125, 100), (125, 105), (95, 105)]),
        encoding="utf-8",
    )
    detector = _MaskedDetector((_box_mask(25, 25, 35, 35), _box_mask(95, 100, 125, 105)))
    run_dir = tmp_path / "run"

    run_detection(
        RunConfig(
            images=[image],
            models=[ModelSpec(detector=detector, alias="m0")],
            output_dir=run_dir,
            params=RunParams(measure_shape=True),
            calibration=CalibrationPolicy(mode="manual", um_per_px=1.0),
        )
    )

    with (run_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_morph = {row["morphotype"] for row in rows}
    csv_methods = {row["method_length"] for row in rows}
    assert csv_morph and csv_morph <= ENGLISH_MORPHOTYPES
    assert csv_methods and csv_methods <= ENGLISH_METHODS
    assert not (csv_morph | csv_methods) & SPANISH_TOKENS
    assert "fragment" in csv_morph and "fibre" in csv_morph
    assert "maximum_feret" in csv_methods

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    gt = payload["images"][0]["gt"]
    json_morph = {entry["morphotype"] for entry in gt}
    json_methods = {entry["method_length"] for entry in gt}
    assert json_morph and json_morph <= ENGLISH_MORPHOTYPES
    assert json_methods and json_methods <= ENGLISH_METHODS
    assert not (json_morph | json_methods) & SPANISH_TOKENS
