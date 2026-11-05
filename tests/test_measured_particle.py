"""Tests for the measured particle: provenance and the shared fallback.

The rule: a size is mask-measured or a declared box fallback, carried as
data in ``Detection.size_source`` — never silently recomputed from box
sides. ``measure_detections`` is the one loop: box size first as the
declared fallback, real shape over it when segmentation manages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_RAIZ = Path(__file__).resolve().parents[1]
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.protocol import compute_box_size_um  # noqa: E402
from polyx.core.detect.yolo import Detection  # noqa: E402
from polyx.core.measure.shape import apply_to_detection, measure_detections  # noqa: E402

UM = 2.0


def _placa_con_disco() -> tuple[np.ndarray, Detection]:
    """A 100x100 black plate with a bright disc: a particle Otsu can isolate."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 18, (255, 255, 255), -1)
    det = Detection(class_id=0, class_name="PET", conf=1.0, x1=28, y1=28, x2=72, y2=72)
    return img, det


def _placa_vacia() -> np.ndarray:
    """A uniform black plate: nothing for the segmentation to isolate."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_box_fallback_declares_itself():
    det = Detection(class_id=0, class_name="PET", conf=1.0, x1=10, y1=10, x2=30, y2=30)
    compute_box_size_um(det, UM)
    assert det.diam_um is not None
    assert det.size_source == "box"


def test_box_fallback_without_calibration_writes_nothing():
    det = Detection(class_id=0, class_name="PET", conf=1.0, x1=10, y1=10, x2=30, y2=30)
    compute_box_size_um(det, None)
    assert det.diam_um is None
    assert det.size_source is None


def test_mask_measuring_declares_itself():
    img, det = _placa_con_disco()
    assert apply_to_detection(det, img, UM) is True
    assert det.diam_um is not None
    assert det.size_source == "mask"


def test_failed_segmentation_keeps_the_box_fallback():
    det = Detection(class_id=0, class_name="PET", conf=1.0, x1=10, y1=10, x2=50, y2=50)
    compute_box_size_um(det, UM)
    assert apply_to_detection(det, _placa_vacia(), UM) is False
    # The box fallback stands, and says so.
    assert det.diam_um is not None
    assert det.size_source == "box"


def test_measure_detections_counts_failures_and_marks_sources():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 18, (255, 255, 255), -1)
    good = Detection(class_id=0, class_name="PET", conf=1.0, x1=28, y1=28, x2=72, y2=72)
    bad = Detection(class_id=1, class_name="PP", conf=1.0, x1=128, y1=28, x2=172, y2=72)

    failures = measure_detections([good, bad], img, UM)

    assert failures == 1
    assert good.size_source == "mask" and good.diam_um is not None
    assert bad.size_source == "box" and bad.diam_um is not None


def test_measure_detections_respects_the_shape_switch():
    img, det = _placa_con_disco()
    failures = measure_detections([det], img, UM, measure_shape=False)
    assert failures == 0
    # Only the box fallback ran: the shape fields stay empty.
    assert det.size_source == "box"
    assert det.morphotype is None


def test_measure_detections_without_calibration_invents_nothing():
    # The contraction pinned end to end: with no um/px, no size and no
    # origin is written, whatever the segmentation manages.
    img, det = _placa_con_disco()
    assert measure_detections([det], img, None) == 0
    assert det.diam_um is None and det.area_um2 is None
    assert det.size_source is None
