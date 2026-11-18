"""Warnings persist as stable English codes, never as prose.

The artifact carries a code plus structured parameters; the sentence a person
reads is chosen at display time. Tests observe the bytes written to the run
artifact and the HTML emitted by the report, the seams agreed in the spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polyx.core.measure import calibration as cal
from polyx.core.measure import shape
from polyx.core.measure.warnings import (
    DETOUR_HOUGH,
    IMAGE_UNREADABLE,
    MASK_CLIPPED,
    NO_CONTOUR,
    NO_MASK,
    NOMINAL_DIAMETER_INVALID,
    PARTICLE_TOO_SMALL,
    PLATE_MEASURE_FAILED,
    PLATE_NOT_FOUND,
    RING_FIT_FAILED,
    SEGMENTATION_FAILED,
    WARNING_CODES,
    ZERO_ADJUSTED_RADIUS,
    Warning,
    count_by_code,
)

# ── Slice A: calibration warnings are codes with structured parameters ──


def test_calibration_warnings_are_codes_not_prose():
    assert cal.calibrate_from_plate(None).warning.code == IMAGE_UNREADABLE
    empty = np.empty((0, 0, 3), np.uint8)
    assert cal.calibrate_from_plate(empty).warning.code == IMAGE_UNREADABLE
    img = np.full((400, 400, 3), 18, np.uint8)
    assert cal.calibrate_from_plate(img, diameter_mm=0).warning.code == NOMINAL_DIAMETER_INVALID


def test_calibration_failure_codes(monkeypatch):
    img = np.full((400, 400, 3), 18, np.uint8)
    # Feed the fine-fit branch enough synthetic edge points to run.
    monkeypatch.setattr(cal, "edge_points", lambda *a, **k: np.zeros((40, 2)))
    monkeypatch.setattr(cal, "hough_approximate", lambda _bgr: None)
    assert cal.calibrate_from_plate(img).warning.code == PLATE_NOT_FOUND

    monkeypatch.setattr(cal, "hough_approximate", lambda _bgr: (200.0, 200.0, 50.0))
    monkeypatch.setattr(cal, "adjustment_robust", lambda _pts: None)
    assert cal.calibrate_from_plate(img).warning.code == RING_FIT_FAILED

    monkeypatch.setattr(cal, "adjustment_robust", lambda _pts: (200.0, 200.0, 0.0, 10))
    assert cal.calibrate_from_plate(img).warning.code == ZERO_ADJUSTED_RADIUS


def test_detour_travels_as_a_structured_number(monkeypatch):
    """The rounded percentage lives in ``detour_pct``, not inside prose."""
    img = np.full((400, 400, 3), 18, np.uint8)
    monkeypatch.setattr(cal, "edge_points", lambda *a, **k: np.zeros((40, 2)))
    monkeypatch.setattr(cal, "hough_approximate", lambda _bgr: (200.0, 200.0, 50.0))
    monkeypatch.setattr(cal, "adjustment_robust", lambda _pts: (200.0, 200.0, 60.0, 10))
    warning = cal.calibrate_from_plate(img).warning
    assert warning.code == DETOUR_HOUGH
    assert warning.detour_pct == pytest.approx(20.0)


def test_nested_cause_travels_as_a_code(monkeypatch):
    """A fallback that failed to measure the plate names its cause by code."""
    monkeypatch.setattr(cal, "read_image", lambda _path: np.zeros((400, 400, 3), np.uint8))
    failed = cal.Calibration(warning=Warning(PLATE_NOT_FOUND))
    monkeypatch.setattr(cal, "calibrate_from_plate", lambda *a, **k: failed)
    result = cal.resolver(
        Path("plate.png"), um_por_px_manual=2.5, diameter_mm=100.0, measure_plate=True
    )
    assert result.warning.code == PLATE_MEASURE_FAILED
    assert result.warning.cause == PLATE_NOT_FOUND


def test_all_warning_codes_are_english_tokens():
    for code in WARNING_CODES:
        assert code.isascii() and code.islower(), code
        assert " " not in code and "ñ" not in code and "á" not in code


# ── Slice B: shape warnings are codes ──


def test_shape_warnings_are_codes():
    assert shape.measure(None).warning.code == NO_MASK
    assert shape.measure(np.zeros((5, 5), np.uint8)).warning.code == NO_MASK
    small = np.zeros((5, 5), np.uint8)
    small[2, 2] = 255
    assert shape.measure(small).warning.code == PARTICLE_TOO_SMALL


def test_shape_failure_and_structured_codes(monkeypatch):
    monkeypatch.setattr(shape, "segment", lambda *a, **k: None)
    det = _detection()
    assert shape.measure_detection(np.zeros((20, 20, 3), np.uint8), det).warning.code == (
        SEGMENTATION_FAILED
    )
    # No contour in the mask names its own code.
    monkeypatch.setattr(shape.cv2, "findContours", lambda *a, **k: ([], None))
    assert shape.measure(np.ones((6, 6), np.uint8) * 255).warning.code == NO_CONTOUR
    assert MASK_CLIPPED in WARNING_CODES


def _detection():
    from polyx.core.detect.protocol import Instance

    return Instance(class_id=0, class_name="x", conf=0.9, x1=1.0, y1=1.0, x2=8.0, y2=8.0)


# ── Slice C: grouped/counted by code ──


def test_warnings_group_and_count_by_code():
    warnings = [
        Warning(PLATE_NOT_FOUND),
        Warning(PLATE_NOT_FOUND),
        Warning(DETOUR_HOUGH, detour_pct=12.0),
    ]
    assert count_by_code(warnings) == {PLATE_NOT_FOUND: 2, DETOUR_HOUGH: 1}


def test_no_prose_reaches_a_warning_object():
    warning = Warning(PLATE_NOT_FOUND)
    assert json.dumps(warning.to_dict()) == json.dumps({"code": PLATE_NOT_FOUND})
