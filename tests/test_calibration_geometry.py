"""Line and circle calibration geometry, without Qt.

The Viewer draws a two-point line or a three-point circle over a known real
size and turns it into µm/px. That maths lives in the calibration core so it
can be tested headless and next to the plate-ring, parallax and index-carry
concerns. A refused calibration names its reason instead of returning zero:
zero is a scale, and callers would then divide measurements by it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import pytest

from polyx.core.measure import calibration

# ── Line ────────────────────────────────────────────────────────────────


def test_line_scale_from_a_known_pixel_length_and_real_size():
    """200 px of drawn line stand for 6200 µm, the 31 µm/px per-photo case."""
    result = calibration.line_calibration((0.0, 0.0), (200.0, 0.0), 6200.0)

    assert result.failure == ""
    assert result.length_px == pytest.approx(200.0)
    assert result.um_per_px == pytest.approx(31.0)


def test_line_scale_does_not_depend_on_the_direction():
    """The 49 µm/px per-photo case, drawn vertically, returns the same scale."""
    result = calibration.line_calibration((10.0, 10.0), (10.0, 210.0), 9800.0)

    assert result.length_px == pytest.approx(200.0)
    assert result.um_per_px == pytest.approx(49.0)


def test_line_length_is_euclidean_not_axis_aligned():
    result = calibration.line_calibration((0.0, 0.0), (3.0, 4.0), 155.0)

    assert result.length_px == pytest.approx(5.0)
    assert result.um_per_px == pytest.approx(31.0)


# ── Circle ──────────────────────────────────────────────────────────────


def test_circle_centre_and_radius_from_three_points():
    """On a right triangle the circumcentre is the hypotenuse midpoint.

    The three points span a 1000 px diameter, so the 100 000 µm Petri dish
    default gives 100 µm/px.
    """
    result = calibration.circle_calibration((0.0, 0.0), (1000.0, 0.0), (500.0, 500.0), 100_000.0)

    assert result.failure == ""
    assert result.center == pytest.approx((500.0, 0.0))
    assert result.radius_px == pytest.approx(500.0)
    assert result.um_per_px == pytest.approx(100.0)


def test_circle_scale_works_away_from_the_origin():
    """The 100 000 µm dish drawn over a 250 px radius gives 200 µm/px."""
    result = calibration.circle_calibration(
        (900.0, 500.0), (650.0, 750.0), (400.0, 500.0), 100_000.0
    )

    assert result.center == pytest.approx((650.0, 500.0))
    assert result.radius_px == pytest.approx(250.0)
    assert result.um_per_px == pytest.approx(200.0)


# ── Refusals ────────────────────────────────────────────────────────────


def test_zero_length_line_is_refused_by_name():
    result = calibration.line_calibration((7.0, 7.0), (7.0, 7.0), 1000.0)

    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_ZERO_LENGTH


@pytest.mark.parametrize("real_um", [0.0, -1.0])
def test_non_positive_real_length_is_refused_by_name(real_um):
    result = calibration.line_calibration((0.0, 0.0), (100.0, 0.0), real_um)

    assert result.length_px == pytest.approx(100.0)
    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_NON_POSITIVE_REAL_SIZE


def test_collinear_points_are_refused_by_name():
    result = calibration.circle_calibration((0.0, 0.0), (10.0, 10.0), (20.0, 20.0), 100_000.0)

    assert result.center is None
    assert result.radius_px is None
    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_COLLINEAR_POINTS


def test_coincident_points_are_refused_by_name():
    result = calibration.circle_calibration((5.0, 5.0), (5.0, 5.0), (10.0, 10.0), 100_000.0)

    assert result.radius_px is None
    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_COINCIDENT_POINTS


@pytest.mark.parametrize("real_diameter_um", [0.0, -100.0])
def test_non_positive_real_diameter_is_refused_by_name(real_diameter_um):
    result = calibration.circle_calibration(
        (0.0, 0.0), (1000.0, 0.0), (500.0, 500.0), real_diameter_um
    )

    assert result.radius_px == pytest.approx(500.0), "the geometry itself is still usable"
    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_NON_POSITIVE_REAL_SIZE


@pytest.mark.parametrize(
    ("p1", "p2"),
    [((float("nan"), 0.0), (1.0, 1.0)), ((0.0, 0.0), (float("inf"), 1.0))],
)
def test_non_finite_line_coordinates_are_refused_by_name(p1, p2):
    result = calibration.line_calibration(p1, p2, 1000.0)

    assert result.length_px is None
    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_NON_FINITE


@pytest.mark.parametrize("real_um", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_real_length_is_refused_by_name(real_um):
    result = calibration.line_calibration((0.0, 0.0), (100.0, 0.0), real_um)

    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_NON_FINITE


@pytest.mark.parametrize(
    "points",
    [
        ((float("nan"), 0.0), (1000.0, 0.0), (500.0, 500.0)),
        ((0.0, 0.0), (1000.0, float("inf")), (500.0, 500.0)),
    ],
)
def test_non_finite_circle_coordinates_are_refused_by_name(points):
    result = calibration.circle_calibration(*points, 100_000.0)

    assert result.center is None
    assert result.radius_px is None
    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_NON_FINITE


@pytest.mark.parametrize("real_diameter_um", [float("nan"), float("inf")])
def test_non_finite_real_diameter_is_refused_by_name(real_diameter_um):
    result = calibration.circle_calibration(
        (0.0, 0.0), (1000.0, 0.0), (500.0, 500.0), real_diameter_um
    )

    assert result.um_per_px is None
    assert result.failure == calibration.FAILURE_NON_FINITE


# ── Point order ─────────────────────────────────────────────────────────


def test_line_scale_is_independent_of_endpoint_order():
    straight = calibration.line_calibration((10.0, 20.0), (310.0, 20.0), 9300.0)
    reversed_ = calibration.line_calibration((310.0, 20.0), (10.0, 20.0), 9300.0)

    assert straight.um_per_px == pytest.approx(31.0)
    assert reversed_.length_px == straight.length_px
    assert reversed_.um_per_px == straight.um_per_px


@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_circle_scale_is_independent_of_point_order(order):
    points = ((0.0, 0.0), (1000.0, 0.0), (500.0, 500.0))
    result = calibration.circle_calibration(*(points[i] for i in order), 100_000.0)

    assert result.center == pytest.approx((500.0, 0.0))
    assert result.radius_px == pytest.approx(500.0)
    assert result.um_per_px == pytest.approx(100.0)


# ── No Qt ───────────────────────────────────────────────────────────────


def test_geometry_imports_without_qt():
    """The geometry must be usable headless: no Qt module may get loaded."""
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys\n"
        "import polyx.core.measure.calibration as c\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.split('.')[0] in {'PySide6', 'PyQt5', 'PyQt6', 'shiboken6'}\n"
        ")\n"
        "print(','.join(loaded))\n"
        "print(c.line_calibration((0.0, 0.0), (10.0, 0.0), 100.0).um_per_px)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    qt_modules, um_per_px = result.stdout.splitlines()
    assert qt_modules == ""
    assert float(um_per_px) == pytest.approx(10.0)
