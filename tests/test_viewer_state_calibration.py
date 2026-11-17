"""Ticket #149: the Viewer state collects the points; the core does the maths.

``ViewerState`` keeps what belongs to the interface -- the clicked ``QPointF``,
the active mode and the signals pages connect to -- and hands the geometry to
``polyx.core.measure.calibration``. A successful finish sets the scale and
emits once; a refused one leaves the previous scale untouched, emits nothing,
and comes back with the named failure so the dialog can say why.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF

from polyx.core.measure import calibration
from polyx.gui.viewer import state as state_module
from polyx.gui.viewer.state import ViewerState


class _RecordingSignals:
    """Everything the state emits, in order, for the three calib signals."""

    def __init__(self, state: ViewerState) -> None:
        self.changed: list[tuple[float, str]] = []
        self.points: list[list[QPointF]] = []
        self.ready: int = 0
        state.calib_changed.connect(lambda um, mode: self.changed.append((um, mode)))
        state.calib_pts_changed.connect(lambda pts: self.points.append(list(pts)))
        state.calib_ready.connect(self._on_ready)

    def _on_ready(self) -> None:
        self.ready += 1


def _collect(mode: str, *points: tuple[float, float]) -> ViewerState:
    state = ViewerState()
    state.start_calib(mode)
    for x, y in points:
        state.add_calib_point(QPointF(x, y))
    return state


# ── Successful finishes ─────────────────────────────────────────────────


def test_a_line_finish_keeps_the_31_um_per_px_scale_and_emits_once(qapp):
    state = _collect("line", (0.0, 0.0), (200.0, 0.0))
    signals = _RecordingSignals(state)

    failure = state.finish_calib_line(6200.0)

    assert failure is None, "no failure is the success signal"
    assert state.um_per_px == pytest.approx(31.0)
    assert len(signals.changed) == 1
    assert signals.changed[0][0] == pytest.approx(31.0)
    assert signals.changed[0][1] == "line", "the signal carries the English mode token"
    assert signals.points == [[]], "the collected points are cleared once"
    assert state.calib_pts == []
    assert state.calib_mode == "none"


def test_a_line_finish_keeps_the_49_um_per_px_scale(qapp):
    state = _collect("line", (10.0, 10.0), (10.0, 210.0))
    signals = _RecordingSignals(state)

    assert state.finish_calib_line(9800.0) is None
    assert state.um_per_px == pytest.approx(49.0)
    assert len(signals.changed) == 1


def test_a_circle_finish_keeps_the_100000_um_dish_scale(qapp):
    state = _collect("circle", (0.0, 0.0), (1000.0, 0.0), (500.0, 500.0))
    signals = _RecordingSignals(state)

    assert state.finish_calib_circle(100_000.0) is None
    assert state.um_per_px == pytest.approx(100.0)
    assert len(signals.changed) == 1
    assert signals.changed[0][1] == "circle"


# ── The geometry helpers are gone ───────────────────────────────────────


def test_the_calibration_modes_are_english_machine_tokens(qapp):
    """The mode is a token, not copy: the words the user reads come from ``_()``.

    It reaches ``calib_changed`` and the canvas, where a Spanish token would be
    a second vocabulary inside Python that no catalog can translate.
    """
    state = ViewerState()
    assert state.calib_mode == "none"

    state.start_calib("line")
    assert state.calib_mode == "line"

    state.start_calib("circle")
    assert state.calib_mode == "circle"


def test_the_state_no_longer_carries_the_geometry_helpers():
    """The circumscribed circle lives in the calibration core, not the GUI."""
    assert not hasattr(state_module, "_circumscribed_radius")
    assert not hasattr(state_module, "_circumscribed_center")


def test_the_state_does_no_line_arithmetic_of_its_own(qapp, monkeypatch):
    """A scale the core would never produce is what the state applies.

    The finish path takes the core's answer, not a number recomputed beside
    the signals: the core is replaced with a stub that returns a value the
    two points cannot give, and that value is what reaches ``um_per_px``.
    """
    calls: list[tuple] = []

    def fake_line(p1, p2, real_um):
        calls.append((p1, p2, real_um))
        return calibration.LineCalibration(length_px=1.0, um_per_px=12.5)

    monkeypatch.setattr(state_module, "line_calibration", fake_line)
    state = _collect("line", (0.0, 0.0), (200.0, 0.0))

    state.finish_calib_line(6200.0)

    assert state.um_per_px == pytest.approx(12.5)
    assert calls == [((0.0, 0.0), (200.0, 0.0), 6200.0)]


# ── Refusals ────────────────────────────────────────────────────────────

_REFUSALS = [
    (
        "line",
        ((0.0, 0.0), (0.0, 0.0)),
        1000.0,
        calibration.FAILURE_ZERO_LENGTH,
    ),
    (
        "line",
        ((0.0, 0.0), (100.0, 0.0)),
        0.0,
        calibration.FAILURE_NON_POSITIVE_REAL_SIZE,
    ),
    (
        "line",
        ((math.nan, 0.0), (1.0, 1.0)),
        1000.0,
        calibration.FAILURE_NON_FINITE,
    ),
    (
        "circle",
        ((0.0, 0.0), (10.0, 10.0), (20.0, 20.0)),
        100_000.0,
        calibration.FAILURE_COLLINEAR_POINTS,
    ),
    (
        "circle",
        ((5.0, 5.0), (5.0, 5.0), (10.0, 10.0)),
        100_000.0,
        calibration.FAILURE_COINCIDENT_POINTS,
    ),
    (
        "circle",
        ((0.0, 0.0), (1000.0, 0.0), (500.0, 500.0)),
        0.0,
        calibration.FAILURE_NON_POSITIVE_REAL_SIZE,
    ),
    (
        "circle",
        ((math.nan, 0.0), (1000.0, 0.0), (500.0, 500.0)),
        100_000.0,
        calibration.FAILURE_NON_FINITE,
    ),
]


@pytest.mark.parametrize(("mode", "points", "real", "expected"), _REFUSALS)
def test_a_degenerate_drawing_reports_the_named_failure(mode, points, real, expected, qapp):
    state = _collect(mode, *points)
    state.um_per_px = 2.5
    signals = _RecordingSignals(state)

    finish = state.finish_calib_line if mode == "line" else state.finish_calib_circle
    failure = finish(real)

    assert failure == expected
    assert state.um_per_px == pytest.approx(2.5), "the previous scale survives"
    assert signals.changed == [], "a refused finish changes no scale"
    assert signals.points == [], "a refused finish redraws nothing"
    assert signals.ready == 0


def test_the_scale_of_an_earlier_calibration_survives_a_refused_circle(qapp):
    """31 µm/px stands when a later collinear drawing is rejected."""
    state = _collect("line", (0.0, 0.0), (200.0, 0.0))
    assert state.finish_calib_line(6200.0) is None
    state.start_calib("circle")
    for point in ((0.0, 0.0), (10.0, 10.0), (20.0, 20.0)):
        state.add_calib_point(QPointF(*point))
    signals = _RecordingSignals(state)

    failure = state.finish_calib_circle(100_000.0)

    assert failure == calibration.FAILURE_COLLINEAR_POINTS
    assert state.um_per_px == pytest.approx(31.0)
    assert signals.changed == []


def test_a_finish_before_the_drawing_is_complete_is_refused_by_name(qapp):
    """No silent return: a caller hears why nothing happened."""
    line = _collect("line", (0.0, 0.0))
    circle = _collect("circle", (0.0, 0.0), (10.0, 0.0))
    line_signals = _RecordingSignals(line)
    circle_signals = _RecordingSignals(circle)

    assert line.finish_calib_line(1000.0) == state_module.FAILURE_NOT_ENOUGH_POINTS
    assert circle.finish_calib_circle(100_000.0) == state_module.FAILURE_NOT_ENOUGH_POINTS
    assert line.um_per_px == 0.0
    assert circle.um_per_px == 0.0
    assert line_signals.changed == []
    assert circle_signals.changed == []


# ── Point collection, preview and readiness ─────────────────────────────


def test_each_click_redraws_the_points_it_has_collected(qapp):
    state = ViewerState()
    state.start_calib("line")
    signals = _RecordingSignals(state)

    state.add_calib_point(QPointF(10.0, 20.0))
    state.add_calib_point(QPointF(30.0, 40.0))

    assert [[(p.x(), p.y()) for p in pts] for pts in signals.points] == [
        [(10.0, 20.0)],
        [(10.0, 20.0), (30.0, 40.0)],
    ]


def test_a_line_becomes_ready_at_two_points(qapp):
    state = _collect("line", (0.0, 0.0))
    signals = _RecordingSignals(state)

    state.add_calib_point(QPointF(10.0, 0.0))

    assert signals.ready == 1


def test_a_circle_becomes_ready_at_three_points_not_two(qapp):
    state = ViewerState()
    state.start_calib("circle")
    state.add_calib_point(QPointF(0.0, 0.0))
    signals = _RecordingSignals(state)

    state.add_calib_point(QPointF(10.0, 0.0))
    assert signals.ready == 0, "two points are not a circle"

    state.add_calib_point(QPointF(0.0, 10.0))
    assert signals.ready == 1


# ── The dialog says why ─────────────────────────────────────────────────

_REFUSAL_MESSAGES = [
    ("line", ((5.0, 5.0), (5.0, 5.0)), "same"),
    ("circle", ((0.0, 0.0), (10.0, 10.0), (20.0, 20.0)), "straight line"),
]


@pytest.mark.parametrize(("mode", "points", "phrase"), _REFUSAL_MESSAGES)
def test_a_refused_finish_tells_the_user_why_and_keeps_the_previous_scale(
    mode, points, phrase, qapp, monkeypatch, selected_language
):
    """The dialog is where the named failure becomes words."""
    from PySide6.QtWidgets import QMessageBox

    from polyx.gui.viewer import window as window_module

    selected_language("en")
    window = window_module.ViewerWindow()
    warnings: list[str] = []
    monkeypatch.setattr(window_module.QInputDialog, "getDouble", lambda *a, **k: (1000.0, True))
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    window.state.um_per_px = 2.5
    window.state.start_calib(mode)
    try:
        # The clicks reach _on_calib_ready through the real signal.
        for point in points:
            window.state.add_calib_point(QPointF(*point))
    finally:
        window.close()

    assert window.state.um_per_px == pytest.approx(2.5)
    assert len(warnings) == 1, "the failure is surfaced, not swallowed"
    assert phrase in warnings[0], "the message names the reason"
    assert "not applied" in window.lbl_status.text()
    assert window.state.calib_mode == "none", "the drawing restarts cleanly"
