"""The Viewer's circle preview draws the core's circumscribed circle.

The preview needs the centre and the radius of the three clicked points and
nothing else: it is the same geometry the calibration applies. It used to call
``circle_calibration`` with a fabricated 1 µm diameter only to get that
circle back, which meant inventing a real size to draw a dashed ring.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QPixmap

from polyx.core.measure import calibration
from polyx.gui.viewer import canvas as canvas_module
from polyx.gui.viewer.state import ViewerState


def _paint_preview(state: ViewerState) -> None:
    """Runs the calibration overlay over an offscreen pixmap."""
    pixmap = QPixmap(200, 200)
    pixmap.fill()
    painter = QPainter(pixmap)
    try:
        canvas_module.ViewerCanvas(state)._draw_calib_overlay(painter)
    finally:
        painter.end()


def test_the_circle_preview_asks_the_core_for_the_circle(qapp, monkeypatch):
    """The centre and radius come from ``circumscribed_circle``, point for point."""
    state = ViewerState()
    state.start_calib("circulo")
    for point in ((10.0, 20.0), (110.0, 20.0), (60.0, 70.0)):
        state.add_calib_point(QPointF(*point))

    calls: list[tuple] = []

    def fake_circle(p1, p2, p3):
        calls.append((p1, p2, p3))
        return calibration.CircumscribedCircle(center=(60.0, 20.0), radius_px=50.0)

    monkeypatch.setattr(canvas_module, "circumscribed_circle", fake_circle)

    _paint_preview(state)

    assert calls == [((10.0, 20.0), (110.0, 20.0), (60.0, 70.0))]


def test_the_preview_draws_no_circle_when_the_points_define_none(qapp, monkeypatch):
    """Collinear points have no circumcircle: nothing is drawn, nothing raises."""
    state = ViewerState()
    state.start_calib("circulo")
    for point in ((0.0, 0.0), (10.0, 10.0), (20.0, 20.0)):
        state.add_calib_point(QPointF(*point))

    monkeypatch.setattr(canvas_module, "circumscribed_circle", lambda *_args: None)

    _paint_preview(state)


def test_the_preview_does_not_fabricate_a_real_size():
    """``circle_calibration`` takes a real diameter and the preview has none."""
    assert not hasattr(canvas_module, "circle_calibration")
