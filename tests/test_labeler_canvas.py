"""The Labeller canvas is a thin shell over the shared editing core (#145).

Every case drives the widget with real ``QTest`` mouse and key events and
asserts what a labeller can observe: the boxes the state reports, the
selection the core holds, the signal the window turns into a status message,
and the bytes on disk. Geometry, hit-testing, handles and the undo stacks
belong to :mod:`polyx.core.editing`; nothing here reaches for them in the
canvas.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from polyx.core.editing import MIN_SIDE_PX
from polyx.core.labels import label_path_for
from polyx.gui.labeler.canvas import BboxCanvas
from polyx.gui.labeler.state import LabelerState

# The bytes the pre-migration canvas wrote for ``test_the_written_yolo_lines...``
# (captured from 80b19a7 with this same event script). A migration that
# shifts a coordinate convention changes this literal.
CHARACTERIZED_YOLO = (
    b"0 0.249474 0.324737 0.332632 0.401053\n"
    b"1 0.668421 0.315263 0.336842 0.356842\n"
    b"2 0.300000 0.776316 0.391579 0.325263\n"
)


def _png(path: Path, width: int = 60, height: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(bytes(buf))
    return path


@pytest.fixture
def labeller(qapp, tmp_path):
    """A canvas over a two-image folder, shown offscreen at a known size."""
    image_dir = tmp_path / "imgs"
    _png(image_dir / "a.png")
    _png(image_dir / "b.png")
    state = LabelerState()
    canvas = BboxCanvas(state)
    canvas.resize(500, 500)
    canvas.show()
    state.load_images(image_dir)
    qapp.processEvents()
    yield state, canvas
    canvas.deleteLater()


def _widget(canvas: BboxCanvas, x: float, y: float) -> QPoint:
    """Image pixel → the widget point a user's click lands on."""
    pt = canvas._i2w(QPointF(x, y))
    return QPoint(round(pt.x()), round(pt.y()))


def _drag(canvas: BboxCanvas, qapp, start, end, button=Qt.LeftButton) -> None:
    QTest.mousePress(canvas, button, Qt.NoModifier, _widget(canvas, *start))
    QTest.mouseMove(canvas, _widget(canvas, *end))
    QTest.mouseRelease(canvas, button, Qt.NoModifier, _widget(canvas, *end))
    qapp.processEvents()


def test_a_draw_produces_one_box_and_selects_it(labeller, qapp):
    state, canvas = labeller

    _drag(canvas, qapp, (5, 5), (25, 21))

    boxes = state.current_boxes
    assert len(boxes) == 1
    assert boxes[0].class_id == state.active_class
    assert state.current_editor.selection == 0
    # The image is 60×40: image pixel (15, 13) is the box centre.
    assert boxes[0].cx == pytest.approx(15 / 60, abs=1e-3)
    assert boxes[0].cy == pytest.approx(13 / 40, abs=1e-3)


def test_the_written_yolo_lines_are_byte_identical_to_the_pre_migration_output(labeller, qapp):
    """Characterization: the same gestures must write the same bytes.

    The expected literal was captured from the pre-migration canvas with
    this exact script (three boxes, one per class, disjoint areas).
    """
    state, canvas = labeller

    _drag(canvas, qapp, (5, 5), (25, 21))
    state.set_active_class(1)
    _drag(canvas, qapp, (30, 5.5), (50.25, 19.75))
    state.set_active_class(2)
    _drag(canvas, qapp, (6.25, 24.5), (29.75, 37.5))

    state.save_current()

    assert label_path_for(state.current_image).read_bytes() == CHARACTERIZED_YOLO


def test_a_box_under_two_px_is_refused_visibly_and_nothing_is_written(labeller, qapp):
    state, canvas = labeller
    rejected = []
    canvas.box_rejected.connect(rejected.append)

    _drag(canvas, qapp, (5, 5), (6, 30))  # 1 px wide
    state.save_current()

    assert state.current_boxes == []
    assert len(rejected) == 1
    assert rejected[0] < MIN_SIDE_PX
    # Nothing written: the census rule still owns the empty file.
    assert not label_path_for(state.current_image).exists()


def test_the_rejection_message_is_shown_by_the_window(tmp_path, qapp):
    from polyx.core.i18n import _
    from polyx.gui.labeler.window import LabelerWindow

    image_dir = tmp_path / "imgs"
    image = _png(image_dir / "a.png")

    window = LabelerWindow()
    rejected = []
    try:
        window.show()
        window.state.load_images(image_dir)
        window.canvas.box_rejected.connect(rejected.append)
        qapp.processEvents()
        canvas = window.canvas
        # Read the status straight after the release: a language change can
        # rebuild the window (and its labels) on the next event flush.
        QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, _widget(canvas, 5, 5))
        QTest.mouseMove(canvas, _widget(canvas, 6, 30))
        QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, _widget(canvas, 6, 30))
        status = window.lbl_status.text()
    finally:
        window.deleteLater()

    assert window.state.current_boxes == []
    assert len(rejected) == 1
    expected = _("⚠ Box of %(side).1f px discarded (minimum %(minimum).0f px) — zoom in") % {
        "side": rejected[0],
        "minimum": MIN_SIDE_PX,
    }
    assert status == expected
    assert not label_path_for(image).exists()


def test_a_move_through_a_real_drag_translates_the_box(labeller, qapp):
    state, canvas = labeller
    _drag(canvas, qapp, (5, 5), (25, 21))
    before = state.current_editor.boxes[0]

    _drag(canvas, qapp, (15, 13), (20, 13))  # grab the centre, drag 5 px right

    moved = state.current_editor.boxes[0]
    assert moved.x1 == pytest.approx(before.x1 + 5, abs=0.2)
    assert moved.y1 == pytest.approx(before.y1, abs=0.2)
    assert moved.width == pytest.approx(before.width, abs=0.2)


def test_a_resize_from_the_bottom_right_handle_grows_the_box(labeller, qapp):
    state, canvas = labeller
    _drag(canvas, qapp, (5, 5), (25, 21))
    before = state.current_editor.boxes[0]

    _drag(canvas, qapp, (25, 21), (30, 26))  # bottom-right corner outward

    resized = state.current_editor.boxes[0]
    assert resized.x1 == pytest.approx(before.x1, abs=0.2)
    assert resized.y1 == pytest.approx(before.y1, abs=0.2)
    assert resized.width == pytest.approx(before.width + 5, abs=0.3)
    assert resized.height == pytest.approx(before.height + 5, abs=0.3)


def test_ctrl_z_and_ctrl_y_undo_and_redo_a_draw(labeller, qapp):
    state, canvas = labeller
    _drag(canvas, qapp, (5, 5), (25, 21))
    assert len(state.current_boxes) == 1

    QTest.keyClick(canvas, Qt.Key_Z, Qt.ControlModifier)
    qapp.processEvents()
    assert state.current_boxes == []

    QTest.keyClick(canvas, Qt.Key_Y, Qt.ControlModifier)
    qapp.processEvents()
    assert len(state.current_boxes) == 1


def test_per_image_history_survives_navigating_away_and_back(labeller, qapp):
    state, canvas = labeller
    first = state.current_image
    _drag(canvas, qapp, (5, 5), (25, 21))

    state.next_image()
    qapp.processEvents()
    assert state.current_image != first
    assert state.current_boxes == []

    state.prev_image()
    qapp.processEvents()
    assert state.current_image == first
    assert len(state.current_boxes) == 1

    QTest.keyClick(canvas, Qt.Key_Z, Qt.ControlModifier)
    qapp.processEvents()
    assert state.current_boxes == []


def test_delete_and_undo_remove_and_restore_the_selection(labeller, qapp):
    state, canvas = labeller
    _drag(canvas, qapp, (5, 5), (25, 21))

    QTest.keyClick(canvas, Qt.Key_Delete)
    qapp.processEvents()
    assert state.current_boxes == []
    assert state.current_editor.selection == -1

    QTest.keyClick(canvas, Qt.Key_Z, Qt.ControlModifier)
    qapp.processEvents()
    assert len(state.current_boxes) == 1
    assert state.current_editor.selection == 0


def test_keys_1_to_9_change_the_class_of_the_next_draw(labeller, qapp):
    state, canvas = labeller

    QTest.keyClick(canvas, Qt.Key_3)
    qapp.processEvents()
    assert state.active_class == 2
    assert state.current_editor.active_class == 2

    _drag(canvas, qapp, (5, 5), (25, 21))
    assert state.current_boxes[0].class_id == 2


def test_a_click_on_empty_space_clears_the_selection(labeller, qapp):
    state, canvas = labeller
    _drag(canvas, qapp, (5, 5), (25, 21))
    assert state.current_editor.selection == 0

    QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, _widget(canvas, 45, 35))
    QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, _widget(canvas, 45, 35))
    qapp.processEvents()

    assert state.current_editor.selection == -1
    assert len(state.current_boxes) == 1  # the zero-size click drew nothing


def test_middle_button_drag_pans_the_view(labeller, qapp):
    _state, canvas = labeller
    before = QPointF(canvas._offset)

    QTest.mousePress(canvas, Qt.MiddleButton, Qt.NoModifier, QPoint(100, 100))
    QTest.mouseMove(canvas, QPoint(130, 115))
    QTest.mouseRelease(canvas, Qt.MiddleButton, Qt.NoModifier, QPoint(130, 115))
    qapp.processEvents()

    assert canvas._offset.x() == pytest.approx(before.x() + 30, abs=1)
    assert canvas._offset.y() == pytest.approx(before.y() + 15, abs=1)


def test_the_wheel_zooms_around_the_cursor(labeller, qapp):
    _state, canvas = labeller
    before = canvas._scale

    event = QWheelEvent(
        QPointF(250, 250),
        QPointF(250, 250),
        QPoint(),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(canvas, event)

    assert canvas._scale == pytest.approx(before * 1.15)
