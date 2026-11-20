"""The Labeller canvas is a thin shell over the shared editing core (#145).

Every case drives the widget with real ``QTest`` mouse and key events and
asserts what a labeller can observe: the boxes the state reports, the
selection the core holds, the signal the window turns into a status message,
and the bytes on disk. Geometry, hit-testing, handles and the undo stacks
belong to :mod:`polyx.core.editing`; nothing here reaches for them in the
canvas. The gesture adapters are shared with the contract tests
(``tests/canvas_rigs.py``), so both modules drive the widget the same way.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from polyx.core.editing import MIN_SIDE_PX, rejection_message
from polyx.core.labels import label_path_for
from tests.canvas_rigs import LabellerRig
from tests.conftest import destroy_windows, write_image

# The bytes the pre-migration canvas wrote for ``test_the_written_yolo_lines...``
# (captured from 80b19a7 with this same event script). A migration that
# shifts a coordinate convention changes this literal.
CHARACTERIZED_YOLO = (
    b"0 0.249474 0.324737 0.332632 0.401053\n"
    b"1 0.668421 0.315263 0.336842 0.356842\n"
    b"2 0.300000 0.776316 0.391579 0.325263\n"
)


@pytest.fixture
def labeller(qapp, tmp_path):
    """A canvas over a two-image folder, shown offscreen at a known size.

    The size is the one the characterization literal was captured at: it
    fixes the widget → image rounding the saved coordinates record.
    """
    image_dir = tmp_path / "imgs"
    write_image(image_dir / "a.png")
    write_image(image_dir / "b.png")
    rig = LabellerRig.over(qapp, image_dir, size=(500, 500))
    yield rig
    rig.canvas.deleteLater()


def test_a_draw_produces_one_box_and_selects_it(labeller):
    rig = labeller

    rig.drag((5, 5), (25, 21))

    boxes = rig.state.current_boxes
    assert len(boxes) == 1
    assert boxes[0].class_id == rig.state.active_class
    assert rig.selection() == 0
    # The image is 60×40: image pixel (15, 13) is the box centre.
    assert boxes[0].cx == pytest.approx(15 / 60, abs=1e-3)
    assert boxes[0].cy == pytest.approx(13 / 40, abs=1e-3)


def test_the_written_yolo_lines_are_byte_identical_to_the_pre_migration_output(labeller):
    """Characterization: the same gestures must write the same bytes.

    The expected literal was captured from the pre-migration canvas with
    this script (three boxes, one per class, disjoint areas). The empty
    clicks only deselect: a class change now reclassifies a selected box
    (review P3), and the bytes are still the pre-migration ones.
    """
    rig = labeller

    rig.drag((5, 5), (25, 21))
    rig.click((45, 35))  # deselect: a class change now reclassifies a selection
    rig.set_class(1)
    rig.drag((30, 5.5), (50.25, 19.75))
    rig.click((45, 35))
    rig.set_class(2)
    rig.drag((6.25, 24.5), (29.75, 37.5))

    rig.state.save_current()

    assert label_path_for(rig.state.current_image).read_bytes() == CHARACTERIZED_YOLO


def test_a_box_under_two_px_is_refused_visibly_and_nothing_is_written(labeller):
    rig = labeller
    rejected = []
    rig.canvas.box_rejected.connect(rejected.append)

    rig.drag((5, 5), (6, 30))  # 1 px wide
    rig.state.save_current()

    assert rig.state.current_boxes == []
    assert len(rejected) == 1
    assert rejected[0] < MIN_SIDE_PX
    # Nothing written: the census rule still owns the empty file.
    assert not label_path_for(rig.state.current_image).exists()


def test_the_rejection_message_is_shown_by_the_window(tmp_path, qapp, capfd):
    from polyx.gui.labeler.window import LabelerWindow

    image_dir = tmp_path / "imgs"
    image = write_image(image_dir / "a.png")

    window = LabelerWindow()
    rejected = []
    try:
        window.show()
        window.state.load_images(image_dir)
        window.canvas.box_rejected.connect(rejected.append)
        qapp.processEvents()
        rig = LabellerRig(window.state, window.canvas)
        # Read the status straight after the release: a language change can
        # rebuild the window (and its labels) on the next event flush.
        rig.drag((5, 5), (6, 30), flush=False)
        status = window.lbl_status.text()
        assert window.state.current_boxes == []
    finally:
        destroy_windows([window], qapp)

    assert len(rejected) == 1
    assert status == rejection_message(rejected[0])
    assert not label_path_for(image).exists()
    # Pump beyond the rejection message's 3.5 s reset, with the window gone.
    # An unowned timer would still try to update the deleted status label.
    capfd.readouterr()
    QTest.qWait(3800)
    assert "Traceback" not in capfd.readouterr().err


def test_a_move_through_a_real_drag_translates_the_box(labeller):
    rig = labeller
    rig.drag((5, 5), (25, 21))
    before = rig.state.current_editor.boxes[0]

    rig.drag((15, 13), (20, 13))  # grab the centre, drag 5 px right

    moved = rig.state.current_editor.boxes[0]
    assert moved.x1 == pytest.approx(before.x1 + 5, abs=0.2)
    assert moved.y1 == pytest.approx(before.y1, abs=0.2)
    assert moved.width == pytest.approx(before.width, abs=0.2)


def test_a_resize_from_the_bottom_right_handle_grows_the_box(labeller):
    rig = labeller
    rig.drag((5, 5), (25, 21))
    before = rig.state.current_editor.boxes[0]

    rig.drag((25, 21), (30, 26))  # bottom-right corner outward

    resized = rig.state.current_editor.boxes[0]
    assert resized.x1 == pytest.approx(before.x1, abs=0.2)
    assert resized.y1 == pytest.approx(before.y1, abs=0.2)
    assert resized.width == pytest.approx(before.width + 5, abs=0.3)
    assert resized.height == pytest.approx(before.height + 5, abs=0.3)


def test_ctrl_z_and_ctrl_y_undo_and_redo_a_draw(labeller):
    rig = labeller
    rig.drag((5, 5), (25, 21))
    assert len(rig.state.current_boxes) == 1

    rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert rig.state.current_boxes == []

    rig.key(Qt.Key_Y, Qt.ControlModifier)
    assert len(rig.state.current_boxes) == 1


def test_per_image_history_survives_navigating_away_and_back(labeller, qapp):
    rig = labeller
    first = rig.state.current_image
    rig.drag((5, 5), (25, 21))

    rig.state.next_image()
    qapp.processEvents()
    assert rig.state.current_image != first
    assert rig.state.current_boxes == []

    rig.state.prev_image()
    qapp.processEvents()
    assert rig.state.current_image == first
    assert len(rig.state.current_boxes) == 1

    rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert rig.state.current_boxes == []


def test_delete_and_undo_remove_and_restore_the_selection(labeller):
    rig = labeller
    rig.drag((5, 5), (25, 21))

    rig.key(Qt.Key_Delete)
    assert rig.state.current_boxes == []
    assert rig.selection() == -1

    rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(rig.state.current_boxes) == 1
    assert rig.selection() == 0


def test_keys_1_to_9_change_the_class_of_the_next_draw(labeller):
    rig = labeller

    rig.key(Qt.Key_3)
    assert rig.state.active_class == 2
    assert rig.state.current_editor.active_class == 2

    rig.drag((5, 5), (25, 21))
    assert rig.state.current_boxes[0].class_id == 2


def test_a_click_on_empty_space_clears_the_selection(labeller):
    rig = labeller
    rig.drag((5, 5), (25, 21))
    assert rig.selection() == 0

    rig.click((45, 35))

    assert rig.selection() == -1
    assert len(rig.state.current_boxes) == 1  # the zero-size click drew nothing


def test_middle_button_drag_pans_the_view(labeller):
    rig = labeller
    canvas = rig.canvas
    before = QPointF(canvas._offset)

    QTest.mousePress(canvas, Qt.MiddleButton, Qt.NoModifier, QPoint(100, 100))
    QTest.mouseMove(canvas, QPoint(130, 115))
    QTest.mouseRelease(canvas, Qt.MiddleButton, Qt.NoModifier, QPoint(130, 115))
    QApplication.processEvents()

    assert canvas._offset.x() == pytest.approx(before.x() + 30, abs=1)
    assert canvas._offset.y() == pytest.approx(before.y() + 15, abs=1)


def test_the_wheel_zooms_around_the_cursor(labeller):
    rig = labeller
    canvas = rig.canvas
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
