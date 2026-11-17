"""The Manual GT canvas through real Qt events (ticket #146).

The GT page's ``AnnotCanvas`` used to be a second, independent box editor: its
own geometry, hit-testing, resize handles and undo stacks. It now delegates
every editing operation to ``polyx.core.editing.BoxEditor`` and keeps only
event translation, zoom/pan, painting and its signals. These tests drive that
shell with the events a user produces (a drag, a key press, the wheel) and
assert the observable result: the boxes, the detections and the visible
rejection message. The gesture adapter is shared with the contract tests
(``tests/canvas_rigs.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from polyx.core.detect.yolo import Detection
from polyx.core.editing import BoxEditor, rejection_message
from tests.canvas_rigs import GtRig
from tests.conftest import write_image

IMAGE_W, IMAGE_H = 400, 300
CLASSES = ["PET", "PP", "HDPE"]


# ── Fixtures and event helpers ────────────────────────────────────────────


@pytest.fixture
def canvas(qapp, tmp_path):
    """A shown canvas with a loaded image, 1:1 with the scene (zoom 100 %)."""
    path = write_image(tmp_path / "plate.png", IMAGE_W, IMAGE_H, value=40)
    rig = GtRig.over(qapp, path, CLASSES)
    yield rig, path
    rig.canvas.close()
    rig.canvas.deleteLater()
    qapp.processEvents()


def _load(rig: GtRig, path: Path, boxes: list[Detection]) -> None:
    rig.canvas.load_image(path, boxes)
    rig.canvas.zoom_100()
    QApplication.processEvents()


def _wheel(rig: GtRig, steps: int = 1) -> float:
    canvas = rig.canvas
    position = QPointF(canvas.width() / 2, canvas.height() / 2)
    before = canvas.transform().m11()
    for _step in range(steps):
        QApplication.sendEvent(
            canvas.viewport(),
            QWheelEvent(
                position,
                canvas.viewport().mapToGlobal(position),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.NoButton,
                Qt.NoModifier,
                Qt.NoScrollPhase,
                False,
            ),
        )
    return before


def _middle_drag(rig: GtRig, dx: float = 60.0, dy: float = 40.0) -> None:
    canvas = rig.canvas
    start = QPointF(canvas.width() / 2, canvas.height() / 2)
    end = QPointF(start.x() + dx, start.y() + dy)

    def send(kind, point, button, buttons):
        QApplication.sendEvent(
            canvas.viewport(),
            QMouseEvent(
                kind,
                point,
                canvas.viewport().mapToGlobal(point),
                button,
                buttons,
                Qt.NoModifier,
            ),
        )

    send(QMouseEvent.MouseButtonPress, start, Qt.MiddleButton, Qt.MiddleButton)
    send(QMouseEvent.MouseMove, end, Qt.NoButton, Qt.MiddleButton)
    send(QMouseEvent.MouseButtonRelease, end, Qt.MiddleButton, Qt.NoButton)


def _assert_box(box, expected: tuple[float, float, float, float]) -> None:
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx(expected, abs=1.0)


# ── Delegation to the core ────────────────────────────────────────────────


def test_the_canvas_delegates_editing_to_the_core_editor(canvas):
    rig, _path = canvas

    assert isinstance(rig.canvas.editor, BoxEditor)

    # No private geometry, hit-testing, handle or undo implementation survives.
    for name in ("_hit_handle", "_hit_box", "_handle_points", "_push_undo", "_snapshot"):
        assert not hasattr(rig.canvas, name), f"the canvas still implements {name}"
    assert not hasattr(rig.canvas, "_undo")
    assert not hasattr(rig.canvas, "_redo")


# ── Draw ─────────────────────────────────────────────────────────────────


def test_a_real_draw_gesture_creates_one_box_and_one_detection(canvas):
    rig, _path = canvas
    rig.set_class(1)

    rig.drag((50, 40), (150, 140))

    assert len(rig.boxes()) == 1
    box = rig.canvas.boxes[0]
    _assert_box(box, (50, 40, 150, 140))
    assert box.class_id == 1

    dets = rig.canvas.detections()
    assert len(dets) == 1
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == pytest.approx(
        (50, 40, 150, 140), abs=1.0
    )
    assert dets[0].class_id == 1
    assert dets[0].class_name == "PP"
    assert dets[0].conf == 1.0


def test_a_box_drawn_outside_the_image_is_clipped_to_its_edge(canvas):
    rig, _path = canvas

    rig.drag((350, 250), (450, 350))

    assert len(rig.boxes()) == 1
    _assert_box(rig.canvas.boxes[0], (350, 250, IMAGE_W, IMAGE_H))


# ── The 2 px minimum side ────────────────────────────────────────────────


def test_a_too_small_draw_is_refused_with_a_visible_message(canvas):
    rig, _path = canvas
    messages: list[str] = []
    rig.canvas.status_changed.connect(messages.append)

    rig.drag((20, 20), (21, 80))

    assert rig.boxes() == []
    assert rig.canvas.detections() == []
    assert messages, "the rejection was not announced"
    assert messages[-1] == rejection_message(1.0)


def test_a_too_small_box_is_refused_and_the_page_writes_no_txt(qapp, tmp_path):
    """The page shows the message; an empty canvas leaves no label behind."""
    from polyx.gui.detector.pages.gt_manual import GTManualPage
    from polyx.gui.detector.state import DetectorState
    from tests.conftest import split_dataset

    root = split_dataset(tmp_path / "dataset", {"train": {"plate": None}})
    image = root / "images" / "train" / "plate.png"
    state = DetectorState()
    state.images = [image]
    page = GTManualPage(state)
    try:
        rig = GtRig(page.canvas)
        page.canvas.resize(800, 600)
        page.canvas.show()
        page.canvas.zoom_100()
        qapp.processEvents()

        rig.drag((20, 20), (21, 38))

        assert page.canvas.boxes == []
        assert page.lbl_canvas_status.text() == rejection_message(1.0)
        page._auto_save(image)
        assert not list(root.rglob("*.txt"))
    finally:
        page.close()
        page.deleteLater()
        qapp.processEvents()


# ── Move ─────────────────────────────────────────────────────────────────


def test_a_real_drag_moves_the_box(canvas):
    rig, path = canvas
    _load(
        rig,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=100, y1=100, x2=200, y2=200)],
    )

    rig.drag((150, 150), (180, 170))

    assert len(rig.boxes()) == 1
    _assert_box(rig.canvas.boxes[0], (130, 120, 230, 220))
    assert rig.canvas.boxes[0].class_id == 0
    _assert_box(rig.canvas.detections()[0], (130, 120, 230, 220))


# ── Resize from every handle ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((50, 50), (30, 30), (30, 30, 150, 150)),  # TOP_LEFT
        ((150, 50), (170, 30), (50, 30, 170, 150)),  # TOP_RIGHT
        ((150, 150), (170, 170), (50, 50, 170, 170)),  # BOTTOM_RIGHT
        ((50, 150), (30, 170), (30, 50, 150, 170)),  # BOTTOM_LEFT
        ((100, 50), (100, 30), (50, 30, 150, 150)),  # TOP
        ((150, 100), (170, 100), (50, 50, 170, 150)),  # RIGHT
        ((100, 150), (100, 170), (50, 50, 150, 170)),  # BOTTOM
        ((50, 100), (30, 100), (30, 50, 150, 150)),  # LEFT
    ],
)
def test_a_real_drag_resizes_from_each_handle(canvas, start, end, expected):
    rig, path = canvas
    _load(
        rig,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=50, y1=50, x2=150, y2=150)],
    )
    rig.click((100, 100))  # select, so the handles exist

    rig.drag(start, end)

    assert len(rig.boxes()) == 1
    _assert_box(rig.canvas.boxes[0], expected)


# ── Undo / redo ──────────────────────────────────────────────────────────


def test_ctrl_z_and_ctrl_y_undo_and_redo_through_real_key_events(canvas):
    rig, _path = canvas
    rig.drag((50, 40), (150, 140))
    assert len(rig.boxes()) == 1

    rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert rig.boxes() == []

    rig.key(Qt.Key_Y, Qt.ControlModifier)
    assert len(rig.boxes()) == 1
    _assert_box(rig.canvas.boxes[0], (50, 40, 150, 140))


def test_undo_after_a_move_restores_the_previous_geometry(canvas):
    rig, path = canvas
    _load(
        rig,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=100, y1=100, x2=200, y2=200)],
    )
    rig.drag((150, 150), (180, 170))
    _assert_box(rig.canvas.boxes[0], (130, 120, 230, 220))

    rig.canvas.undo()

    _assert_box(rig.canvas.boxes[0], (100, 100, 200, 200))


# ── Class keys, delete, clear ────────────────────────────────────────────


def test_class_keys_reclassify_the_selection_and_delete_removes_it(canvas):
    rig, path = canvas
    _load(
        rig,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=50, y1=50, x2=150, y2=150)],
    )
    rig.click((100, 100))

    rig.key(Qt.Key_3)

    assert rig.active_class() == 2
    assert rig.canvas.boxes[0].class_id == 2

    rig.key(Qt.Key_Delete)

    assert rig.boxes() == []


def test_clear_all_removes_every_box_and_is_undoable(canvas):
    rig, path = canvas
    _load(
        rig,
        path,
        [
            Detection(class_id=0, class_name="PET", conf=1.0, x1=20, y1=20, x2=60, y2=60),
            Detection(class_id=1, class_name="PP", conf=1.0, x1=100, y1=100, x2=160, y2=160),
        ],
    )

    rig.canvas.clear_boxes()
    assert rig.boxes() == []

    rig.canvas.undo()
    assert len(rig.boxes()) == 2


# ── Zoom and pan ─────────────────────────────────────────────────────────


def test_the_wheel_zooms_and_a_middle_drag_pans(canvas):
    rig, _path = canvas
    canvas_widget = rig.canvas
    before = canvas_widget.transform().m11()

    before = _wheel(rig, steps=4)
    assert canvas_widget.transform().m11() > before

    # Start from the middle of the scroll range: a drag from a clamped edge
    # would leave the value at 0 and prove nothing.
    canvas_widget.horizontalScrollBar().setValue(canvas_widget.horizontalScrollBar().maximum() // 2)
    canvas_widget.verticalScrollBar().setValue(canvas_widget.verticalScrollBar().maximum() // 2)
    h_before = canvas_widget.horizontalScrollBar().value()
    v_before = canvas_widget.verticalScrollBar().value()
    _middle_drag(rig)

    assert (
        canvas_widget.horizontalScrollBar().value(),
        canvas_widget.verticalScrollBar().value(),
    ) != (h_before, v_before)
