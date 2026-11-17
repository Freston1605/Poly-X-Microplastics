"""The Manual GT canvas through real Qt events (ticket #146).

The GT page's ``AnnotCanvas`` used to be a second, independent box editor: its
own geometry, hit-testing, resize handles and undo stacks. It now delegates
every editing operation to ``polyx.core.editing.BoxEditor`` and keeps only
event translation, zoom/pan, painting and its signals. These tests drive that
shell with the events a user produces (a drag, a key press, the wheel) and
assert the observable result: the boxes, the detections and the visible
rejection message.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from polyx.core.detect.yolo import Detection
from polyx.core.editing import BoxEditor
from polyx.gui.detector.pages.gt_manual import AnnotCanvas

IMAGE_W, IMAGE_H = 400, 300
CLASSES = ["PET", "PP", "HDPE"]


# ── Fixtures and event helpers ────────────────────────────────────────────


def _write_image(path: Path) -> Path:
    ok, buffer = cv2.imencode(".png", np.full((IMAGE_H, IMAGE_W, 3), 40, np.uint8))
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


@pytest.fixture
def canvas(qapp, tmp_path):
    """A shown canvas with a loaded image, 1:1 with the scene (zoom 100 %)."""
    path = _write_image(tmp_path / "plate.png")
    widget = AnnotCanvas()
    widget.resize(800, 600)
    widget.show()
    widget.set_class_names(CLASSES)
    widget.load_image(path, [])
    widget.zoom_100()
    qapp.processEvents()
    yield widget, path
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _load(canvas: AnnotCanvas, path: Path, boxes: list[Detection]) -> None:
    canvas.load_image(path, boxes)
    canvas.zoom_100()
    QApplication.processEvents()


def _scene(canvas: AnnotCanvas, point: tuple[float, float] | QPointF) -> QPointF:
    return QPointF(canvas.mapFromScene(QPointF(*point)))


def _mouse(
    canvas: AnnotCanvas,
    kind: QEvent.Type,
    point: QPointF,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> None:
    # A QGraphicsView receives mouse events through its viewport, exactly as a
    # user's click does; positions are in viewport (= view) coordinates.
    event = QMouseEvent(
        kind,
        point,
        canvas.viewport().mapToGlobal(point),
        button,
        buttons,
        Qt.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), event)


def _drag(
    canvas: AnnotCanvas,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """A left-button drag between two image (scene) points."""
    _mouse(
        canvas, QMouseEvent.MouseButtonPress, _scene(canvas, start), Qt.LeftButton, Qt.LeftButton
    )
    _mouse(canvas, QMouseEvent.MouseMove, _scene(canvas, end), Qt.NoButton, Qt.LeftButton)
    _mouse(canvas, QMouseEvent.MouseButtonRelease, _scene(canvas, end), Qt.LeftButton, Qt.NoButton)


def _click(canvas: AnnotCanvas, point: tuple[float, float]) -> None:
    scene = _scene(canvas, point)
    _mouse(canvas, QMouseEvent.MouseButtonPress, scene, Qt.LeftButton, Qt.LeftButton)
    _mouse(canvas, QMouseEvent.MouseButtonRelease, scene, Qt.LeftButton, Qt.NoButton)


def _key(canvas: AnnotCanvas, key: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.NoModifier) -> None:
    QApplication.sendEvent(canvas, QKeyEvent(QEvent.KeyPress, key, modifiers))


def _wheel(canvas: AnnotCanvas, steps: int = 1) -> float:
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


def _middle_drag(canvas: AnnotCanvas, dx: float = 60.0, dy: float = 40.0) -> None:
    start = QPointF(canvas.width() / 2, canvas.height() / 2)
    end = QPointF(start.x() + dx, start.y() + dy)
    _mouse(canvas, QMouseEvent.MouseButtonPress, start, Qt.MiddleButton, Qt.MiddleButton)
    _mouse(canvas, QMouseEvent.MouseMove, end, Qt.NoButton, Qt.MiddleButton)
    _mouse(canvas, QMouseEvent.MouseButtonRelease, end, Qt.MiddleButton, Qt.NoButton)


def _assert_box(box, expected: tuple[float, float, float, float]) -> None:
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx(expected, abs=1.0)


# ── Delegation to the core ────────────────────────────────────────────────


def test_the_canvas_delegates_editing_to_the_core_editor(canvas):
    widget, _path = canvas

    assert isinstance(widget.editor, BoxEditor)

    # No private geometry, hit-testing, handle or undo implementation survives.
    for name in ("_hit_handle", "_hit_box", "_handle_points", "_push_undo", "_snapshot"):
        assert not hasattr(widget, name), f"the canvas still implements {name}"
    assert not hasattr(widget, "_undo")
    assert not hasattr(widget, "_redo")


# ── Draw ─────────────────────────────────────────────────────────────────


def test_a_real_draw_gesture_creates_one_box_and_one_detection(canvas):
    widget, _path = canvas
    widget.set_active_class(1)

    _drag(widget, (50, 40), (150, 140))

    assert len(widget.boxes) == 1
    box = widget.boxes[0]
    _assert_box(box, (50, 40, 150, 140))
    assert box.class_id == 1

    dets = widget.detections()
    assert len(dets) == 1
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == pytest.approx(
        (50, 40, 150, 140), abs=1.0
    )
    assert dets[0].class_id == 1
    assert dets[0].class_name == "PP"
    assert dets[0].conf == 1.0


def test_a_box_drawn_outside_the_image_is_clipped_to_its_edge(canvas):
    widget, _path = canvas

    _drag(widget, (350, 250), (450, 350))

    assert len(widget.boxes) == 1
    _assert_box(widget.boxes[0], (350, 250, IMAGE_W, IMAGE_H))


# ── The 2 px minimum side ────────────────────────────────────────────────


def test_a_too_small_draw_is_refused_with_a_visible_message(canvas):
    widget, _path = canvas
    messages: list[str] = []
    widget.status_changed.connect(messages.append)

    _drag(widget, (20, 20), (21, 80))

    assert widget.boxes == []
    assert widget.detections() == []
    assert messages, "the rejection was not announced"
    assert "descartada" in messages[-1]
    assert "(mínimo 2 px)" in messages[-1]


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
        canvas = page.canvas
        canvas.resize(800, 600)
        canvas.show()
        canvas.zoom_100()
        qapp.processEvents()

        _drag(canvas, (20, 20), (21, 38))

        assert page.canvas.boxes == []
        assert "descartada" in page.lbl_canvas_status.text()
        page._auto_save(image)
        assert not list(root.rglob("*.txt"))
    finally:
        page.close()
        page.deleteLater()
        qapp.processEvents()


# ── Move ─────────────────────────────────────────────────────────────────


def test_a_real_drag_moves_the_box(canvas):
    widget, path = canvas
    _load(
        widget,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=100, y1=100, x2=200, y2=200)],
    )

    _drag(widget, (150, 150), (180, 170))

    assert len(widget.boxes) == 1
    _assert_box(widget.boxes[0], (130, 120, 230, 220))
    assert widget.boxes[0].class_id == 0
    _assert_box(widget.detections()[0], (130, 120, 230, 220))


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
    widget, path = canvas
    _load(
        widget,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=50, y1=50, x2=150, y2=150)],
    )
    _click(widget, (100, 100))  # select, so the handles exist

    _drag(widget, start, end)

    assert len(widget.boxes) == 1
    _assert_box(widget.boxes[0], expected)


# ── Undo / redo ──────────────────────────────────────────────────────────


def test_ctrl_z_and_ctrl_y_undo_and_redo_through_real_key_events(canvas):
    widget, _path = canvas
    _drag(widget, (50, 40), (150, 140))
    assert len(widget.boxes) == 1

    _key(widget, Qt.Key_Z, Qt.ControlModifier)
    assert widget.boxes == []

    _key(widget, Qt.Key_Y, Qt.ControlModifier)
    assert len(widget.boxes) == 1
    _assert_box(widget.boxes[0], (50, 40, 150, 140))


def test_undo_after_a_move_restores_the_previous_geometry(canvas):
    widget, path = canvas
    _load(
        widget,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=100, y1=100, x2=200, y2=200)],
    )
    _drag(widget, (150, 150), (180, 170))
    _assert_box(widget.boxes[0], (130, 120, 230, 220))

    widget.undo()

    _assert_box(widget.boxes[0], (100, 100, 200, 200))


# ── Class keys, delete, clear ────────────────────────────────────────────


def test_class_keys_reclassify_the_selection_and_delete_removes_it(canvas):
    widget, path = canvas
    _load(
        widget,
        path,
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=50, y1=50, x2=150, y2=150)],
    )
    _click(widget, (100, 100))

    _key(widget, Qt.Key_3)

    assert widget.active_class == 2
    assert widget.boxes[0].class_id == 2

    _key(widget, Qt.Key_Delete)

    assert widget.boxes == []


def test_clear_all_removes_every_box_and_is_undoable(canvas):
    widget, path = canvas
    _load(
        widget,
        path,
        [
            Detection(class_id=0, class_name="PET", conf=1.0, x1=20, y1=20, x2=60, y2=60),
            Detection(class_id=1, class_name="PP", conf=1.0, x1=100, y1=100, x2=160, y2=160),
        ],
    )

    widget.clear_boxes()
    assert widget.boxes == []

    widget.undo()
    assert len(widget.boxes) == 2


# ── Zoom and pan ─────────────────────────────────────────────────────────


def test_the_wheel_zooms_and_a_middle_drag_pans(canvas):
    widget, _path = canvas
    before = widget.transform().m11()

    before = _wheel(widget, steps=4)
    assert widget.transform().m11() > before

    # Start from the middle of the scroll range: a drag from a clamped edge
    # would leave the value at 0 and prove nothing.
    widget.horizontalScrollBar().setValue(widget.horizontalScrollBar().maximum() // 2)
    widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum() // 2)
    h_before = widget.horizontalScrollBar().value()
    v_before = widget.verticalScrollBar().value()
    _middle_drag(widget)

    assert (
        widget.horizontalScrollBar().value(),
        widget.verticalScrollBar().value(),
    ) != (h_before, v_before)
