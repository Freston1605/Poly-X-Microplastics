"""One editing path for the two annotation canvases — the card 07 contract (#147).

The Labeller's ``BboxCanvas`` and the Manual GT page's ``AnnotCanvas`` are two
thin shells over :mod:`polyx.core.editing`. Card 07 moved the editing domain
into that core; this module pins the part the shells still own and that must
stay identical:

* the same gesture sequence (draw, move, resize, delete, undo/redo, an empty
  click) leaves the same boxes and the same selection on both;
* the same sub-2 px box is refused by the same core invariant with the same
  visible message;
* a class is painted with the colour the one theme helper gives it, including
  classes outside the palette — a local colour table would paint another.

The Qt-free guarantee of the core is owned by
``tests/test_box_editing.py::test_the_editing_core_imports_without_qt``; the
guard here covers the static half (no Qt import in the core source) and the
shells (no private editing maths, no colour tables, no minimum-side alias).

The Viewer canvas also carries a ``_CLASS_COLORS`` table with the same literal
palette. It is out of card 07's scope (the two annotation editors) and its
index-ordered colours are a different rule from the theme's name lookup, so it
is left alone and reported rather than silently re-coloured.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from polyx.core.editing import MIN_SIDE_PX
from polyx.core.labels import STUDY_CLASSES
from polyx.gui import theme as T
from polyx.gui.detector.pages import gt_manual
from polyx.gui.detector.pages.gt_manual import AnnotCanvas
from polyx.gui.labeler import canvas as labeler_canvas
from polyx.gui.labeler import window as labeler_window
from polyx.gui.labeler.canvas import BboxCanvas
from polyx.gui.labeler.state import LabelerState

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

IMAGE_W, IMAGE_H = 400, 300
CLASSES = [*STUDY_CLASSES, "HDPE"]


# ── The two rigs: the same gestures through each shell's real events ──────


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", np.full((IMAGE_H, IMAGE_W, 3), 40, np.uint8))
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


@pytest.fixture
def image(tmp_path) -> Path:
    return _write_image(tmp_path / "imgs" / "plate.png")


class _Rig:
    """The editing gestures a labeller makes, driven through one shell.

    Both editors receive the same image coordinates; each adapter converts
    them with its own widget transform, which is exactly the shell concern
    that remains after card 07.
    """

    canvas: object

    def boxes(self) -> list[tuple[float, float, float, float, int]]: ...

    def selection(self) -> int: ...

    def can_undo(self) -> bool: ...

    def set_class(self, class_id: int) -> None: ...

    def drag(
        self, start: tuple[float, float], end: tuple[float, float], flush: bool = True
    ) -> None: ...

    def click(self, point: tuple[float, float]) -> None: ...

    def key(self, key: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.NoModifier) -> None: ...


class _LabellerRig(_Rig):
    def __init__(self, state: LabelerState, canvas: BboxCanvas):
        self.state = state
        self.canvas = canvas

    @classmethod
    def over(cls, qapp, image: Path) -> _LabellerRig:
        state = LabelerState()
        state.set_class_names(list(CLASSES))
        canvas = BboxCanvas(state)
        canvas.resize(800, 600)
        canvas.show()
        state.load_images(image.parent)
        qapp.processEvents()
        return cls(state, canvas)

    def _point(self, image_xy: tuple[float, float]) -> QPoint:
        point = self.canvas._i2w(QPointF(*image_xy))
        return QPoint(round(point.x()), round(point.y()))

    def boxes(self):
        editor = self.state.current_editor
        return (
            [] if editor is None else [(b.x1, b.y1, b.x2, b.y2, b.class_id) for b in editor.boxes]
        )

    def selection(self) -> int:
        editor = self.state.current_editor
        return -1 if editor is None else editor.selection

    def can_undo(self) -> bool:
        editor = self.state.current_editor
        return False if editor is None else editor.can_undo

    def set_class(self, class_id: int) -> None:
        self.state.set_active_class(class_id)

    def drag(self, start, end, flush: bool = True) -> None:
        QTest.mousePress(self.canvas, Qt.LeftButton, Qt.NoModifier, self._point(start))
        QTest.mouseMove(self.canvas, self._point(end))
        QTest.mouseRelease(self.canvas, Qt.LeftButton, Qt.NoModifier, self._point(end))
        if flush:
            QApplication.processEvents()

    def click(self, point) -> None:
        QTest.mousePress(self.canvas, Qt.LeftButton, Qt.NoModifier, self._point(point))
        QTest.mouseRelease(self.canvas, Qt.LeftButton, Qt.NoModifier, self._point(point))
        QApplication.processEvents()

    def key(self, key, modifiers=Qt.NoModifier) -> None:
        QTest.keyClick(self.canvas, key, modifiers)
        QApplication.processEvents()

    def tag_color(self, name: str) -> QColor:
        """Colour the canvas painted the class tag of its newest box."""
        x1, y1, x2, y2, _cls = self.boxes()[-1]
        rect = self.canvas._box_rect(x1, y1, x2, y2)
        tag = QRectF(rect.x(), rect.y() - 17, max(50, len(name) * 7 + 8), 17)
        image = self.canvas.grab().toImage()
        return image.pixelColor(round(tag.right() - 4), round(tag.y() + 8))


class _GtRig(_Rig):
    def __init__(self, canvas: AnnotCanvas):
        self.canvas = canvas

    @classmethod
    def over(cls, qapp, image: Path) -> _GtRig:
        canvas = AnnotCanvas()
        canvas.resize(800, 600)
        canvas.show()
        canvas.set_class_names(list(CLASSES))
        canvas.load_image(image, [])
        canvas.zoom_100()
        qapp.processEvents()
        return cls(canvas)

    def _point(self, image_xy: tuple[float, float]) -> QPointF:
        return QPointF(self.canvas.mapFromScene(QPointF(*image_xy)))

    def _send(self, kind, point: QPointF, button, buttons) -> None:
        event = QMouseEvent(
            kind,
            point,
            self.canvas.viewport().mapToGlobal(point),
            button,
            buttons,
            Qt.NoModifier,
        )
        QApplication.sendEvent(self.canvas.viewport(), event)

    def boxes(self):
        return [(b.x1, b.y1, b.x2, b.y2, b.class_id) for b in self.canvas.boxes]

    def selection(self) -> int:
        return self.canvas.selection

    def can_undo(self) -> bool:
        editor = self.canvas.editor
        return False if editor is None else editor.can_undo

    def set_class(self, class_id: int) -> None:
        self.canvas.set_active_class(class_id)

    def drag(self, start, end, flush: bool = True) -> None:
        self._send(QMouseEvent.MouseButtonPress, self._point(start), Qt.LeftButton, Qt.LeftButton)
        self._send(QMouseEvent.MouseMove, self._point(end), Qt.NoButton, Qt.LeftButton)
        self._send(QMouseEvent.MouseButtonRelease, self._point(end), Qt.LeftButton, Qt.NoButton)
        if flush:
            QApplication.processEvents()

    def click(self, point) -> None:
        self._send(QMouseEvent.MouseButtonPress, self._point(point), Qt.LeftButton, Qt.LeftButton)
        self._send(QMouseEvent.MouseButtonRelease, self._point(point), Qt.LeftButton, Qt.NoButton)
        QApplication.processEvents()

    def key(self, key, modifiers=Qt.NoModifier) -> None:
        QApplication.sendEvent(self.canvas, QKeyEvent(QEvent.KeyPress, key, modifiers))
        QApplication.processEvents()

    def tag_color(self, name: str) -> QColor:
        """Colour the scene painted the class tag of its newest box."""
        x1, y1, _x2, _y2, _cls = self.boxes()[-1]
        width = len(name) * 9 + 12
        point = self.canvas.mapFromScene(QPointF(x1 + width - 4, y1 - 18 + 9))
        return self.canvas.viewport().grab().toImage().pixelColor(point)


@pytest.fixture
def rigs(qapp, image):
    labeller = _LabellerRig.over(qapp, image)
    gt = _GtRig.over(qapp, image)
    yield labeller, gt
    labeller.canvas.deleteLater()
    gt.canvas.close()
    gt.canvas.deleteLater()
    qapp.processEvents()


# ── The contract: one gesture sequence, one result ─────────────────────────


def _assert_same(labeller: _Rig, gt: _Rig) -> None:
    left, right = labeller.boxes(), gt.boxes()
    assert len(left) == len(right), (
        f"{len(left)} boxes in the Labeller, {len(right)} in the GT page"
    )
    for labeller_box, gt_box in zip(left, right, strict=True):
        assert labeller_box[:4] == pytest.approx(gt_box[:4], abs=1.0), (
            f"Labeller {labeller_box} != GT {gt_box}"
        )
        assert labeller_box[4] == gt_box[4]
    assert labeller.selection() == gt.selection()


def _move_by(rig: _Rig, index: int, dx: float, dy: float) -> None:
    x1, y1, x2, y2, _cls = rig.boxes()[index]
    centre = ((x1 + x2) / 2, (y1 + y2) / 2)
    rig.drag(centre, (centre[0] + dx, centre[1] + dy))


def _resize_bottom_right_by(rig: _Rig, index: int, dx: float, dy: float) -> None:
    _x1, _y1, x2, y2, _cls = rig.boxes()[index]
    rig.drag((x2, y2), (x2 + dx, y2 + dy))


def test_the_same_draw_move_resize_delete_and_undo_leaves_both_editors_identical(rigs):
    labeller, gt = rigs

    # Draw two boxes; each drawing is selected as it lands.
    for rig in (labeller, gt):
        rig.drag((50, 40), (150, 140))
    assert len(labeller.boxes()) == len(gt.boxes()) == 1
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 0

    for rig in (labeller, gt):
        rig.drag((200, 60), (280, 160))
    assert len(labeller.boxes()) == len(gt.boxes()) == 2
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 1

    # Move the newest box by grabbing its centre: the press selects it and
    # the drag translates it by the same delta in both editors.
    _move_by(labeller, 1, 20, 10)
    _move_by(gt, 1, 20, 10)
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 1

    # Resize it from its bottom-right handle: the box only grows there.
    _resize_bottom_right_by(labeller, 1, 15, 15)
    _resize_bottom_right_by(gt, 1, 15, 15)
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 1

    # Draw a third box, then delete it: the selection goes with it.
    for rig in (labeller, gt):
        rig.drag((60, 200), (120, 260))
    assert len(labeller.boxes()) == len(gt.boxes()) == 3
    _assert_same(labeller, gt)
    for rig in (labeller, gt):
        rig.key(Qt.Key_Delete)
    assert labeller.boxes() and len(labeller.boxes()) == len(gt.boxes()) == 2
    assert labeller.selection() == gt.selection() == -1
    _assert_same(labeller, gt)

    # Undo the delete (box and selection come back), redo it, undo it again.
    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 3
    assert labeller.selection() == gt.selection() == 2
    _assert_same(labeller, gt)

    for rig in (labeller, gt):
        rig.key(Qt.Key_Y, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 2
    assert labeller.selection() == gt.selection() == -1
    _assert_same(labeller, gt)

    # Walk the rest of the history back: the delete again, the draw, the
    # resize and the move, asserting both editors at every step.
    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 3
    assert labeller.selection() == gt.selection() == 2
    _assert_same(labeller, gt)

    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 2
    _assert_same(labeller, gt)
    resized_height = labeller.boxes()[1][3] - labeller.boxes()[1][1]

    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    _assert_same(labeller, gt)
    restored_height = labeller.boxes()[1][3] - labeller.boxes()[1][1]
    assert restored_height == pytest.approx(resized_height - 15, abs=1.0)

    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    _assert_same(labeller, gt)
    assert labeller.boxes()[1][0] == pytest.approx(200, abs=1.0)


def test_an_empty_click_is_a_click_in_both_editors(rigs):
    """No box is drawn and the selection is cleared, as the Labeller does."""
    labeller, gt = rigs

    for rig in (labeller, gt):
        rig.click((350, 280))
    assert labeller.boxes() == gt.boxes() == []
    assert labeller.selection() == gt.selection() == -1
    assert labeller.can_undo() is False
    assert gt.can_undo() is False

    for rig in (labeller, gt):
        rig.drag((50, 40), (150, 140))
    assert labeller.selection() == gt.selection() == 0

    for rig in (labeller, gt):
        rig.click((350, 280))
    assert len(labeller.boxes()) == len(gt.boxes()) == 1
    assert labeller.selection() == gt.selection() == -1


# ── The 2 px rule: one core invariant, one on-screen message ──────────────


def _side_of(message: str) -> float:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*px", message)
    assert match, f"the message does not name the side: {message!r}"
    return float(match.group(1).replace(",", "."))


def test_both_canvases_refuse_a_sub_two_pixel_box_with_the_same_visible_message(qapp, image):
    from polyx.gui.labeler.window import LabelerWindow

    window = LabelerWindow()
    gt = _GtRig.over(qapp, image)
    messages: list[str] = []
    gt.canvas.status_changed.connect(messages.append)
    try:
        window.show()
        window.state.load_images(image.parent)
        qapp.processEvents()
        labeller = _LabellerRig(window.state, window.canvas)

        # A language change can rebuild the window on the next event flush;
        # read the status straight after the release, as the labeller sees it.
        labeller.drag((20, 20), (21, 80), flush=False)
        labeller_message = window.lbl_status.text()

        gt.drag((20, 20), (21, 80))
        gt_message = messages[-1]
    finally:
        gt.canvas.close()
        gt.canvas.deleteLater()
        window.deleteLater()
        qapp.processEvents()

    assert window.state.current_boxes == []
    assert labeller.can_undo() is False
    assert gt.boxes() == []
    assert gt.can_undo() is False

    for message in (labeller_message, gt_message):
        assert _side_of(message) < MIN_SIDE_PX
        assert f"{MIN_SIDE_PX:.0f} px" in message

    # The same template and the same minimum, only the measured side differs.
    assert re.sub(r"\d+(?:[.,]\d+)?\s*px", "<side>", labeller_message) == re.sub(
        r"\d+(?:[.,]\d+)?\s*px", "<side>", gt_message
    )


# ── One class-colour path ─────────────────────────────────────────────────


def test_class_qcolor_at_resolves_names_and_falls_back():
    assert T.class_qcolor_at(CLASSES, 0).name() == T.CLASS_COLOR_HEX["PET"]
    assert T.class_qcolor_at(CLASSES, 3).name() == T.FALLBACK_COLOR_HEX
    # Out of range and unknown both fall back instead of raising.
    assert T.class_qcolor_at(CLASSES, 99).name() == T.FALLBACK_COLOR_HEX
    assert T.class_qcolor_at([], 0).name() == T.FALLBACK_COLOR_HEX


@pytest.mark.parametrize("name", CLASSES)
def test_both_canvases_paint_the_class_colour_the_theme_gives(rigs, name):
    labeller, gt = rigs
    class_id = CLASSES.index(name)

    for rig in (labeller, gt):
        rig.set_class(class_id)
        rig.drag((50, 40), (150, 140))
        assert rig.boxes()[0][4] == class_id

    expected = T.class_qcolor(name).name()
    assert labeller.tag_color(name).name() == expected
    assert gt.tag_color(name).name() == expected


# ── Guards: no second implementation may grow back ────────────────────────


def test_the_editing_core_stays_qt_free():
    """The static half of the core's Qt-free guarantee.

    The dynamic half (a clean interpreter imports it and PySide6 never
    loads) is owned by ``tests/test_box_editing.py``; this one catches an
    import that a conditional would hide from that run.
    """
    source = (SRC / "polyx" / "core" / "editing.py").read_text(encoding="utf-8")
    assert "PySide6" not in source
    assert not re.search(
        r"^\s*(from|import)\s+(PySide6|PyQt|QtCore|QtGui|QtWidgets)", source, re.MULTILINE
    )


def test_neither_canvas_reintroduces_private_editing_maths(rigs):
    private = (
        "_hit_box",
        "_hit_handle",
        "_handle_points",
        "_handles_for",
        "_push_undo",
        "_snapshot",
        "_undo_stack",
        "_redo_stack",
        "_undo",
        "_redo",
        "_clamp_box",
        "_normalize_box",
        "_normalized_box",
    )
    for rig in rigs:
        for name in private:
            assert not hasattr(rig.canvas, name), f"{type(rig.canvas).__name__} regrew {name}"

    # A geometry helper cannot come back as a module-level function either.
    regrown = re.compile(r"^_(hit|handle_points|snapshot|push_undo|clamp|normalis|normalize)")
    for module in (labeler_canvas, gt_manual):
        functions = [
            name
            for name, value in vars(module).items()
            if inspect.isfunction(value)
            and value.__module__ == module.__name__
            and regrown.match(name)
        ]
        assert functions == [], f"{module.__name__} defines {functions}"


def test_the_canvases_and_the_labeller_panel_hold_no_colour_table():
    def tables(module) -> list[str]:
        return [
            name
            for name, value in vars(module).items()
            if isinstance(value, (list, tuple, dict)) and re.search(r"colou?r", name, re.IGNORECASE)
        ]

    for module in (labeler_canvas, labeler_window, gt_manual):
        assert tables(module) == [], f"{module.__name__} keeps {tables(module)}"


def test_the_minimum_side_has_one_name():
    assert not hasattr(BboxCanvas, "SIDE_MINIMUM_PX")
    offenders = [
        path
        for path in (SRC / "polyx").rglob("*.py")
        if "SIDE_MINIMUM_PX" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
