"""The two canvas rigs: one gesture script, either shell.

Both annotation canvases are thin shells over :mod:`polyx.core.editing`, and
these adapters drive them through the real Qt events a user produces (a drag,
a click, a key press). A test asserts what a labeller observes — the boxes,
the selection, the undo state — without reaching into either shell, and one
test can feed the same script to both.

The adapters live here, once, instead of being copy-pasted into every canvas
test module. Each still converts image coordinates with its own widget
transform: that is exactly the shell concern that remains after card 07.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from polyx.gui.detector.pages.gt_manual import AnnotCanvas
from polyx.gui.labeler.canvas import BboxCanvas
from polyx.gui.labeler.state import LabelerState


class CanvasRig:
    """The editing gestures a labeller makes, driven through one shell."""

    canvas: object

    def boxes(self) -> list[tuple[float, float, float, float, int]]: ...

    def selection(self) -> int: ...

    def active_class(self) -> int: ...

    def can_undo(self) -> bool: ...

    def set_class(self, class_id: int) -> None: ...

    def drag(
        self, start: tuple[float, float], end: tuple[float, float], flush: bool = True
    ) -> None: ...

    def click(self, point: tuple[float, float]) -> None: ...

    def key(self, key: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.NoModifier) -> None: ...

    def hover(self, point: tuple[float, float]) -> None:
        """Move the pointer without a button, as a user inspects a handle."""
        ...

    def cursor(self) -> Qt.CursorShape:
        """The cursor the shell is showing now."""
        ...

    def image_scale(self) -> float:
        """Image pixels → widget pixels at the current zoom."""
        ...

    def tag_color(self, name: str) -> QColor:
        """Colour the shell painted the class tag of its newest box."""
        ...


class LabellerRig(CanvasRig):
    """``BboxCanvas`` over a :class:`LabelerState`, as the window wires it."""

    def __init__(self, state: LabelerState, canvas: BboxCanvas):
        self.state = state
        self.canvas = canvas

    @classmethod
    def over(
        cls,
        qapp,
        image_dir: Path,
        class_names: list[str] | None = None,
        size: tuple[int, int] = (800, 600),
    ) -> LabellerRig:
        state = LabelerState()
        if class_names is not None:
            state.set_class_names(list(class_names))
        canvas = BboxCanvas(state)
        canvas.resize(*size)
        canvas.show()
        state.load_images(image_dir)
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

    def active_class(self) -> int:
        return self.state.active_class

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

    def hover(self, point) -> None:
        event = QMouseEvent(
            QEvent.MouseMove,
            self._point(point),
            self.canvas.mapToGlobal(self._point(point)),
            Qt.NoButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
        QApplication.sendEvent(self.canvas, event)
        QApplication.processEvents()

    def cursor(self) -> Qt.CursorShape:
        return self.canvas.cursor().shape()

    def image_scale(self) -> float:
        return self.canvas._scale

    def tag_color(self, name: str) -> QColor:
        """Colour the canvas painted the class tag of its newest box."""
        x1, y1, x2, y2, _cls = self.boxes()[-1]
        rect = self.canvas._box_rect(x1, y1, x2, y2)
        tag = QRectF(rect.x(), rect.y() - 17, max(50, len(name) * 7 + 8), 17)
        image = self.canvas.grab().toImage()
        return image.pixelColor(round(tag.right() - 4), round(tag.y() + 8))


class GtRig(CanvasRig):
    """``AnnotCanvas`` from the Manual GT page, loaded at zoom 100 %."""

    def __init__(self, canvas: AnnotCanvas):
        self.canvas = canvas

    @classmethod
    def over(cls, qapp, image: Path, class_names: list[str] | None = None) -> GtRig:
        canvas = AnnotCanvas()
        canvas.resize(800, 600)
        canvas.show()
        if class_names is not None:
            canvas.set_class_names(list(class_names))
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

    def active_class(self) -> int:
        return self.canvas.active_class

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

    def hover(self, point) -> None:
        self._send(QMouseEvent.MouseMove, self._point(point), Qt.NoButton, Qt.NoButton)
        QApplication.processEvents()

    def cursor(self) -> Qt.CursorShape:
        return self.canvas.cursor().shape()

    def image_scale(self) -> float:
        return self.canvas.transform().m11()

    def tag_color(self, name: str) -> QColor:
        """Colour the scene painted the class tag of its newest box."""
        x1, y1, _x2, _y2, _cls = self.boxes()[-1]
        width = len(name) * 9 + 12
        point = self.canvas.mapFromScene(QPointF(x1 + width - 4, y1 - 18 + 9))
        return self.canvas.viewport().grab().toImage().pixelColor(point)
