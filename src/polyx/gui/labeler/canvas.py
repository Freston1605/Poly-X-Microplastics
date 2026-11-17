"""Interactive canvas to draw and edit YOLO bounding boxes.

The shell only translates events, zooms and pans, paints, and reports what
the labeller did. Geometry, hit-testing, the resize handles, selection and
undo belong to :class:`polyx.core.editing.BoxEditor`, one per image, held by
:class:`~polyx.gui.labeler.state.LabelerState`.

The pixels are decoded by :func:`polyx.gui.image_pixmap.qpixmap_from_file`,
the same EXIF-applied frame ``size_image`` measures and the editor works in:
painting, hit-testing and the saved normalisation cannot disagree.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from polyx.core.editing import (
    HANDLE_TOLERANCE_PX,
    BoxEditor,
    EditMode,
    Handle,
    RejectionReason,
    rejection_message,
)
from polyx.core.i18n import _
from polyx.core.labels import class_name_for
from polyx.gui import theme as T
from polyx.gui.image_pixmap import qpixmap_from_file

from .state import LabelerState

# A press that never moves this far is a click (select), not a move.
_DRAG_THRESHOLD_PX = 3.0

_DRAW = "draw"
_MOVE = "move"
_RESIZE = "resize"


class BboxCanvas(QWidget):
    """Shows the image and allows drawing/editing YOLO bboxes.

    Shortcuts:
      - Drag on empty space → draws a new box
      - Drag a box          → moves it
      - Drag a handle       → resizes the selected box
      - Left click on a box → selects it
      - Right click on a box→ assigns the active class
      - `Del` / Backspace   → deletes the selected box
      - 1-9                 → changes the active class
      - Space               → marks reviewed and moves on
      - F                   → reframes (fits to the window again)
      - Ctrl+Z / Ctrl+Y     → undo / redo
      - Ctrl+S              → save
      - ← →                 → previous / next image
      - Mouse wheel         → zoom
      - Middle button       → pan
    """

    box_drawn = Signal()
    box_rejected = Signal(float)  # side in px of the discarded box

    # Painted handle size, in widget pixels. The grab distance is the core's
    # HANDLE_TOLERANCE_PX, scaled to image pixels at the current zoom.
    HANDLE_RADIUS_PX = 4.0

    def __init__(self, state: LabelerState):
        super().__init__()
        self.state = state
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 400)
        self.setCursor(Qt.CrossCursor)

        self._pixmap: QPixmap | None = None
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        # With hundreds of crops, reframing on every image forces redoing the
        # zoom hundreds of times. Keeping it preserves the work pace.
        self.keep_zoom = True
        self._zoom_user = False

        # The gesture the shell is translating; the editor holds the box.
        self._gesture: str | None = None
        # A press on a box does not move it until the pointer actually moves.
        self._move_armed = False
        self._move_origin = QPointF(0.0, 0.0)  # widget coords
        self._move_image_origin = QPointF(0.0, 0.0)  # image coords

        # Pan
        self._panning = False
        self._pan_start: QPointF | None = None
        self._pan_offset_start = QPointF(0.0, 0.0)

        state.annotations_changed.connect(self.update)
        state.image_changed.connect(self._on_image_changed)

    # ── Image loading ────────────────────────────────────────
    def _on_image_changed(self, _idx: int):
        self._gesture = None
        self._move_armed = False
        self._panning = False
        editor = self.state.current_editor
        if editor is not None:
            editor.select(-1)  # the selection does not travel between images
        img_path = self.state.current_image
        if img_path is None:
            self._pixmap = None
            self.update()
            return

        pix = qpixmap_from_file(img_path)
        previous = self._pixmap
        self._pixmap = pix if not pix.isNull() else None
        # Keeps zoom and framing if the user adjusted them and the image is of
        # equivalent size. With tolerance: when split into a grid the crops
        # differ by 1 px due to rounding (815x1630 versus 814x1629), and demanding
        # exact equality made the zoom get lost almost always.
        same_size = False
        if previous is not None and self._pixmap is not None:
            da = abs(previous.width() - self._pixmap.width())
            db = abs(previous.height() - self._pixmap.height())
            same_size = da <= max(4, previous.width() * 0.02) and db <= max(
                4, previous.height() * 0.02
            )
        if not (self.keep_zoom and self._zoom_user and same_size):
            self._fit_to_window()
        self.update()

    def _fit_to_window(self):
        if self._pixmap is None:
            return
        W, H = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return
        self._scale = min(W / pw, H / ph) * 0.95
        self._offset = QPointF(
            (W - pw * self._scale) / 2,
            (H - ph * self._scale) / 2,
        )

    def resizeEvent(self, ev):
        self._fit_to_window()
        super().resizeEvent(ev)

    # ── Coordinate conversion ────────────────────────────────
    def _w2i(self, pt: QPointF) -> QPointF:
        """Widget → image (pixels)."""
        return QPointF(
            (pt.x() - self._offset.x()) / self._scale,
            (pt.y() - self._offset.y()) / self._scale,
        )

    def _i2w(self, pt: QPointF) -> QPointF:
        """Image → widget."""
        return QPointF(
            pt.x() * self._scale + self._offset.x(),
            pt.y() * self._scale + self._offset.y(),
        )

    def _box_rect(self, x1: float, y1: float, x2: float, y2: float) -> QRectF:
        """A box in image pixels as the widget rectangle that shows it."""
        return QRectF(
            self._i2w(QPointF(x1, y1)),
            self._i2w(QPointF(x2, y2)),
        ).normalized()

    # ── Painting ─────────────────────────────────────────────
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(T.BG_SOFT))

        if self._pixmap is None:
            p.setPen(QColor(T.INK3))
            p.setFont(QFont(T.APP_FONT_FAMILY, 12))
            p.drawText(self.rect(), Qt.AlignCenter, _("Open an image folder to begin"))
            return

        # Image
        iw, ih = self._pixmap.width(), self._pixmap.height()
        dest = QRectF(self._offset.x(), self._offset.y(), iw * self._scale, ih * self._scale)
        p.drawPixmap(dest, self._pixmap, QRectF(self._pixmap.rect()))

        editor = self.state.current_editor
        if editor is None:
            return
        boxes = editor.boxes
        # While drawing, the editor's last box is the provisional one: it is
        # painted dashed, not as a saved mark.
        live = len(boxes) - 1 if editor.mode is EditMode.DRAW and boxes else -1

        font = QFont(T.APP_FONT_FAMILY, 8, QFont.Bold)
        p.setFont(font)
        names = dict(enumerate(self.state.class_names))
        for i, box in enumerate(boxes):
            if i == live:
                continue
            rect_w = self._box_rect(box.x1, box.y1, box.x2, box.y2)
            color = T.class_qcolor_at(self.state.class_names, box.class_id)
            is_sel = i == editor.selection

            fill = QColor(color)
            fill.setAlpha(55 if is_sel else 35)
            p.fillRect(rect_w, fill)

            pen = QPen(color, 2.5 if is_sel else 1.8)
            if is_sel:
                pen.setStyle(Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect_w)

            # Class label: the codec's one class-name lookup.
            cls_name = class_name_for(box.class_id, names)
            tag = QRectF(rect_w.x(), rect_w.y() - 17, max(50, len(cls_name) * 7 + 8), 17)
            p.fillRect(tag, color)
            p.setPen(QColor("white"))
            p.drawText(tag, Qt.AlignCenter, cls_name)

        # Box in progress
        if live >= 0:
            box = boxes[live]
            rect_draw = self._box_rect(box.x1, box.y1, box.x2, box.y2)
            act_color = T.class_qcolor_at(self.state.class_names, box.class_id)
            fill_d = QColor(act_color)
            fill_d.setAlpha(55)
            p.fillRect(rect_draw, fill_d)
            p.setPen(QPen(act_color, 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect_draw)

        selected = editor.selected_box
        if selected is not None and editor.mode is not EditMode.DRAW:
            color = T.class_qcolor_at(self.state.class_names, selected.class_id)
            self._paint_handles(p, editor, color)

    def _paint_handles(self, p: QPainter, editor: BoxEditor, color: QColor):
        box_side = self.HANDLE_RADIUS_PX * 2
        p.setPen(QPen(QColor("white"), 1.2))
        p.setBrush(color)
        for hx, hy in editor.handle_points():
            pt = self._i2w(QPointF(hx, hy))
            p.drawRect(QRectF(pt.x() - box_side / 2, pt.y() - box_side / 2, box_side, box_side))

    # ── Mouse events ─────────────────────────────────────────
    def _pos(self, ev) -> QPointF:
        return ev.position() if hasattr(ev, "position") else QPointF(ev.pos())

    def _handle_tolerance(self) -> float:
        """The widget grab radius, in image pixels at the current zoom."""
        return HANDLE_TOLERANCE_PX / self._scale

    def mousePressEvent(self, ev: QMouseEvent):
        editor = self.state.current_editor
        if self._pixmap is None or editor is None:
            return
        pos = self._pos(ev)

        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = pos
            self._pan_offset_start = QPointF(self._offset)
            return

        img_pt = self._w2i(pos)
        x, y = img_pt.x(), img_pt.y()

        if ev.button() == Qt.RightButton:
            idx = editor.box_at(x, y)
            if idx >= 0 and editor.assign_class(self.state.active_class, idx):
                self.state.notify_edited()
            return

        if ev.button() == Qt.LeftButton:
            if editor.select_at(x, y) < 0:
                # Start drawing a new box.
                editor.begin_draw(x, y)
                self._gesture = _DRAW
            else:
                handle = editor.handle_at(x, y, self._handle_tolerance())
                if handle is not Handle.NONE and editor.begin_resize(handle, x, y):
                    self._gesture = _RESIZE
                else:
                    # A click selects; the move starts only if the pointer moves.
                    self._move_armed = True
                    self._move_origin = pos
                    self._move_image_origin = img_pt
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pos = self._pos(ev)
        if self._panning and self._pan_start is not None:
            delta = pos - self._pan_start
            self._offset = self._pan_offset_start + delta
            self._zoom_user = True
            self.update()
            return

        editor = self.state.current_editor
        if editor is None:
            return
        img_pt = self._w2i(pos)

        if self._gesture == _DRAW:
            editor.update_draw(img_pt.x(), img_pt.y())
            self.update()
        elif self._gesture == _MOVE:
            editor.update_move(img_pt.x(), img_pt.y())
            self.update()
        elif self._gesture == _RESIZE:
            editor.update_resize(img_pt.x(), img_pt.y())
            self.update()
        elif self._move_armed:
            if (pos - self._move_origin).manhattanLength() > _DRAG_THRESHOLD_PX:
                origin = self._move_image_origin
                if editor.begin_move(origin.x(), origin.y()):
                    self._gesture = _MOVE
                    editor.update_move(img_pt.x(), img_pt.y())
                self._move_armed = False
                self.update()
        elif not self._panning:
            self._update_cursor(img_pt)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MiddleButton:
            self._panning = False
            return
        if ev.button() != Qt.LeftButton:
            return
        editor = self.state.current_editor
        gesture, self._gesture = self._gesture, None
        self._move_armed = False
        if editor is None:
            return

        if gesture == _DRAW:
            self._settle(editor.finish_draw(), drawn=True)
        elif gesture == _MOVE:
            self._settle(editor.finish_move())
        elif gesture == _RESIZE:
            self._settle(editor.finish_resize())
        self.update()

    def _settle(self, result, drawn: bool = False):
        """Turn a committed/refused gesture into the canvas's signals."""
        if result.committed:
            self.state.notify_edited()
            if drawn:
                self.box_drawn.emit()
        elif result.reason is RejectionReason.TOO_SMALL:
            # It used to be discarded silently: the operator believed they had
            # marked the particle and the mark disappeared without warning.
            # The core owns the threshold and the wording (silent click).
            if rejection_message(result.side_px) is not None:
                self.box_rejected.emit(result.side_px)

    def _update_cursor(self, img_pt: QPointF):
        editor = self.state.current_editor
        if editor is None:
            return
        handle = editor.handle_at(img_pt.x(), img_pt.y(), self._handle_tolerance())
        if handle is not Handle.NONE:
            self.setCursor(_HANDLE_CURSORS[handle])
        elif editor.box_at(img_pt.x(), img_pt.y()) >= 0:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, ev: QWheelEvent):
        if self._pixmap is None:
            return
        pos = self._pos(ev)
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        img_pt = self._w2i(pos)
        self._scale = max(0.05, min(50.0, self._scale * factor))
        self._offset = QPointF(
            pos.x() - img_pt.x() * self._scale,
            pos.y() - img_pt.y() * self._scale,
        )
        self._zoom_user = True
        self.update()

    def reframe(self):
        """Fits the image to the window again and forgets the manual zoom."""
        self._zoom_user = False
        self._fit_to_window()
        self.update()

    # ── Keyboard ─────────────────────────────────────────────
    def keyPressEvent(self, ev: QKeyEvent):
        key = ev.key()
        mod = ev.modifiers()

        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            editor = self.state.current_editor
            if editor is not None and editor.delete_selected():
                self.state.notify_edited()

        elif Qt.Key_1 <= key <= Qt.Key_9:
            cls = key - Qt.Key_1
            if cls < len(self.state.class_names):
                self.state.set_active_class(cls)

        elif key == Qt.Key_Space:
            # Declares the image reviewed and moves on: it is the gesture for empty plates,
            # which are the majority, and records that they were in fact looked at.
            self.state.mark_reviewed()
            self.state.next_image()
        elif key == Qt.Key_F:
            self.reframe()
        elif key == Qt.Key_Z and mod & Qt.ControlModifier:
            self.state.undo()
        elif key == Qt.Key_Y and mod & Qt.ControlModifier:
            self.state.redo()
        elif key == Qt.Key_S and mod & Qt.ControlModifier:
            self.state.save_current()
        elif key == Qt.Key_Left:
            self.state.prev_image()
        elif key == Qt.Key_Right:
            self.state.next_image()
        else:
            super().keyPressEvent(ev)


_HANDLE_CURSORS = {
    Handle.TOP_LEFT: Qt.SizeFDiagCursor,
    Handle.TOP_RIGHT: Qt.SizeBDiagCursor,
    Handle.BOTTOM_RIGHT: Qt.SizeFDiagCursor,
    Handle.BOTTOM_LEFT: Qt.SizeBDiagCursor,
    Handle.TOP: Qt.SizeVerCursor,
    Handle.RIGHT: Qt.SizeHorCursor,
    Handle.BOTTOM: Qt.SizeVerCursor,
    Handle.LEFT: Qt.SizeHorCursor,
}
