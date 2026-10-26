"""Interactive canvas to draw and edit YOLO bounding boxes."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from polyx.gui import theme as T

from .state import BBox, LabelerState

# Extended palette: PET, PP, LDPE + extra colors for custom classes
_CLASS_COLORS = [
    T.CLASS_COLOR_HEX.get("PET", "#e3342f"),
    T.CLASS_COLOR_HEX.get("PP", "#ff8c00"),
    T.CLASS_COLOR_HEX.get("LDPE", "#ffd700"),
    "#00b4d8",
    "#48cae4",
    "#0077b6",
    "#023e8a",
    "#7b2d8b",
]


def _cls_color(cls_id: int) -> QColor:
    return QColor(_CLASS_COLORS[cls_id % len(_CLASS_COLORS)])


class BboxCanvas(QWidget):
    """Shows the image and allows drawing/editing YOLO bboxes.

    Shortcuts:
      - Drag (left button)   → draws a new box
      - Left click on a box  → selects it
      - Right click on a box → assigns the active class
      - `Del` / Backspace    → deletes the selected box
      - 1-9                  → changes the active class
      - Space                → marks reviewed and moves on
      - F                    → reframes (fits to the window again)
      - Ctrl+Z / Ctrl+Y      → undo / redo
      - Ctrl+S               → save
      - ← →                  → previous / next image
      - Mouse wheel          → zoom
      - Middle button        → pan
    """

    box_drawn = Signal()
    box_rejected = Signal(float)  # side in px of the discarded box

    # Minimum box side, in image pixels. The smallest particles in the
    # study measure ~8 px, so a 5 px threshold discarded legitimate marks.
    SIDE_MINIMUM_PX = 2.0

    def __init__(self, state: LabelerState):
        super().__init__()
        self.state = state
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 400)

        self._pixmap: QPixmap | None = None
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        # With hundreds of crops, reframing on every image forces redoing the
        # zoom hundreds of times. Keeping it preserves the work pace.
        self.keep_zoom = True
        self._zoom_user = False

        # New box drawing
        self._drawing = False
        self._draw_start: QPointF | None = None  # image coords (pixels)
        self._draw_end: QPointF | None = None

        # Pan
        self._panning = False
        self._pan_start: QPointF | None = None
        self._pan_offset_start = QPointF(0.0, 0.0)

        # Selection
        self._selected_idx: int = -1

        state.annotations_changed.connect(self.update)
        state.image_changed.connect(self._on_image_changed)

    # ── Image loading ────────────────────────────────────────
    def _on_image_changed(self, _idx: int):
        img_path = self.state.current_image
        self._selected_idx = -1
        if img_path is None:
            self._pixmap = None
            self.update()
            return

        pix = QPixmap(str(img_path))
        if pix.isNull():
            try:
                import numpy as np
                from PIL import Image

                arr = np.array(Image.open(str(img_path)).convert("RGB"))
                h, w, c = arr.shape
                qimg = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
            except Exception:  # noqa: BLE001, S110
                pass
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

    def _norm_to_widget_rect(self, cx, cy, w, h) -> QRectF:
        if not self._pixmap:
            return QRectF()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        tl = self._i2w(QPointF((cx - w / 2) * iw, (cy - h / 2) * ih))
        br = self._i2w(QPointF((cx + w / 2) * iw, (cy + h / 2) * ih))
        return QRectF(tl, br)

    def _img_rect_to_norm(self, x, y, w, h):
        """Rect in image pixels → (cx, cy, w, h) normalized."""
        iw, ih = self._pixmap.width(), self._pixmap.height()
        cx = (x + w / 2) / iw
        cy = (y + h / 2) / ih
        return cx, cy, w / iw, h / ih

    # ── Painting ─────────────────────────────────────────────
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(T.BG_SOFT))

        if self._pixmap is None:
            p.setPen(QColor(T.INK3))
            p.setFont(QFont(T.APP_FONT_FAMILY, 12))
            p.drawText(self.rect(), Qt.AlignCenter, "Abre una carpeta de imágenes para comenzar")
            return

        # Image
        iw, ih = self._pixmap.width(), self._pixmap.height()
        dest = QRectF(self._offset.x(), self._offset.y(), iw * self._scale, ih * self._scale)
        p.drawPixmap(dest, self._pixmap, QRectF(self._pixmap.rect()))

        # Saved bboxes
        font = QFont(T.APP_FONT_FAMILY, 8, QFont.Bold)
        p.setFont(font)
        for i, box in enumerate(self.state.current_boxes):
            rect_w = self._norm_to_widget_rect(box.cx, box.cy, box.w, box.h)
            color = _cls_color(box.class_id)
            is_sel = i == self._selected_idx

            fill = QColor(color)
            fill.setAlpha(55 if is_sel else 35)
            p.fillRect(rect_w, fill)

            pen = QPen(color, 2.5 if is_sel else 1.8)
            if is_sel:
                pen.setStyle(Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect_w)

            # Class label
            cls_name = (
                self.state.class_names[box.class_id]
                if box.class_id < len(self.state.class_names)
                else str(box.class_id)
            )
            tag = QRectF(rect_w.x(), rect_w.y() - 17, max(50, len(cls_name) * 7 + 8), 17)
            p.fillRect(tag, color)
            p.setPen(QColor("white"))
            p.drawText(tag, Qt.AlignCenter, cls_name)

        # Box in progress
        if self._drawing and self._draw_start and self._draw_end:
            s_w = self._i2w(self._draw_start)
            e_w = self._i2w(self._draw_end)
            rect_draw = QRectF(s_w, e_w).normalized()
            act_color = _cls_color(self.state.active_class)
            fill_d = QColor(act_color)
            fill_d.setAlpha(55)
            p.fillRect(rect_draw, fill_d)
            p.setPen(QPen(act_color, 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect_draw)

    # ── Mouse events ─────────────────────────────────────────
    def _pos(self, ev) -> QPointF:
        return ev.position() if hasattr(ev, "position") else QPointF(ev.pos())

    def mousePressEvent(self, ev: QMouseEvent):
        if self._pixmap is None:
            return
        pos = self._pos(ev)

        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = pos
            self._pan_offset_start = QPointF(self._offset)
            return

        if ev.button() == Qt.RightButton:
            idx = self._box_at(pos)
            if idx >= 0:
                boxes = list(self.state.current_boxes)
                b = boxes[idx]
                boxes[idx] = BBox(self.state.active_class, b.cx, b.cy, b.w, b.h)
                self.state.set_current_boxes(boxes)
            return

        if ev.button() == Qt.LeftButton:
            idx = self._box_at(pos)
            if idx >= 0:
                self._selected_idx = idx
                self.update()
                return
            # Start drawing a new box
            self._drawing = True
            self._draw_start = self._w2i(pos)
            self._draw_end = self._draw_start
            self._selected_idx = -1
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pos = self._pos(ev)
        if self._panning and self._pan_start is not None:
            delta = pos - self._pan_start
            self._offset = self._pan_offset_start + delta
            self._zoom_user = True
            self.update()
        elif self._drawing:
            self._draw_end = self._w2i(pos)
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        pos = self._pos(ev)  # noqa: F841
        if ev.button() == Qt.MiddleButton:
            self._panning = False
            return
        if ev.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            if self._draw_start and self._draw_end and self._pixmap:
                x1 = min(self._draw_start.x(), self._draw_end.x())
                y1 = min(self._draw_start.y(), self._draw_end.y())
                x2 = max(self._draw_start.x(), self._draw_end.x())
                y2 = max(self._draw_start.y(), self._draw_end.y())
                side = min(x2 - x1, y2 - y1)
                if side < self.SIDE_MINIMUM_PX:
                    # It used to be discarded silently: the operator believed they had
                    # marked the particle and the mark disappeared without warning.
                    if side > 0.3:
                        self.box_rejected.emit(side)
                else:
                    cx, cy, w, h = self._img_rect_to_norm(x1, y1, x2 - x1, y2 - y1)
                    # Clamp to [0,1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.001, min(1.0, w))
                    h = max(0.001, min(1.0, h))
                    new_box = BBox(self.state.active_class, cx, cy, w, h)
                    boxes = list(self.state.current_boxes) + [new_box]  # noqa: RUF005
                    self.state.set_current_boxes(boxes)
                    self._selected_idx = len(boxes) - 1
                    self.box_drawn.emit()
            self._draw_start = None
            self._draw_end = None
            self.update()

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
            if 0 <= self._selected_idx < len(self.state.current_boxes):
                boxes = list(self.state.current_boxes)
                boxes.pop(self._selected_idx)
                self._selected_idx = -1
                self.state.set_current_boxes(boxes)

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

    # ── Utilities ────────────────────────────────────────────
    def _box_at(self, widget_pos: QPointF) -> int:
        """Index of the box under the cursor (the most recent one has priority)."""
        for i, box in reversed(list(enumerate(self.state.current_boxes))):
            rect_w = self._norm_to_widget_rect(box.cx, box.cy, box.w, box.h)
            if rect_w.contains(widget_pos):
                return i
        return -1
