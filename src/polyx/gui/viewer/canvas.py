"""Viewer canvas — image + detection overlays + calibration modes."""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
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

from .state import ViewerState, _circumscribed_center, _circumscribed_radius

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


class ViewerCanvas(QWidget):
    """Interactive canvas for the Viewer.

    Modes:
      - `Normal`    : zoom/pan, shows image + detections
      - `Calibración línea`  : 2 clicks → emits state.calib_ready
      - `Calibración círculo`: 3 clicks → emits state.calib_ready
    Shortcuts:
      - Mouse wheel       → zoom
      - Middle button     → pan
      - Esc / right click → cancel calibration
      - ← →               → previous/next image
      - Ctrl+D            → detect (handled in window)
    """

    def __init__(self, state: ViewerState):
        super().__init__()
        self.state = state
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(500, 500)

        self._pixmap: Optional[QPixmap] = None
        self._scale: float = 1.0
        self._offset: QPointF = QPointF(0.0, 0.0)
        self._filter_cls: Optional[str] = None

        # Pan
        self._panning = False
        self._pan_start: Optional[QPointF] = None
        self._pan_offset_start = QPointF(0.0, 0.0)

        # Calibration cursor (current mouse position in calibration mode)
        self._cursor_img: Optional[QPointF] = None

        state.image_loaded.connect(self._on_image_loaded)
        state.detections_changed.connect(self.update)
        state.calib_pts_changed.connect(lambda _: self.update())

    # ── Image ────────────────────────────────────────────────
    def _on_image_loaded(self, path):
        pix = QPixmap(str(path))
        if pix.isNull():
            try:
                import numpy as np
                from PIL import Image

                arr = np.array(Image.open(str(path)).convert("RGB"))
                h, w, c = arr.shape
                qimg = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
            except Exception:
                pass
        self._pixmap = pix if not pix.isNull() else None
        self._fit_to_window()
        self.update()

    def _fit_to_window(self):
        if not self._pixmap:
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

    def set_filter_class(self, cls_name: Optional[str]):
        """None = show all the classes."""
        self._filter_cls = cls_name
        self.update()

    # ── Coordinates ──────────────────────────────────────────
    def _w2i(self, pt: QPointF) -> QPointF:
        return QPointF(
            (pt.x() - self._offset.x()) / self._scale,
            (pt.y() - self._offset.y()) / self._scale,
        )

    def _i2w(self, pt: QPointF) -> QPointF:
        return QPointF(
            pt.x() * self._scale + self._offset.x(),
            pt.y() * self._scale + self._offset.y(),
        )

    def _i2w_pt(self, x: float, y: float) -> QPointF:
        return self._i2w(QPointF(x, y))

    # ── Painting ─────────────────────────────────────────────
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(T.BG_SOFT))

        if self._pixmap is None:
            p.setPen(QColor(T.INK3))
            p.setFont(QFont(T.APP_FONT_FAMILY, 12))
            p.drawText(self.rect(), Qt.AlignCenter, "Abre una imagen o una carpeta para comenzar")
            return

        # Image
        iw, ih = self._pixmap.width(), self._pixmap.height()
        dest = QRectF(self._offset.x(), self._offset.y(), iw * self._scale, ih * self._scale)
        p.drawPixmap(dest, self._pixmap, QRectF(self._pixmap.rect()))

        # Detections
        self._draw_detections(p, iw, ih)

        # Calibration overlay
        self._draw_calib_overlay(p)

    def _draw_detections(self, p: QPainter, iw: int, ih: int):
        font = QFont(T.APP_FONT_FAMILY, 8, QFont.Bold)
        p.setFont(font)
        for i, det in enumerate(self.state.detections):
            if self._filter_cls and det.class_name != self._filter_cls:
                continue
            color = _cls_color(det.class_id)

            tl = self._i2w_pt(det.x1, det.y1)
            br = self._i2w_pt(det.x2, det.y2)
            rect_w = QRectF(tl, br)

            fill = QColor(color)
            fill.setAlpha(40)
            p.fillRect(rect_w, fill)
            p.setPen(QPen(color, 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect_w)

            # Label: class + size when calibration is available
            w_px = det.x2 - det.x1
            h_px = det.y2 - det.y1
            diam_px = math.sqrt(w_px * h_px)
            um = self.state.px_to_um(diam_px)
            if um is not None:
                label = f"{det.class_name}  {um:.1f} μm"
            else:
                label = f"{det.class_name}  {diam_px:.0f} px"

            tag_w = max(70, len(label) * 7 + 8)
            tag = QRectF(tl.x(), tl.y() - 17, tag_w, 17)
            p.fillRect(tag, color)
            p.setPen(QColor("white"))
            p.drawText(tag, Qt.AlignCenter, label)

        p.setPen(QColor(T.INK))

    def _draw_calib_overlay(self, p: QPainter):
        pts = self.state.calib_pts
        mode = self.state.calib_mode
        if mode == "none" or not pts:
            return

        accent = QColor(T.ACCENT)
        dot_pen = QPen(accent, 2)
        line_pen = QPen(accent, 1.5, Qt.DashLine)

        # Points already placed
        w_pts = [self._i2w(pt) for pt in pts]
        for wp in w_pts:
            p.setPen(dot_pen)
            p.setBrush(QBrush(accent))
            p.drawEllipse(wp, 5, 5)

        if mode == "linea":
            if len(w_pts) >= 2:
                p.setPen(line_pen)
                p.drawLine(w_pts[0], w_pts[1])
                # Distance in px
                dx = pts[1].x() - pts[0].x()
                dy = pts[1].y() - pts[0].y()
                dist = math.hypot(dx, dy)
                mid = QPointF(
                    (w_pts[0].x() + w_pts[1].x()) / 2, (w_pts[0].y() + w_pts[1].y()) / 2 - 10
                )
                p.setPen(QPen(accent))
                p.setFont(QFont(T.APP_FONT_FAMILY, 9, QFont.Bold))
                p.drawText(mid, f"{dist:.1f} px")
            elif len(w_pts) == 1 and self._cursor_img:
                # Provisional line to the cursor
                cw = self._i2w(self._cursor_img)
                p.setPen(line_pen)
                p.drawLine(w_pts[0], cw)

        elif mode == "circulo":
            if len(w_pts) >= 2:
                # Lines between points
                p.setPen(line_pen)
                for i in range(len(w_pts)):
                    p.drawLine(w_pts[i], w_pts[(i + 1) % len(w_pts)])

            if len(pts) >= 3:
                # Circumscribed circle
                center_img = _circumscribed_center(pts)
                radius_img = _circumscribed_radius(pts)
                cw = self._i2w(center_img)
                r_w = radius_img * self._scale
                circ_color = QColor(T.ACCENT)
                circ_color.setAlpha(180)
                p.setPen(QPen(circ_color, 1.5, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(cw, r_w, r_w)
                # Center
                p.setPen(QPen(accent, 1))
                p.setBrush(Qt.NoBrush)
                p.drawLine(QPointF(cw.x() - 6, cw.y()), QPointF(cw.x() + 6, cw.y()))
                p.drawLine(QPointF(cw.x(), cw.y() - 6), QPointF(cw.x(), cw.y() + 6))

    # ── Mouse ────────────────────────────────────────────────
    def _pos(self, ev) -> QPointF:
        return ev.position() if hasattr(ev, "position") else QPointF(ev.pos())

    def mousePressEvent(self, ev: QMouseEvent):
        pos = self._pos(ev)

        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = pos
            self._pan_offset_start = QPointF(self._offset)
            return

        if ev.button() == Qt.RightButton:
            if self.state.calib_mode != "none":
                self.state.cancel_calib()
            return

        if ev.button() == Qt.LeftButton:
            if self.state.calib_mode != "none":
                img_pt = self._w2i(pos)
                self.state.add_calib_point(img_pt)
            # In normal mode the canvas is inspection-only (it does not edit)

    def mouseMoveEvent(self, ev: QMouseEvent):
        pos = self._pos(ev)
        if self._panning and self._pan_start is not None:
            delta = pos - self._pan_start
            self._offset = self._pan_offset_start + delta
            self.update()
        elif self.state.calib_mode != "none":
            self._cursor_img = self._w2i(pos)
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MiddleButton:
            self._panning = False

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
        self.update()

    # ── Keyboard ─────────────────────────────────────────────
    def keyPressEvent(self, ev: QKeyEvent):
        key = ev.key()
        mod = ev.modifiers()
        if key == Qt.Key_Escape:
            self.state.cancel_calib()
        elif key == Qt.Key_Left:
            self.state.prev_image()
        elif key == Qt.Key_Right:
            self.state.next_image()
        else:
            super().keyPressEvent(ev)
