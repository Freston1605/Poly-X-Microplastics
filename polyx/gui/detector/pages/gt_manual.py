"""Página 3 — GT manual.

Anotador completo de bounding boxes con:
- Zoom con rueda del mouse
- Pan con botón medio o Espacio+arrastre
- Dibujar cajas con clic-izquierdo+arrastre
- Seleccionar haciendo clic en una caja
- Mover caja seleccionada (arrastre)
- Redimensionar con handles en las esquinas/lados
- Eliminar con Supr / Del
- Atajos 1..9 para cambiar clase activa (también en caja seleccionada)
- Undo / Redo (Ctrl+Z, Ctrl+Y)
- Navegación con flechas izq/der (autoguarda)
- Exporta .txt YOLO + classes.txt
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import copy
import json

import numpy as np
import cv2

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSize
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QBrush, QImage, QFont, QCursor,
    QKeySequence, QShortcut, QAction,
)
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox, QListWidget,
    QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsSimpleTextItem, QMessageBox, QFrame, QSizePolicy,
)

from ._base import DetectorPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.model.yolo_wrap import (
    Detection, read_yolo_txt, find_gt_for_image, tamano_imagen,
)
from polyx.model.i18n import tr


def _qpixmap_de(path: Path) -> QPixmap:
    """Decodifica con cv2, no con el lector de Qt.

    El anotador y el detector tienen que trabajar en el MISMO marco de pixeles.
    Qt aplica la rotacion EXIF al abrir un JPEG y cv2 no, asi que una foto
    marcada como girada se anotaba en un marco y se leia en otro: las cajas
    guardadas salian desplazadas -- alguna incluso fuera de la placa -- aunque
    en el anotador se vieran perfectamente puestas. Y el .txt no tiene forma de
    declarar en que marco se escribio.

    De paso, cv2.imdecode sobre np.fromfile es lo unico que abre con fiabilidad
    rutas con acentos en Windows, que es la razon por la que el resto del
    proyecto ya lee asi.
    """
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return QPixmap()
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # .copy() para que la QImage sea duena de sus bytes: el array de numpy se
    # libera al salir y sin la copia quedaria apuntando a memoria muerta.
    return QPixmap.fromImage(
        QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())

DEFAULT_CLASSES = ["PET", "PP", "LDPE"]

# Modos del canvas
MODE_IDLE = 0
MODE_DRAW = 1
MODE_MOVE = 2
MODE_RESIZE = 3
MODE_PAN = 4

# Handles de redimensionar (orden importa para los cursores)
HANDLE_NONE = -1
HANDLE_TL = 0   # top-left
HANDLE_TR = 1
HANDLE_BR = 2
HANDLE_BL = 3
HANDLE_T  = 4   # top
HANDLE_R  = 5
HANDLE_B  = 6
HANDLE_L  = 7

HANDLE_SIZE = 8   # px en pantalla


# ────────────────────────────────────────────────────────────────────
@dataclass
class Box:
    """Caja de anotación en coordenadas absolutas de imagen."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0

    def normalize(self):
        if self.x1 > self.x2: self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2: self.y1, self.y2 = self.y2, self.y1

    def rect(self) -> QRectF:
        return QRectF(self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)


# ────────────────────────────────────────────────────────────────────
class AnnotCanvas(QGraphicsView):
    """Canvas avanzado: zoom, pan, dibujar/mover/redimensionar cajas."""
    boxes_changed = Signal()
    selection_changed = Signal(int)   # idx o -1
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(
            f"QGraphicsView {{ border: 1px solid {T.RULE}; background: #1a1d22; }}"
        )
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._pixmap_item = None
        self._image_size = (0, 0)
        self._image_path: Optional[Path] = None
        self.boxes: List[Box] = []
        self.selected_idx: int = -1
        self.active_class: int = 0
        self.class_names: List[str] = list(DEFAULT_CLASSES)

        # Estado de interacción
        self._mode = MODE_IDLE
        self._mouse_press_scene: Optional[QPointF] = None
        self._mouse_press_box: Optional[Box] = None
        self._drag_handle: int = HANDLE_NONE
        self._space_pressed = False
        self._last_pan_pos = None

        # Historial undo/redo
        self._undo: list[list[Box]] = []
        self._redo: list[list[Box]] = []

    # ── API pública ───────────────────────────────────────────────
    def set_class_names(self, names: List[str]):
        self.class_names = list(names)
        if self.active_class >= len(self.class_names):
            self.active_class = 0
        self._refresh()

    def set_active_class(self, idx: int):
        if 0 <= idx < len(self.class_names):
            self.active_class = idx
            # si hay caja seleccionada, cambiar su clase
            if 0 <= self.selected_idx < len(self.boxes):
                self._push_undo()
                self.boxes[self.selected_idx].class_id = idx
                self._refresh()
                self.boxes_changed.emit()

    def load_image(self, path: Path, existing: List[Detection]):
        self._image_path = path
        pm = _qpixmap_de(path)
        if pm.isNull():
            self.scene_.clear()
            self._pixmap_item = None
            self.boxes = []
            self.selected_idx = -1
            self.status_changed.emit("No se pudo abrir la imagen.")
            return
        self._image_size = (pm.width(), pm.height())
        self.scene_.clear()
        self._pixmap_item = self.scene_.addPixmap(pm)
        self.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))
        # Convertir Detection -> Box
        self.boxes = [
            Box(d.x1, d.y1, d.x2, d.y2, class_id=d.class_id)
            for d in existing
        ]
        self.selected_idx = -1
        self._undo.clear(); self._redo.clear()
        # Fit inicial
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._refresh()
        self.boxes_changed.emit()

    def detections(self) -> List[Detection]:
        out = []
        for b in self.boxes:
            name = self.class_names[b.class_id] if b.class_id < len(self.class_names) else str(b.class_id)
            out.append(Detection(
                class_id=b.class_id, class_name=name, conf=1.0,
                x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
            ))
        return out

    def clear_boxes(self):
        if not self.boxes: return
        self._push_undo()
        self.boxes = []
        self.selected_idx = -1
        self._refresh()
        self.boxes_changed.emit()

    def delete_selected(self):
        if not (0 <= self.selected_idx < len(self.boxes)): return
        self._push_undo()
        del self.boxes[self.selected_idx]
        self.selected_idx = -1
        self._refresh()
        self.boxes_changed.emit()
        self.selection_changed.emit(-1)

    def undo(self):
        if not self._undo: return
        self._redo.append(self._snapshot())
        self.boxes = self._undo.pop()
        if self.selected_idx >= len(self.boxes):
            self.selected_idx = -1
        self._refresh()
        self.boxes_changed.emit()

    def redo(self):
        if not self._redo: return
        self._undo.append(self._snapshot())
        self.boxes = self._redo.pop()
        if self.selected_idx >= len(self.boxes):
            self.selected_idx = -1
        self._refresh()
        self.boxes_changed.emit()

    def zoom_fit(self):
        if self._pixmap_item is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def zoom_100(self):
        self.resetTransform()

    # ── helpers internos ───────────────────────────────────────────
    def _snapshot(self) -> list[Box]:
        return [copy.copy(b) for b in self.boxes]

    def _push_undo(self):
        self._undo.append(self._snapshot())
        if len(self._undo) > 100:
            self._undo.pop(0)
        self._redo.clear()

    def _class_color(self, cid: int) -> QColor:
        name = self.class_names[cid] if cid < len(self.class_names) else ""
        hex_ = T.CLASS_COLOR_HEX.get(name, "#33aaff")
        return QColor(hex_)

    def _refresh(self):
        """Redibuja todas las cajas + handles + labels (sin tocar la imagen)."""
        # Quitar todos los items menos el pixmap
        for it in list(self.scene_.items()):
            if it is self._pixmap_item:
                continue
            self.scene_.removeItem(it)

        for i, b in enumerate(self.boxes):
            color = self._class_color(b.class_id)
            sel = (i == self.selected_idx)
            pen = QPen(color, 3 if sel else 2)
            if sel:
                pen.setStyle(Qt.SolidLine)
            rect = QGraphicsRectItem(b.rect())
            rect.setPen(pen)
            fill = QColor(color); fill.setAlpha(45 if sel else 0)
            rect.setBrush(QBrush(fill))
            self.scene_.addItem(rect)

            # Label
            name = self.class_names[b.class_id] if b.class_id < len(self.class_names) else str(b.class_id)
            label_bg = QGraphicsRectItem(QRectF(b.x1, b.y1 - 18, len(name) * 9 + 12, 18))
            label_bg.setPen(QPen(color, 1))
            label_bg.setBrush(QBrush(color))
            self.scene_.addItem(label_bg)
            txt = QGraphicsSimpleTextItem(name)
            txt.setBrush(QBrush(Qt.white))
            f = QFont(T.APP_FONT_FAMILY, 9, QFont.Bold)
            txt.setFont(f)
            txt.setPos(b.x1 + 4, b.y1 - 17)
            self.scene_.addItem(txt)

            if sel:
                # Handles 8 puntos
                hs = HANDLE_SIZE / max(self.transform().m11(), 0.0001)  # tamaño en escena
                points = self._handle_points(b, hs)
                for hidx, (hx, hy) in enumerate(points):
                    h = QGraphicsRectItem(QRectF(hx - hs/2, hy - hs/2, hs, hs))
                    h.setPen(QPen(Qt.white, 1.2))
                    h.setBrush(QBrush(color))
                    h.setZValue(10)
                    self.scene_.addItem(h)

    def _handle_points(self, b: Box, hs: float) -> list[tuple[float, float]]:
        cx = (b.x1 + b.x2) / 2
        cy = (b.y1 + b.y2) / 2
        return [
            (b.x1, b.y1),  # TL
            (b.x2, b.y1),  # TR
            (b.x2, b.y2),  # BR
            (b.x1, b.y2),  # BL
            (cx,   b.y1),  # T
            (b.x2, cy  ),  # R
            (cx,   b.y2),  # B
            (b.x1, cy  ),  # L
        ]

    def _hit_handle(self, b: Box, pos: QPointF) -> int:
        hs = HANDLE_SIZE / max(self.transform().m11(), 0.0001)
        pts = self._handle_points(b, hs)
        for hidx, (hx, hy) in enumerate(pts):
            if abs(pos.x() - hx) <= hs and abs(pos.y() - hy) <= hs:
                return hidx
        return HANDLE_NONE

    def _hit_box(self, pos: QPointF) -> int:
        """Devuelve idx de la primera caja que contiene `pos`, -1 si ninguna.
        Prefiere la seleccionada si está bajo el cursor."""
        # primero la actualmente seleccionada (preferencia)
        if 0 <= self.selected_idx < len(self.boxes):
            if self.boxes[self.selected_idx].rect().contains(pos):
                return self.selected_idx
        for i, b in enumerate(self.boxes):
            if b.rect().contains(pos):
                return i
        return -1

    def _set_cursor_for_handle(self, h: int):
        cursors = {
            HANDLE_TL: Qt.SizeFDiagCursor, HANDLE_BR: Qt.SizeFDiagCursor,
            HANDLE_TR: Qt.SizeBDiagCursor, HANDLE_BL: Qt.SizeBDiagCursor,
            HANDLE_T : Qt.SizeVerCursor,   HANDLE_B : Qt.SizeVerCursor,
            HANDLE_L : Qt.SizeHorCursor,   HANDLE_R : Qt.SizeHorCursor,
        }
        self.setCursor(cursors.get(h, Qt.CrossCursor))

    # ── eventos ────────────────────────────────────────────────────
    def wheelEvent(self, ev):
        if self._pixmap_item is None:
            return
        delta = ev.angleDelta().y()
        factor = 1.20 if delta > 0 else 1 / 1.20
        # límites
        cur = self.transform().m11()
        if (factor > 1 and cur < 40) or (factor < 1 and cur > 0.05):
            self.scale(factor, factor)
            self._refresh()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Space:
            self._space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif k == Qt.Key_Escape:
            self.selected_idx = -1
            self._refresh()
            self.selection_changed.emit(-1)
        elif Qt.Key_1 <= k <= Qt.Key_9:
            idx = k - Qt.Key_1
            self.set_active_class(idx)
        elif ev.matches(QKeySequence.Undo):
            self.undo()
        elif ev.matches(QKeySequence.Redo):
            self.redo()
        elif k == Qt.Key_F:
            self.zoom_fit()
        else:
            super().keyPressEvent(ev); return
        ev.accept()

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Space:
            self._space_pressed = False
            self.setCursor(Qt.CrossCursor)
        super().keyReleaseEvent(ev)

    def mousePressEvent(self, ev):
        if self._pixmap_item is None:
            super().mousePressEvent(ev); return
        scene_pos = self.mapToScene(ev.position().toPoint())
        self._mouse_press_scene = scene_pos

        # Pan: rueda media o espacio+izq
        if ev.button() == Qt.MiddleButton or (self._space_pressed and ev.button() == Qt.LeftButton):
            self._mode = MODE_PAN
            self._last_pan_pos = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept(); return

        if ev.button() == Qt.LeftButton:
            # ¿Estoy sobre un handle de la seleccionada?
            if 0 <= self.selected_idx < len(self.boxes):
                h = self._hit_handle(self.boxes[self.selected_idx], scene_pos)
                if h != HANDLE_NONE:
                    self._mode = MODE_RESIZE
                    self._drag_handle = h
                    self._mouse_press_box = copy.copy(self.boxes[self.selected_idx])
                    self._push_undo()
                    self._set_cursor_for_handle(h)
                    ev.accept(); return

            # ¿Estoy sobre una caja?
            hit = self._hit_box(scene_pos)
            if hit >= 0:
                self.selected_idx = hit
                self._mode = MODE_MOVE
                self._mouse_press_box = copy.copy(self.boxes[hit])
                self._push_undo()
                self.setCursor(Qt.ClosedHandCursor)
                self._refresh()
                self.selection_changed.emit(self.selected_idx)
                ev.accept(); return

            # Sino, dibujar nueva
            self._mode = MODE_DRAW
            self.selected_idx = -1
            self._push_undo()
            new_box = Box(scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y(),
                          class_id=self.active_class)
            self.boxes.append(new_box)
            self.selected_idx = len(self.boxes) - 1
            self._refresh()
            ev.accept(); return

        if ev.button() == Qt.RightButton:
            # asignar clase activa a caja bajo cursor
            hit = self._hit_box(scene_pos)
            if hit >= 0:
                self._push_undo()
                self.boxes[hit].class_id = self.active_class
                self.selected_idx = hit
                self._refresh()
                self.boxes_changed.emit()
                self.selection_changed.emit(hit)
            ev.accept(); return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pixmap_item is None:
            super().mouseMoveEvent(ev); return
        scene_pos = self.mapToScene(ev.position().toPoint())

        if self._mode == MODE_PAN and self._last_pan_pos is not None:
            delta = ev.position() - self._last_pan_pos
            self._last_pan_pos = ev.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            ev.accept(); return

        if self._mode == MODE_DRAW and self.selected_idx >= 0:
            b = self.boxes[self.selected_idx]
            b.x2 = scene_pos.x()
            b.y2 = scene_pos.y()
            self._refresh()
            self.status_changed.emit(
                f"Dibujando: {abs(b.x2-b.x1):.0f}×{abs(b.y2-b.y1):.0f} px"
            )
            ev.accept(); return

        if self._mode == MODE_MOVE and self._mouse_press_box and 0 <= self.selected_idx < len(self.boxes):
            dx = scene_pos.x() - self._mouse_press_scene.x()
            dy = scene_pos.y() - self._mouse_press_scene.y()
            base = self._mouse_press_box
            b = self.boxes[self.selected_idx]
            b.x1, b.x2 = base.x1 + dx, base.x2 + dx
            b.y1, b.y2 = base.y1 + dy, base.y2 + dy
            self._refresh()
            ev.accept(); return

        if self._mode == MODE_RESIZE and self._mouse_press_box and 0 <= self.selected_idx < len(self.boxes):
            base = self._mouse_press_box
            b = self.boxes[self.selected_idx]
            mx, my = scene_pos.x(), scene_pos.y()
            x1, y1, x2, y2 = base.x1, base.y1, base.x2, base.y2
            h = self._drag_handle
            if h in (HANDLE_TL, HANDLE_T, HANDLE_TR): y1 = my
            if h in (HANDLE_TR, HANDLE_R, HANDLE_BR): x2 = mx
            if h in (HANDLE_BR, HANDLE_B, HANDLE_BL): y2 = my
            if h in (HANDLE_BL, HANDLE_L, HANDLE_TL): x1 = mx
            b.x1, b.y1, b.x2, b.y2 = x1, y1, x2, y2
            self._refresh()
            ev.accept(); return

        # Sin botón: actualizar cursor según hover
        if 0 <= self.selected_idx < len(self.boxes):
            h = self._hit_handle(self.boxes[self.selected_idx], scene_pos)
            if h != HANDLE_NONE:
                self._set_cursor_for_handle(h)
                ev.accept(); return
        # Hover sobre caja: cursor "mano"
        if self._hit_box(scene_pos) >= 0:
            self.setCursor(Qt.OpenHandCursor)
        elif self._space_pressed:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._mode == MODE_DRAW and 0 <= self.selected_idx < len(self.boxes):
            b = self.boxes[self.selected_idx]
            b.normalize()
            # Descarta cajas mini
            if (b.x2 - b.x1) < 4 or (b.y2 - b.y1) < 4:
                del self.boxes[self.selected_idx]
                self.selected_idx = -1
                # también el undo agregado, podemos dejarlo (no daña)
            else:
                # Clip a la imagen
                W, H = self._image_size
                b.x1 = max(0, min(W, b.x1)); b.x2 = max(0, min(W, b.x2))
                b.y1 = max(0, min(H, b.y1)); b.y2 = max(0, min(H, b.y2))
            self._refresh()
            self.boxes_changed.emit()
            self.selection_changed.emit(self.selected_idx)

        elif self._mode in (MODE_MOVE, MODE_RESIZE) and 0 <= self.selected_idx < len(self.boxes):
            b = self.boxes[self.selected_idx]
            b.normalize()
            W, H = self._image_size
            b.x1 = max(0, min(W, b.x1)); b.x2 = max(0, min(W, b.x2))
            b.y1 = max(0, min(H, b.y1)); b.y2 = max(0, min(H, b.y2))
            self._refresh()
            self.boxes_changed.emit()

        self._mode = MODE_IDLE
        self._mouse_press_scene = None
        self._mouse_press_box = None
        self._drag_handle = HANDLE_NONE
        self._last_pan_pos = None
        self.setCursor(Qt.CrossCursor)
        super().mouseReleaseEvent(ev)


# ────────────────────────────────────────────────────────────────────
class GTManualPage(DetectorPage):
    STEP_N = 3
    STEP_TITLE = tr("GT manual")
    STEP_DESCRIPTION = (
        tr("Anotador completo de Ground Truth. Click izquierdo para dibujar, click sobre una "
        "caja para seleccionar, arrastra los handles para redimensionar, Espacio o botón "
        "medio para pan, rueda para zoom. Teclas 1..9: clase activa · Supr: borrar · "
        "Ctrl+Z/Y: undo/redo · ←/→: imagen anterior/siguiente (autoguarda).")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Tarjeta principal ──
        c1, l1 = self.card(tr("Anotación"), "editar")

        # Barra superior: nombre + clase activa + zoom
        top = QHBoxLayout()
        top.setSpacing(8)
        self.lbl_image_name = QLabel(tr("(sin imagen seleccionada)"))
        self.lbl_image_name.setStyleSheet(
            f"color: {T.INK2}; font-weight: 600; font-size: 11pt; border: none;"
        )
        top.addWidget(self.lbl_image_name, 1)

        top.addWidget(QLabel(tr("Clase:")))
        self.combo_class = QComboBox()
        self.combo_class.addItems(DEFAULT_CLASSES)
        self.combo_class.currentIndexChanged.connect(self._on_class_combo)
        self.combo_class.setMinimumWidth(120)
        top.addWidget(self.combo_class)

        btn_fit = QPushButton(tr("Ajustar (F)"))
        btn_fit.clicked.connect(lambda: self.canvas.zoom_fit())
        top.addWidget(btn_fit)
        btn_100 = QPushButton("100 %")
        btn_100.clicked.connect(lambda: self.canvas.zoom_100())
        top.addWidget(btn_100)
        l1.addLayout(top)

        # Cuerpo: lista de imágenes + canvas
        body = QHBoxLayout()
        body.setSpacing(12)

        # Lista de imágenes (compacta)
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        lbl_files = QLabel(tr("Imágenes"))
        lbl_files.setStyleSheet(f"color: {T.INK2}; font-weight: 600; border: none;")
        left_col.addWidget(lbl_files)
        self.list = QListWidget()
        self.list.setMinimumWidth(260)
        self.list.setMaximumWidth(320)
        self.list.itemSelectionChanged.connect(self._on_list_change)
        left_col.addWidget(self.list, 1)
        body.addLayout(left_col)

        # Canvas
        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        self.canvas = AnnotCanvas()
        self.canvas.set_class_names(DEFAULT_CLASSES)
        self.canvas.setMinimumSize(QSize(560, 420))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_col.addWidget(self.canvas, 1)

        # Mini-status del canvas
        self.lbl_canvas_status = QLabel(tr("Listo. Click y arrastra para dibujar."))
        self.lbl_canvas_status.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        right_col.addWidget(self.lbl_canvas_status)

        body.addLayout(right_col, 1)
        l1.addLayout(body)

        # Botonera inferior
        btns = QHBoxLayout()
        btns.setSpacing(8)
        b_save = QPushButton(tr("Guardar GT (.txt YOLO)"))
        b_save.setIcon(iconos.icono("guardar", 15, T.ON_ACCENT))
        b_save.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        b_save.setCursor(Qt.PointingHandCursor)
        b_save.clicked.connect(self._save_current)
        btns.addWidget(b_save)

        b_prev = QPushButton(tr("←  Anterior"))
        b_prev.clicked.connect(self._go_prev)
        btns.addWidget(b_prev)
        b_next = QPushButton(tr("Siguiente  →"))
        b_next.clicked.connect(self._go_next)
        btns.addWidget(b_next)

        b_undo = QPushButton(tr("Deshacer"))
        b_undo.setIcon(iconos.icono("deshacer", 15, T.INK2))
        b_undo.clicked.connect(lambda: self.canvas.undo())
        btns.addWidget(b_undo)
        b_redo = QPushButton(tr("Rehacer"))
        b_redo.setIcon(iconos.icono("rehacer", 15, T.INK2))
        b_redo.clicked.connect(lambda: self.canvas.redo())
        btns.addWidget(b_redo)

        b_del = QPushButton(tr("Borrar selección"))
        b_del.setIcon(iconos.icono("papelera", 15, T.INK2))
        b_del.clicked.connect(lambda: self.canvas.delete_selected())
        btns.addWidget(b_del)
        b_clear = QPushButton(tr("Limpiar todo"))
        b_clear.clicked.connect(self._clear_with_confirm)
        btns.addWidget(b_clear)
        btns.addStretch(1)

        self.lbl_count = QLabel(tr("0 cajas"))
        self.lbl_count.setStyleSheet(f"color: {T.INK3}; border: none;")
        btns.addWidget(self.lbl_count)
        l1.addLayout(btns)

        self.body.addWidget(c1)

        # Atajos de teclado globales para navegación con flechas
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._go_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._go_next)

        # Suscripciones
        self.state.images_changed.connect(self._refresh_list)
        self.canvas.boxes_changed.connect(self._on_boxes_changed)
        self.canvas.status_changed.connect(self.lbl_canvas_status.setText)
        self._refresh_list()

    # ── Lista de imágenes ──────────────────────────────────────────
    def _refresh_list(self):
        # Conservar selección si existe
        prev = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.state.images:
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            gt = find_gt_for_image(p, self.state.gt_folder)
            if gt:
                it.setForeground(QColor(T.OK))
                it.setText(f"✓ {p.name}")
            self.list.addItem(it)
        self.list.blockSignals(False)
        if 0 <= prev < self.list.count():
            self.list.setCurrentRow(prev)
        elif self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _go_prev(self):
        if self.list.count() == 0: return
        r = max(0, self.list.currentRow() - 1)
        self.list.setCurrentRow(r)
        self.canvas.setFocus()

    def _go_next(self):
        if self.list.count() == 0: return
        r = min(self.list.count() - 1, self.list.currentRow() + 1)
        self.list.setCurrentRow(r)
        self.canvas.setFocus()

    def _on_class_combo(self, idx: int):
        self.canvas.set_active_class(idx)
        self.canvas.setFocus()

    def _on_list_change(self):
        # Auto-guardar la imagen anterior si tiene cajas
        # (lo hacemos antes de cargar la nueva)
        if hasattr(self, "_current_path") and self._current_path is not None:
            self._auto_save(self._current_path, silent=True)

        it = self.list.currentItem()
        if not it:
            return
        path = Path(it.data(Qt.UserRole))
        self._current_path = path
        self.lbl_image_name.setText(path.name)
        existing: List[Detection] = []
        gt_txt = find_gt_for_image(path, self.state.gt_folder)
        if gt_txt:
            # Mismas dimensiones que usa el detector. PIL lee la cabecera sin
            # aplicar la rotacion EXIF, igual que cv2; QImage si la aplicaba.
            wh = tamano_imagen(path)
            if wh:
                existing = read_yolo_txt(
                    gt_txt, wh[0], wh[1],
                    {i: n for i, n in enumerate(DEFAULT_CLASSES)},
                )
        self.canvas.load_image(path, existing)
        self._on_boxes_changed()
        self.canvas.setFocus()

    def _on_boxes_changed(self):
        n = len(self.canvas.boxes)
        self.lbl_count.setText(f"{n} caja{'s' if n != 1 else ''}")

    def _clear_with_confirm(self):
        if not self.canvas.boxes:
            return
        r = QMessageBox.question(
            self, tr("Limpiar todo"),
            f"¿Borrar las {len(self.canvas.boxes)} cajas de esta imagen?\n"
            "(se puede deshacer con Ctrl+Z)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if r == QMessageBox.Yes:
            self.canvas.clear_boxes()

    # ── Guardado ────────────────────────────────────────────────
    def _gt_out_path(self, img_path: Path) -> tuple[Path, Path]:
        """Devuelve (carpeta_destino, archivo_destino) según convenciones YOLO."""
        if self.state.gt_folder:
            out_dir = self.state.gt_folder
        elif img_path.parent.name.lower() == "images":
            out_dir = img_path.parent.parent / "labels"
        else:
            out_dir = img_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, out_dir / f"{img_path.stem}.txt"

    def _save_to_txt(self, img_path: Path, dets: List[Detection]):
        out_dir, out_path = self._gt_out_path(img_path)
        wh = tamano_imagen(img_path)
        if not wh or wh[0] <= 0 or wh[1] <= 0:
            raise ValueError(f"No se pudo leer el tamano de {img_path.name}")
        W, H = wh
        lines = []
        for d in dets:
            cx = (d.x1 + d.x2) / 2 / W
            cy = (d.y1 + d.y2) / 2 / H
            w  = (d.x2 - d.x1) / W
            h  = (d.y2 - d.y1) / H
            lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        out_path.write_text("\n".join(lines), encoding="utf-8")

        # classes.txt junto a labels o a la carpeta
        if out_dir.name.lower() == "labels":
            cls_path = out_dir.parent / "classes.txt"
        else:
            cls_path = out_dir / "classes.txt"
        if not cls_path.exists():
            cls_path.write_text("\n".join(DEFAULT_CLASSES), encoding="utf-8")
        return out_path

    def _auto_save(self, path: Path, silent: bool = False):
        dets = self.canvas.detections()
        if not dets:
            return
        try:
            self._save_to_txt(path, dets)
            if not silent:
                self.lbl_canvas_status.setText(f"Auto-guardado.")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, tr("Error guardando"), str(e))

    def _save_current(self):
        it = self.list.currentItem()
        if not it:
            QMessageBox.information(self, tr("GT manual"), tr("Selecciona primero una imagen."))
            return
        path = Path(it.data(Qt.UserRole))
        dets = self.canvas.detections()
        if not dets:
            # Guardar cero cajas es un dato, no un error. Antes se rechazaba y el
            # .txt anterior sobrevivia intacto: borrar todas las cajas mal puestas
            # de una placa no tenia ningun efecto, y el informe seguia mostrando
            # la anotacion vieja como si nada. Ademas el detector solo cuenta los
            # falsos positivos de una placa vacia si su .txt existe.
            r = QMessageBox.question(
                self, tr("GT manual"),
                tr("No hay ninguna caja dibujada. ¿Guardar esta imagen como "
                   "revisada con cero partículas? Se sobrescribirá la anotación "
                   "anterior, si la hubiera."),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        out_path = self._save_to_txt(path, dets)
        # El informe no relee los .txt: usa el GT que se cargo durante la corrida.
        # Sin este aviso, reetiquetar y regenerar el informe parece no hacer nada.
        aviso = ""
        if any(self.state.results.values()):
            aviso = tr("   ⚠ Vuelve a Ejecutar la detección: el informe usa el "
                       "GT leído en la última corrida.")
        self.lbl_canvas_status.setText(f"✓ Guardado en: {out_path}{aviso}")
        self._refresh_list()
