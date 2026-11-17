"""Page 3 — `GT manual`.

Full bounding-box annotator with:
- Zoom with the mouse wheel
- Pan with the middle button or Space+drag
- Draw boxes with left-click+drag
- Select by clicking on a box
- Move the selected box (drag)
- Resize with handles on the corners/sides
- Delete with `Supr` / `Del`
- Shortcuts 1..9 to change the active class (also on the selected box)
- Undo / Redo (Ctrl+Z, Ctrl+Y)
- Navigation with left/right arrows (auto-saves)
- Exports YOLO .txt + classes.txt
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from polyx.core.detect.protocol import size_image
from polyx.core.detect.yolo import Detection
from polyx.core.editing import (
    HANDLE_TOLERANCE_PX,
    BoxEditor,
    EditBox,
    EditMode,
    Handle,
    RejectionReason,
    rejection_message,
)
from polyx.core.i18n import _, lazy, ngettext
from polyx.core.labels import (
    STUDY_CLASSES,
    bbox_from,
    classes_dir_for,
    find_label_file,
    label_dir_for,
    label_path_for,
    read_classes,
    read_label_file,
    write_classes,
    write_lines,
)
from polyx.gui import iconos
from polyx.gui import theme as T
from polyx.gui.image_pixmap import qpixmap_from_file

from ._base import DetectorPage

HANDLE_SIZE = 8  # px on screen


# ────────────────────────────────────────────────────────────────────
class AnnotCanvas(QGraphicsView):
    """Advanced canvas: zoom, pan, draw/move/resize boxes.

    A thin shell around the shared editing core: it translates events,
    paints, keeps zoom/pan and its signals. The boxes, selection, resize
    handles and the undo/redo history live in
    :class:`polyx.core.editing.BoxEditor`, so the Manual GT page and the
    Labeller cannot drift apart again.
    """

    boxes_changed = Signal()
    selection_changed = Signal(int)  # idx or -1
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(f"QGraphicsView {{ border: 1px solid {T.RULE}; background: #1a1d22; }}")
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._pixmap_item = None
        self._image_path: Path | None = None
        self._editor: BoxEditor | None = None
        self._active_class = 0
        self.class_names: list[str] = list(STUDY_CLASSES)

        # Interaction state the core does not own: panning and the space key.
        self._space_pressed = False
        self._panning = False
        self._last_pan_pos: QPointF | None = None

    # ── Public API ───────────────────────────────────────────────
    @property
    def editor(self) -> BoxEditor | None:
        """The shared editing core of the loaded image, or ``None``."""
        return self._editor

    @property
    def boxes(self) -> list[EditBox]:
        """The current boxes in image pixels (a copy; edit through the canvas)."""
        return list(self._editor.boxes) if self._editor is not None else []

    @property
    def selection(self) -> int:
        """Index of the selected box, or ``-1``."""
        return self._editor.selection if self._editor is not None else -1

    @property
    def selected_idx(self) -> int:
        """The selected box index; kept for the old public attribute."""
        return self.selection

    @property
    def active_class(self) -> int:
        """The class new boxes get; the core owns it once an image is loaded."""
        return self._editor.active_class if self._editor is not None else self._active_class

    def set_class_names(self, names: list[str]):
        self.class_names = list(names)
        if not 0 <= self.active_class < len(self.class_names):
            self._active_class = 0
            if self._editor is not None:
                self._editor.set_active_class(0)
        self._refresh()

    def set_active_class(self, idx: int):
        if not 0 <= idx < len(self.class_names):
            return
        self._active_class = idx
        if self._editor is None:
            return
        self._editor.set_active_class(idx)
        # if a box is selected, change its class (the core records the undo)
        if self._editor.assign_class(idx):
            self._refresh()
            self.boxes_changed.emit()

    def load_image(self, path: Path, existing: list[Detection]):
        self._image_path = path
        pm = qpixmap_from_file(path)
        if pm.isNull():
            self.scene_.clear()
            self._pixmap_item = None
            self._editor = None
            self.status_changed.emit(_("Could not open the image."))
            return
        self.scene_.clear()
        self._pixmap_item = self.scene_.addPixmap(pm)
        self.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))
        self._editor = BoxEditor(
            pm.width(),
            pm.height(),
            [EditBox.from_detection(d) for d in existing],
            active_class=self._active_class,
        )
        # Initial fit
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._refresh()
        self.boxes_changed.emit()

    def detections(self) -> list[Detection]:
        if self._editor is None:
            return []
        return self._editor.detections(dict(enumerate(self.class_names)))

    def clear_boxes(self):
        if self._editor is None or not self._editor.clear():
            return
        self._refresh()
        self.boxes_changed.emit()

    def delete_selected(self):
        if self._editor is None or not self._editor.delete_selected():
            return
        self._refresh()
        self.boxes_changed.emit()
        self.selection_changed.emit(-1)

    def undo(self):
        if self._editor is None or not self._editor.undo():
            return
        self._refresh()
        self.boxes_changed.emit()

    def redo(self):
        if self._editor is None or not self._editor.redo():
            return
        self._refresh()
        self.boxes_changed.emit()

    def zoom_fit(self):
        if self._pixmap_item is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def zoom_100(self):
        self.resetTransform()

    # ── internal helpers ───────────────────────────────────────────
    def _handle_tolerance(self) -> float:
        """The handle hit target, as a distance in image pixels."""
        return HANDLE_TOLERANCE_PX / max(self.transform().m11(), 0.0001)

    def _refresh(self):
        """Redraws all the boxes + handles + labels (without touching the image)."""
        # Remove all items except the pixmap
        for it in list(self.scene_.items()):
            if it is self._pixmap_item:
                continue
            self.scene_.removeItem(it)
        if self._editor is None:
            return

        selected = self._editor.selection
        for i, b in enumerate(self._editor.boxes):
            color = T.class_qcolor_at(self.class_names, b.class_id)
            sel = i == selected
            pen = QPen(color, 3 if sel else 2)
            if sel:
                pen.setStyle(Qt.SolidLine)
            rect = QGraphicsRectItem(QRectF(b.x1, b.y1, b.width, b.height))
            rect.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(45 if sel else 0)
            rect.setBrush(QBrush(fill))
            self.scene_.addItem(rect)

            # Label
            name = (
                self.class_names[b.class_id]
                if b.class_id < len(self.class_names)
                else str(b.class_id)
            )
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
                # 8-point handles, positioned by the core; the painted side
                # stays the shell's HANDLE_SIZE, the grab distance is the
                # core's HANDLE_TOLERANCE_PX.
                hs = HANDLE_SIZE / max(self.transform().m11(), 0.0001)
                for hx, hy in self._editor.handle_points(b):
                    h = QGraphicsRectItem(QRectF(hx - hs / 2, hy - hs / 2, hs, hs))
                    h.setPen(QPen(Qt.white, 1.2))
                    h.setBrush(QBrush(color))
                    h.setZValue(10)
                    self.scene_.addItem(h)

    def _set_cursor_for_handle(self, handle: Handle):
        cursors = {
            Handle.TOP_LEFT: Qt.SizeFDiagCursor,
            Handle.BOTTOM_RIGHT: Qt.SizeFDiagCursor,
            Handle.TOP_RIGHT: Qt.SizeBDiagCursor,
            Handle.BOTTOM_LEFT: Qt.SizeBDiagCursor,
            Handle.TOP: Qt.SizeVerCursor,
            Handle.BOTTOM: Qt.SizeVerCursor,
            Handle.LEFT: Qt.SizeHorCursor,
            Handle.RIGHT: Qt.SizeHorCursor,
        }
        self.setCursor(cursors.get(handle, Qt.CrossCursor))

    def _report_rejection(self, side_px: float):
        """Announce a refused gesture, unless it was an accidental click."""
        message = rejection_message(side_px)
        if message is not None:
            self.status_changed.emit(message)

    # ── events ────────────────────────────────────────────────────
    def wheelEvent(self, ev):
        if self._pixmap_item is None:
            return
        delta = ev.angleDelta().y()
        factor = 1.20 if delta > 0 else 1 / 1.20
        # bounds
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
            if self._editor is not None:
                self._editor.select(-1)
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
            super().keyPressEvent(ev)
            return
        ev.accept()

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Space:
            self._space_pressed = False
            self.setCursor(Qt.CrossCursor)
        super().keyReleaseEvent(ev)

    def mousePressEvent(self, ev):
        if self._editor is None:
            super().mousePressEvent(ev)
            return
        scene_pos = self.mapToScene(ev.position().toPoint())

        # Pan: middle wheel or space+left
        if ev.button() == Qt.MiddleButton or (self._space_pressed and ev.button() == Qt.LeftButton):
            self._panning = True
            self._last_pan_pos = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
            return

        if ev.button() == Qt.LeftButton:
            # Am I over a handle of the selected box?
            handle = self._editor.handle_at(scene_pos.x(), scene_pos.y(), self._handle_tolerance())
            if handle is not Handle.NONE and self._editor.begin_resize(
                handle, scene_pos.x(), scene_pos.y()
            ):
                self._set_cursor_for_handle(handle)
                ev.accept()
                return

            # Am I over a box?
            if self._editor.begin_move(scene_pos.x(), scene_pos.y()):
                self.setCursor(Qt.ClosedHandCursor)
                self._refresh()
                self.selection_changed.emit(self._editor.selection)
                ev.accept()
                return

            # Otherwise, draw a new one. The empty press clears the selection,
            # as the Labeller does and as this canvas did before the shared
            # core: the zero-size draw that starts here is refused, and the
            # refusal must not bring the old selection back with it.
            self._editor.select(-1)
            self._editor.begin_draw(scene_pos.x(), scene_pos.y())
            self._refresh()
            ev.accept()
            return

        if ev.button() == Qt.RightButton:
            # assign the active class to the box under the cursor
            index = self._editor.box_at(scene_pos.x(), scene_pos.y())
            if index >= 0:
                changed = self._editor.assign_class(self.active_class, index)
                self._editor.select(index)
                self._refresh()
                if changed:
                    self.boxes_changed.emit()
                self.selection_changed.emit(index)
            ev.accept()
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._editor is None:
            super().mouseMoveEvent(ev)
            return
        scene_pos = self.mapToScene(ev.position().toPoint())

        if self._panning and self._last_pan_pos is not None:
            delta = ev.position() - self._last_pan_pos
            self._last_pan_pos = ev.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            ev.accept()
            return

        mode = self._editor.mode
        if mode is EditMode.DRAW:
            self._editor.update_draw(scene_pos.x(), scene_pos.y())
            self._refresh()
            box = self._editor.selected_box
            if box is not None:
                self.status_changed.emit(
                    _("Drawing: %(width).0f×%(height).0f px")
                    % {"width": abs(box.width), "height": abs(box.height)}
                )
            ev.accept()
            return

        if mode is EditMode.MOVE:
            self._editor.update_move(scene_pos.x(), scene_pos.y())
            self._refresh()
            ev.accept()
            return

        if mode is EditMode.RESIZE:
            self._editor.update_resize(scene_pos.x(), scene_pos.y())
            self._refresh()
            ev.accept()
            return

        # No button: update the cursor according to hover
        handle = self._editor.handle_at(scene_pos.x(), scene_pos.y(), self._handle_tolerance())
        if handle is not Handle.NONE:
            self._set_cursor_for_handle(handle)
            ev.accept()
            return
        # Hover over a box: hand cursor
        if self._editor.box_at(scene_pos.x(), scene_pos.y()) >= 0 or self._space_pressed:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._editor is not None:
            mode = self._editor.mode
            result = None
            if mode is EditMode.DRAW:
                result = self._editor.finish_draw()
            elif mode is EditMode.MOVE:
                result = self._editor.finish_move()
            elif mode is EditMode.RESIZE:
                result = self._editor.finish_resize()
            if result is not None:
                self._refresh()
                self.boxes_changed.emit()
                if mode is EditMode.DRAW:
                    self.selection_changed.emit(self._editor.selection)
                if result.reason is RejectionReason.TOO_SMALL:
                    self._report_rejection(result.side_px)

        self._panning = False
        self._last_pan_pos = None
        self.setCursor(Qt.CrossCursor)
        super().mouseReleaseEvent(ev)


# ────────────────────────────────────────────────────────────────────
class GTManualPage(DetectorPage):
    STEP_N = 3
    STEP_TITLE = lazy("Manual GT")
    STEP_DESCRIPTION = lazy(
        "Full Ground Truth annotator. Left click to draw, click a box to select it, drag the handles to resize, Space or middle button to pan, wheel to zoom. Keys 1..9: active class · Del: delete · Ctrl+Z/Y: undo/redo · ←/→: previous/next image (autosaves)."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Main card ──
        c1, l1 = self.card(_("Annotation"), "editar")

        # Top bar: name + active class + zoom
        top = QHBoxLayout()
        top.setSpacing(8)
        self.lbl_image_name = QLabel(_("(no image selected)"))
        self.lbl_image_name.setStyleSheet(
            f"color: {T.INK2}; font-weight: 600; font-size: 11pt; border: none;"
        )
        top.addWidget(self.lbl_image_name, 1)

        top.addWidget(QLabel(_("Class:")))
        self.combo_class = QComboBox()
        self.combo_class.addItems(STUDY_CLASSES)
        self.combo_class.currentIndexChanged.connect(self._on_class_combo)
        self.combo_class.setMinimumWidth(120)
        top.addWidget(self.combo_class)

        btn_fit = QPushButton(_("Fit (F)"))
        btn_fit.clicked.connect(lambda: self.canvas.zoom_fit())  # noqa: PLW0108
        top.addWidget(btn_fit)
        btn_100 = QPushButton("100 %")
        btn_100.clicked.connect(lambda: self.canvas.zoom_100())  # noqa: PLW0108
        top.addWidget(btn_100)
        l1.addLayout(top)

        # Body: image list + canvas
        body = QHBoxLayout()
        body.setSpacing(12)

        # Image list (compact)
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        lbl_files = QLabel(_("Images"))
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
        self.canvas.set_class_names(STUDY_CLASSES)
        self.canvas.setMinimumSize(QSize(560, 420))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_col.addWidget(self.canvas, 1)

        # Canvas mini-status
        self.lbl_canvas_status = QLabel(_("Ready. Click and drag to draw."))
        self.lbl_canvas_status.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        right_col.addWidget(self.lbl_canvas_status)

        body.addLayout(right_col, 1)
        l1.addLayout(body)

        # Bottom button row
        btns = QHBoxLayout()
        btns.setSpacing(8)
        b_save = QPushButton(_("Save ground truth (.txt YOLO)"))
        b_save.setIcon(iconos.icon("guardar", 15, T.ON_ACCENT))
        b_save.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        b_save.setCursor(Qt.PointingHandCursor)
        b_save.clicked.connect(self._save_current)
        btns.addWidget(b_save)

        b_prev = QPushButton(_("←  Previous"))
        b_prev.clicked.connect(self._go_prev)
        btns.addWidget(b_prev)
        b_next = QPushButton(_("Next  →"))
        b_next.clicked.connect(self._go_next)
        btns.addWidget(b_next)

        b_undo = QPushButton(_("Undo"))
        b_undo.setIcon(iconos.icon("deshacer", 15, T.INK2))
        b_undo.clicked.connect(lambda: self.canvas.undo())  # noqa: PLW0108
        btns.addWidget(b_undo)
        b_redo = QPushButton(_("Redo"))
        b_redo.setIcon(iconos.icon("rehacer", 15, T.INK2))
        b_redo.clicked.connect(lambda: self.canvas.redo())  # noqa: PLW0108
        btns.addWidget(b_redo)

        b_del = QPushButton(_("Delete selection"))
        b_del.setIcon(iconos.icon("papelera", 15, T.INK2))
        b_del.clicked.connect(lambda: self.canvas.delete_selected())  # noqa: PLW0108
        btns.addWidget(b_del)
        b_clear = QPushButton(_("Clear all"))
        b_clear.clicked.connect(self._clear_with_confirm)
        btns.addWidget(b_clear)
        btns.addStretch(1)

        self.lbl_count = QLabel(_("0 boxes"))
        self.lbl_count.setStyleSheet(f"color: {T.INK3}; border: none;")
        btns.addWidget(self.lbl_count)
        l1.addLayout(btns)

        self.body.addWidget(c1)

        # Global keyboard shortcuts for arrow navigation
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._go_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._go_next)

        # Subscriptions
        self.state.images_changed.connect(self._refresh_list)
        self.canvas.boxes_changed.connect(self._on_boxes_changed)
        self.canvas.status_changed.connect(self.lbl_canvas_status.setText)
        self._refresh_list()

    # ── Image list ──────────────────────────────────────────
    def _refresh_list(self):
        # Preserve the selection if it exists. The language change rebuilds
        # this page, so the image the user was annotating lives in the state.
        wanted = self.state.gt_image
        prev = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.state.images:
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            gt = find_label_file(p, self.state.gt_folder)
            if gt:
                it.setForeground(QColor(T.OK))
                it.setText(f"✓ {p.name}")
            self.list.addItem(it)
        self.list.blockSignals(False)
        if wanted is not None:
            for row in range(self.list.count()):
                if Path(self.list.item(row).data(Qt.UserRole)) == wanted:
                    prev = row
                    break
        if 0 <= prev < self.list.count():
            self.list.setCurrentRow(prev)
        elif self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _go_prev(self):
        if self.list.count() == 0:
            return
        r = max(0, self.list.currentRow() - 1)
        self.list.setCurrentRow(r)
        self.canvas.setFocus()

    def _go_next(self):
        if self.list.count() == 0:
            return
        r = min(self.list.count() - 1, self.list.currentRow() + 1)
        self.list.setCurrentRow(r)
        self.canvas.setFocus()

    def _on_class_combo(self, idx: int):
        self.canvas.set_active_class(idx)
        self.canvas.setFocus()

    def _on_list_change(self):
        # Auto-save the previous image if it has boxes
        # (we do it before loading the new one)
        if hasattr(self, "_current_path") and self._current_path is not None:
            self._auto_save(self._current_path, silent=True)

        it = self.list.currentItem()
        if not it:
            return
        path = Path(it.data(Qt.UserRole))
        self._current_path = path
        self.state.gt_image = path
        self.lbl_image_name.setText(path.name)
        existing: list[Detection] = []
        gt_txt = find_label_file(path, self.state.gt_folder)
        # Where a save would go: that folder's classes.txt governs a fresh
        # dataset, so opening it after another never inherits its names.
        names = read_classes(classes_dir_for(label_dir_for(path, self.state.gt_folder)))
        if gt_txt:
            # The dataset root owns the class names: a split layout keeps one
            # classes.txt next to images/ and labels/, so every split agrees.
            names = read_classes(classes_dir_for(gt_txt.parent))
            # Same dimensions the detector uses: header-only and EXIF-aware,
            # so the frame matches the one the boxes were annotated on.
            wh = size_image(path)
            if wh:
                existing, _malformed = read_label_file(
                    gt_txt,
                    wh[0],
                    wh[1],
                    dict(enumerate(names)),
                )
        self._set_class_names(names)
        self.canvas.load_image(path, existing)
        self._on_boxes_changed()
        self.canvas.setFocus()

    def _set_class_names(self, names: list[str]) -> None:
        """Keep the canvas and the class picker on the dataset's class names."""
        if not names:
            return
        combo_names = [self.combo_class.itemText(i) for i in range(self.combo_class.count())]
        if self.canvas.class_names == names and combo_names == names:
            return
        self.canvas.set_class_names(names)
        self.combo_class.blockSignals(True)
        self.combo_class.clear()
        self.combo_class.addItems(names)
        self.combo_class.setCurrentIndex(self.canvas.active_class)
        self.combo_class.blockSignals(False)

    def _on_boxes_changed(self):
        n = len(self.canvas.boxes)
        self.lbl_count.setText(ngettext("%(count)d box", "%(count)d boxes", n) % {"count": n})

    def _clear_with_confirm(self):
        if not self.canvas.boxes:
            return
        count = len(self.canvas.boxes)
        message = ngettext(
            "Delete the %(count)d box on this image?",
            "Delete the %(count)d boxes on this image?",
            count,
        ) % {"count": count}
        message += "\n" + _("(You can undo with Ctrl+Z)")
        r = QMessageBox.question(
            self,
            _("Clear all"),
            message,
            QMessageBox.Yes | QMessageBox.No,
        )
        if r == QMessageBox.Yes:
            self.canvas.clear_boxes()

    # ── Saving ────────────────────────────────────────────────
    def _save_to_txt(self, img_path: Path, dets: list[Detection]):
        wh = size_image(img_path)
        if not wh or wh[0] <= 0 or wh[1] <= 0:
            raise ValueError(_("Could not read the size of %(name)s") % {"name": img_path.name})
        W, H = wh
        out_path = write_lines(
            label_path_for(img_path, self.state.gt_folder),
            [bbox_from(d, W, H) for d in dets],
        )

        # classes.txt next to the labels — but only when there is none yet:
        # overwriting would corrupt a dataset that annotates other classes.
        # ``classes_dir_for`` puts it once at the dataset root for a split
        # layout, so labels/train and labels/val share one list.
        cls_dir = classes_dir_for(out_path.parent)
        if not (cls_dir / "classes.txt").exists():
            write_classes(cls_dir, self.canvas.class_names)
        return out_path

    def _auto_save(self, path: Path, silent: bool = False):
        dets = self.canvas.detections()
        if not dets:
            return
        try:
            self._save_to_txt(path, dets)
            if not silent:
                self.lbl_canvas_status.setText(_("Autosaved."))
        except Exception as e:  # noqa: BLE001
            if not silent:
                QMessageBox.warning(self, _("Error while saving"), str(e))

    def _save_current(self):
        it = self.list.currentItem()
        if not it:
            QMessageBox.information(self, _("Manual GT"), _("Select an image first."))
            return
        path = Path(it.data(Qt.UserRole))
        dets = self.canvas.detections()
        if not dets:
            # Saving zero boxes is a datum, not an error. It used to be rejected and the
            # previous .txt survived intact: deleting all the misplaced boxes
            # from a plate had no effect, and the report kept showing
            # the old annotation as if nothing had happened. Also, the detector only counts the
            # false positives of an empty plate if its .txt exists.
            r = QMessageBox.question(
                self,
                _("Manual GT"),
                _(
                    "No boxes have been drawn. Save this image as reviewed with zero particles? Any previous annotation will be overwritten."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return
        out_path = self._save_to_txt(path, dets)
        # The report does not re-read the .txt files: it uses the GT loaded during the run.
        # Without this warning, relabeling and regenerating the report seems to do nothing.
        warning_2 = ""
        if any(self.state.results.values()):
            warning_2 = _(
                "   ⚠ Run the detection again: the report uses the ground truth read during the last run."
            )
        self.lbl_canvas_status.setText(
            _("✓ Saved to: %(path)s%(warning)s") % {"path": out_path, "warning": warning_2}
        )
        self._refresh_list()
