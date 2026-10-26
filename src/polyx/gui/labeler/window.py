"""Main window of the Labeller — 3 panels: list | canvas | classes."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from polyx.core.i18n import tr
from polyx.gui import iconos, marca
from polyx.gui import theme as T
from polyx.gui.widgets import HLine, LogoBadge

from .canvas import BboxCanvas
from .state import LabelerState

_PRESETS: dict[str, list[str]] = {
    "Microplásticos Nile Red": ["PET", "PP", "LDPE"],
    "General (1 clase)": ["objeto"],
    "Personalizado": [],
}

# Colors matching those of the canvas
_CLS_COLORS = [
    "#e3342f",
    "#ff8c00",
    "#ffd700",
    "#00b4d8",
    "#48cae4",
    "#0077b6",
    "#023e8a",
    "#7b2d8b",
]


class LabelerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        marca.titular(self, "etiquetador")
        self.resize(1400, 880)
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")

        self.state = LabelerState()
        self._pre_model = None  # YoloModel loaded on demand

        # Thumbnails in the background: generating them all at once froze the
        # window for ~9 s with 552 crops, because each one decodes the full
        # image.
        self._queue_thumbs: list[int] = []
        self._thumb_timer = QTimer(self)
        self._thumb_timer.timeout.connect(self._process_thumbs)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_left())
        self.canvas = BboxCanvas(self.state)
        root.addWidget(self.canvas, 1)
        root.addWidget(self._build_right())

        self._connect_signals()
        self._install_shortcuts()
        self._apply_preset("Microplásticos Nile Red")

        # ── Auto-save every 60 s ──
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start(60_000)

    # ── Left panel ────────────────────────────────────────────
    def _build_left(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(230)
        panel.setStyleSheet(f"QFrame {{ background: {T.BG}; border-right: 1px solid {T.RULE}; }}")
        lv = QVBoxLayout(panel)
        lv.setContentsMargins(12, 14, 12, 12)
        lv.setSpacing(8)

        lv.addWidget(LogoBadge("POLY-X", "Etiquetador"))
        lv.addSpacing(6)

        btn_open = QPushButton(tr("Abrir carpeta…"))
        btn_open.setIcon(iconos.icon("dataset", 15, T.INK2))
        btn_open.setObjectName("primary")
        btn_open.setCursor(Qt.PointingHandCursor)
        btn_open.clicked.connect(self._open_folder)
        lv.addWidget(btn_open)

        self.lbl_count = QLabel(tr("Sin imágenes"))
        self.lbl_count.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        lv.addWidget(self.lbl_count)

        self.img_list = QListWidget()
        self.img_list.setIconSize(QSize(52, 40))
        self.img_list.setStyleSheet(f"""
            QListWidget {{
                border: 1px solid {T.RULE}; border-radius: 6px; background: {T.BG};
            }}
            QListWidget::item {{ padding: 4px 6px; font-size: 9pt; }}
            QListWidget::item:selected {{
                background: #dde7f4; color: {T.ACCENT_D_TX};
            }}
        """)
        self.img_list.currentRowChanged.connect(self._on_list_select)
        lv.addWidget(self.img_list, 1)

        nav = QHBoxLayout()
        nav.setSpacing(6)
        btn_prev = QPushButton(tr("← Anterior"))
        btn_next = QPushButton(tr("Siguiente →"))
        for b in (btn_prev, btn_next):
            b.setFixedHeight(28)
        btn_prev.clicked.connect(self.state.prev_image)
        btn_next.clicked.connect(self.state.next_image)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        lv.addLayout(nav)

        self.lbl_idx = QLabel("")
        self.lbl_idx.setAlignment(Qt.AlignCenter)
        self.lbl_idx.setStyleSheet(f"color: {T.INK3}; font-size: 8.5pt;")
        lv.addWidget(self.lbl_idx)

        # Count progress: essential when there are hundreds of crops
        # spread over several sessions.
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {T.RULE}; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {T.OK}; border-radius: 3px; }}"
        )
        lv.addWidget(self.bar)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setAlignment(Qt.AlignCenter)
        self.lbl_progress.setStyleSheet(f"color: {T.INK2}; font-size: 8.5pt;")
        lv.addWidget(self.lbl_progress)

        btn_pend = QPushButton(tr("⏭  Siguiente sin revisar"))
        btn_pend.setToolTip(tr("Salta a la próxima imagen que aún no has revisado (Tab)"))
        btn_pend.clicked.connect(self._go_next_pending)
        lv.addWidget(btn_pend)

        return panel

    # ── Right panel ──────────────────────────────────────────
    def _build_right(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(244)
        panel.setStyleSheet(f"QFrame {{ background: {T.BG}; border-left: 1px solid {T.RULE}; }}")
        rv = QVBoxLayout(panel)
        rv.setContentsMargins(14, 14, 14, 14)
        rv.setSpacing(9)

        # Preset
        rv.addWidget(QLabel(tr("Preset de clases:")))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(_PRESETS.keys())
        self.combo_preset.currentTextChanged.connect(self._apply_preset)
        rv.addWidget(self.combo_preset)

        rv.addWidget(HLine())
        rv.addWidget(QLabel(tr("Clases activas:")))

        self.class_frame = QFrame()
        self.class_frame.setStyleSheet("QFrame { border: none; background: transparent; }")
        self._class_layout = QVBoxLayout(self.class_frame)
        self._class_layout.setContentsMargins(0, 0, 0, 0)
        self._class_layout.setSpacing(4)
        self._class_btns: list[QPushButton] = []
        rv.addWidget(self.class_frame)

        btn_add = QPushButton(tr("+ Agregar clase"))
        btn_add.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {T.ACCENT_TX}; "
            f"border: 1px dashed {T.ACCENT}; border-radius: 4px; padding: 4px; }}"
            f"QPushButton:hover {{ background: {T.BG_SOFT}; }}"
        )
        btn_add.clicked.connect(self._add_class)
        rv.addWidget(btn_add)

        rv.addWidget(HLine())

        # Statistics of the current image
        lbl_stat = QLabel(tr("Anotaciones (imagen actual):"))
        lbl_stat.setStyleSheet(f"color: {T.INK2}; font-weight: 600;")
        rv.addWidget(lbl_stat)
        self.lbl_box_info = QLabel("—")
        self.lbl_box_info.setWordWrap(True)
        self.lbl_box_info.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        rv.addWidget(self.lbl_box_info)

        rv.addWidget(HLine())

        # Pre-annotation
        lbl_pre = QLabel(tr("Pre-anotación automática"))
        lbl_pre.setStyleSheet("font-weight: 600; font-size: 10pt;")
        rv.addWidget(lbl_pre)

        rv.addWidget(QLabel(tr("Modelo .pt:")))
        self.lbl_model = QLabel(tr("Sin modelo"))
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet(
            f"color: {T.INK3}; font-size: 9pt; background: {T.BG_SOFT}; "
            f"border: 1px solid {T.RULE}; border-radius: 4px; padding: 4px;"
        )
        rv.addWidget(self.lbl_model)

        btn_load_model = QPushButton(tr("Cargar modelo…"))
        btn_load_model.setIcon(iconos.icon("dataset", 15, T.INK2))
        btn_load_model.clicked.connect(self._load_model)
        rv.addWidget(btn_load_model)

        row_pre = QHBoxLayout()
        row_pre.setSpacing(6)
        row_pre.addWidget(QLabel(tr("conf:")))
        self.sb_pre_conf = QDoubleSpinBox()
        self.sb_pre_conf.setRange(0.01, 0.99)
        self.sb_pre_conf.setSingleStep(0.05)
        self.sb_pre_conf.setDecimals(2)
        self.sb_pre_conf.setValue(0.10)
        row_pre.addWidget(self.sb_pre_conf)
        row_pre.addWidget(QLabel(tr("imgsz:")))
        self.sb_pre_imgsz = QSpinBox()
        self.sb_pre_imgsz.setRange(320, 8192)
        self.sb_pre_imgsz.setSingleStep(64)
        self.sb_pre_imgsz.setValue(2080)
        self.sb_pre_imgsz.setToolTip(
            tr(
                "Resolución de inferencia. Con partículas diminutas en fotos grandes,\n"
                "un valor bajo no propone nada."
            )
        )
        row_pre.addWidget(self.sb_pre_imgsz)
        rv.addLayout(row_pre)

        btn_pre_one = QPushButton(tr("Pre-anotar imagen actual"))
        btn_pre_one.clicked.connect(self._pre_annotate_current)
        rv.addWidget(btn_pre_one)

        btn_pre_all = QPushButton(tr("Pre-anotar TODAS"))
        btn_pre_all.setObjectName("primary")
        btn_pre_all.clicked.connect(self._pre_annotate_all)
        rv.addWidget(btn_pre_all)

        rv.addWidget(HLine())

        self.chk_zoom = QCheckBox(tr("Mantener zoom entre imágenes"))
        self.chk_zoom.setChecked(True)
        self.chk_zoom.setToolTip(
            tr(
                "Conserva el nivel de acercamiento al cambiar de recorte.\n"
                "Con cientos de recortes, reencuadrar cada vez cuesta mucho tiempo.\n"
                "F reencuadra manualmente."
            )
        )
        self.chk_zoom.toggled.connect(lambda v: setattr(self.canvas, "keep_zoom", v))
        rv.addWidget(self.chk_zoom)

        rv.addStretch(1)

        btn_rev = QPushButton(tr("✓  Revisada, siguiente   (Espacio)"))
        btn_rev.setToolTip(
            tr(
                "Deja constancia de que miraste esta imagen aunque no tenga partículas.\n"
                "Sin esto, una placa vacía revisada es indistinguible de una sin mirar."
            )
        )
        btn_rev.clicked.connect(self._mark_reviewed)
        rv.addWidget(btn_rev)

        btn_save = QPushButton(tr("Guardar (.txt)"))
        btn_save.setIcon(iconos.icon("guardar", 15, T.INK2))
        btn_save.setObjectName("primary")
        btn_save.clicked.connect(self._save)
        rv.addWidget(btn_save)

        self.lbl_status = QLabel(tr("● Listo"))
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 9pt;")
        rv.addWidget(self.lbl_status)

        return panel

    # ── Connections and shortcuts ─────────────────────────────
    def _connect_signals(self):
        self.state.images_loaded.connect(self._on_images_loaded)
        self.state.image_changed.connect(self._on_image_changed)
        self.state.annotations_changed.connect(self._on_annotations_changed)
        self.state.active_class_changed.connect(self._on_active_class_changed)
        self.state.classes_changed.connect(self._rebuild_class_buttons)
        self.state.progress_changed.connect(self._on_progress_changed)
        self.canvas.box_rejected.connect(self._on_box_rejected)

    def _on_progress_changed(self):
        idx = self.state.current_idx
        if 0 <= idx < self.img_list.count():
            self.img_list.item(idx).setText(self._text_item(self.state.images[idx]))
        self._update_progress()

    def _on_box_rejected(self, side: float):
        self._set_status(
            f"⚠ Caja de {side:.1f} px descartada (mínimo "
            f"{self.canvas.SIDE_MINIMUM_PX:.0f} px) — acerca el zoom",
            T.WARN,
        )
        QTimer.singleShot(3500, lambda: self._set_status("● Listo", T.OK))

    def _install_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, self._save)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.state.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.state.redo)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.state.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.state.next_image)
        QShortcut(QKeySequence(Qt.Key_Tab), self, self._go_next_pending)

    # ── Class management ─────────────────────────────────────
    def _rebuild_class_buttons(self, names: list[str] | None = None):
        if names is None:
            names = self.state.class_names
        for b in self._class_btns:
            b.setParent(None)
        self._class_btns.clear()

        for i, name in enumerate(names):
            color = _CLS_COLORS[i % len(_CLS_COLORS)]
            b = QPushButton(f"  {i + 1}  {name}")
            b.setCheckable(True)
            b.setChecked(i == self.state.active_class)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton {{ background: {T.BG}; color: {T.INK2}; "
                f"border: 1.5px solid {T.RULE}; border-radius: 5px; "
                f"padding: 5px 8px; text-align: left; }}"
                f"QPushButton:checked {{ background: {color}22; border-color: {color}; "
                f"color: {T.INK}; font-weight: 600; }}"
                f"QPushButton:hover {{ background: {T.BG_SOFT}; }}"
            )
            b.clicked.connect(lambda _, x=i: self.state.set_active_class(x))
            self._class_layout.addWidget(b)
            self._class_btns.append(b)

    def _on_active_class_changed(self, cls_id: int):
        for i, b in enumerate(self._class_btns):
            b.setChecked(i == cls_id)

    def _apply_preset(self, preset: str):
        names = _PRESETS.get(preset, [])
        if names:
            self.state.set_class_names(names)

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "Nueva clase", "Nombre de la clase:")
        if ok and name.strip():
            self.state.set_class_names(list(self.state.class_names) + [name.strip()])  # noqa: RUF005

    # ── Files ────────────────────────────────────────────────
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if folder:
            self.state.load_images(Path(folder))
            self.state.save_classes_txt()
            self._set_status("● Carpeta cargada", T.OK)

    def _save(self):
        self.state.save_current()
        self.state.save_classes_txt()
        self._set_status("● Guardado", T.OK)

    # ── Pre-annotation ────────────────────────────────────────
    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar modelo YOLO", "", "Modelo YOLO (*.pt)")
        if not path:
            return
        try:
            from polyx.core.detect.yolo import YoloModel

            self._pre_model = YoloModel(path)
            self.lbl_model.setText(Path(path).name)
            self._set_status("● Modelo cargado", T.OK)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Error al cargar modelo"), str(e))

    def _pre_annotate_current(self):
        if not self._pre_model:
            QMessageBox.warning(self, tr("Sin modelo"), tr("Carga un modelo .pt primero."))
            return
        img = self.state.current_image
        if img is None:
            return
        if self.state.current_boxes:
            r = QMessageBox.question(
                self,
                tr("Sobrescribir"),
                tr("Esta imagen ya tiene anotaciones. ¿Sobrescribir con el modelo?"),
            )
            if r != QMessageBox.Yes:
                return
        try:
            dets = self._pre_model.predict_auto(str(img), **self._pre_annotation_params())
            boxes = self._dets_to_boxes(dets, img)
            self.state.set_current_boxes(boxes)
            self._set_status(f"● Pre-anotadas {len(boxes)} cajas", T.OK)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Error en pre-anotación"), str(e))

    def _pre_annotate_all(self):
        if not self._pre_model:
            QMessageBox.warning(self, tr("Sin modelo"), tr("Carga un modelo .pt primero."))
            return
        if not self.state.images:
            return

        total = len(self.state.images)
        pd = QProgressDialog("Pre-anotando imágenes…", "Cancelar", 0, total, self)
        pd.setWindowTitle(tr("Pre-anotando"))
        pd.setMinimumDuration(0)
        pd.setValue(0)

        for i, img_path in enumerate(self.state.images):
            if pd.wasCanceled():
                break
            pd.setValue(i)
            pd.setLabelText(f"{i + 1}/{total}: {img_path.name}")
            key = str(img_path)
            # Only annotate the empty ones
            existing = self.state._annotations.get(key)
            if existing:
                continue
            try:
                dets = self._pre_model.predict_auto(str(img_path), **self._pre_annotation_params())
                boxes = self._dets_to_boxes(dets, img_path)
                self.state._annotations[key] = boxes
                self.state._save_labels_for(img_path, boxes)
            except Exception:  # noqa: BLE001, S110
                pass

        pd.setValue(total)
        self.state.annotations_changed.emit()
        self._set_status("● Pre-anotación completa", T.OK)

    def _pre_annotation_params(self) -> dict:
        """Inference parameters for the pre-annotation.

        They used to be fixed to ``conf=0.25, device="cpu"`` and without
        ``imgsz``, i.e. 640 by default: with ~12 px particles in
        high-resolution photos it proposed nothing, and on CPU it took
        forever.
        """
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
        # Automatic tiling: a full plate (~3260 px) is split into overlapping
        # tiles and the boxes return to plate coordinates. The study crops
        # (1630 px per side) stay below the threshold and are
        # inferred in one piece, so nothing changes here.
        return dict(  # noqa: C408
            conf=float(self.sb_pre_conf.value()),
            imgsz=int(self.sb_pre_imgsz.value()),
            device=device,
            tiling="auto",
            threshold_px=2000,
            overlap=0.25,
        )

    def _dets_to_boxes(self, dets, img_path: Path):
        from .state import BBox

        try:
            from PIL import Image

            with Image.open(str(img_path)) as im:
                iw, ih = im.size
        except Exception:  # noqa: BLE001
            iw = ih = 640

        boxes = []
        for d in dets:
            w = d.x2 - d.x1
            h = d.y2 - d.y1
            cx = (d.x1 + w / 2) / iw
            cy = (d.y1 + h / 2) / ih
            boxes.append(BBox(d.class_id, cx, cy, w / iw, h / ih))
        return boxes

    # ── State callbacks ──────────────────────────────────────
    def _on_images_loaded(self, imgs: list):
        self.img_list.clear()
        for img in imgs:
            item = QListWidgetItem(self._text_item(img, count_2=True))
            self.img_list.addItem(item)
        self.lbl_count.setText(f"{len(imgs)} imagen(es)")
        # Thumbnails are generated later, in batches, so as not to freeze the
        # window: each one decodes the full image and there are hundreds.
        self._queue_thumbs = list(range(len(imgs)))
        self._thumb_timer.start(0)
        self._update_progress()

    def _text_item(self, img: Path, count_2: bool = False) -> str:
        """Status mark: ✓ with particles · reviewed, empty · ○ not reviewed."""
        if not self.state.is_reviewed(img):
            return f"○  {img.name}"
        n = len(self.state._annotations.get(str(img), []))
        if count_2 and str(img) not in self.state._annotations:
            # Without loading into memory: it counts the lines of the .txt directly
            txt = self.state._label_path_for(img)
            try:
                n = sum(1 for l in txt.read_text(encoding="utf-8").splitlines() if l.strip())  # noqa: E741
            except OSError:
                n = 0
        return f"✓  {img.name}  ({n})" if n else f"·  {img.name}  (0)"

    def _process_thumbs(self):
        """Generates a few thumbnails per tick, without blocking the interface."""
        if not self._queue_thumbs:
            self._thumb_timer.stop()
            return
        for _ in range(4):
            if not self._queue_thumbs:
                break
            i = self._queue_thumbs.pop(0)
            if i >= self.img_list.count() or i >= len(self.state.images):
                continue
            self.img_list.item(i).setIcon(self._make_thumb(self.state.images[i]))

    def _make_thumb(self, img_path: Path) -> QIcon:
        """Loads the 52×40 thumbnail of the image."""
        try:
            pix = QPixmap(str(img_path))
            if pix.isNull():
                return QIcon()
            return QIcon(pix.scaled(QSize(52, 40), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:  # noqa: BLE001
            return QIcon()

    def _update_progress(self):
        total = len(self.state.images)
        made = self.state.n_reviewed()
        if not total:
            self.lbl_progress.setText("")
            self.bar.setValue(0)
            return
        self.bar.setMaximum(total)
        self.bar.setValue(made)
        self.lbl_progress.setText(
            f"{made} / {total} revisadas  ({100 * made / total:.0f}%)   faltan {total - made}"
        )

    def _go_next_pending(self):
        i = self.state.next_unreviewed()
        if i < 0:
            QMessageBox.information(
                self, tr("Conteo completo"), tr("No quedan imágenes sin revisar.")
            )
            return
        self.state.goto(i)

    def _mark_reviewed(self):
        self.state.mark_reviewed()
        self.state.next_image()

    def _on_image_changed(self, idx: int):
        self.img_list.blockSignals(True)
        self.img_list.setCurrentRow(idx)
        self.img_list.blockSignals(False)
        total = len(self.state.images)
        self.lbl_idx.setText(f"{idx + 1} / {total}")

    def _on_annotations_changed(self):
        boxes = self.state.current_boxes
        n = len(boxes)

        cls_counts: dict[str, int] = {}
        for b in boxes:
            name = (
                self.state.class_names[b.class_id]
                if b.class_id < len(self.state.class_names)
                else str(b.class_id)
            )
            cls_counts[name] = cls_counts.get(name, 0) + 1

        if cls_counts:
            lines = [f"{n} caja(s)"] + [f"  {k}: {v}" for k, v in cls_counts.items()]
            self.lbl_box_info.setText("\n".join(lines))
        else:
            self.lbl_box_info.setText(tr("Sin anotaciones"))

        # Update the status indicator in the list (without losing the icon)
        idx = self.state.current_idx
        if 0 <= idx < self.img_list.count():
            self.img_list.item(idx).setText(self._text_item(self.state.images[idx]))
        self._update_progress()

    def _on_list_select(self, row: int):
        if row >= 0 and row != self.state.current_idx:
            self.state.goto(row)

    def _autosave(self):
        """Silent auto-save every 60 s."""
        if self.state.current_image is None:
            return
        self.state.save_current()
        self.state.save_classes_txt()
        self._set_status("● Autoguardado", T.INK3)
        QTimer.singleShot(2500, lambda: self._set_status("● Listo", T.OK))

    def _set_status(self, text: str, color: str = T.OK):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt;")
