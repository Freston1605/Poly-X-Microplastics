"""Main window of the Viewer — canvas (left) | control panel (right)."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from polyx.core.i18n import tr
from polyx.gui import iconos, marca
from polyx.gui import theme as T
from polyx.gui.widgets import HLine, LogoBadge

from .canvas import ViewerCanvas
from .state import ViewerState


def _there_are_gpu() -> bool:
    """CUDA available. Queried lazily: importing torch is slow."""

    try:
        import torch

        return bool(torch.cuda.is_available())

    except Exception:
        return False


class ViewerWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        marca.titular(self, "visor")

        self.resize(1420, 900)

        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")

        self.state = ViewerState()

        central = QWidget()

        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        root.setContentsMargins(0, 0, 0, 0)

        root.setSpacing(0)

        # Canvas (center/left)

        canvas_wrap = QWidget()

        canvas_wrap.setStyleSheet(f"background: {T.BG_SOFT};")

        cw_lay = QVBoxLayout(canvas_wrap)

        cw_lay.setContentsMargins(0, 0, 0, 0)

        cw_lay.setSpacing(0)

        self.canvas = ViewerCanvas(self.state)

        cw_lay.addWidget(self.canvas, 1)

        # Bottom status/calibration bar

        self.status_bar = QLabel(tr("  Sin imagen  —  Sin calibración"))

        self.status_bar.setFixedHeight(28)

        self.status_bar.setStyleSheet(
            f"background: {T.BG}; border-top: 1px solid {T.RULE}; "
            f"color: {T.INK2}; font-size: 9.5pt; padding: 0 12px;"
        )

        cw_lay.addWidget(self.status_bar)

        root.addWidget(canvas_wrap, 1)

        # Right panel

        right_scroll = QScrollArea()

        right_scroll.setFixedWidth(280)

        right_scroll.setWidgetResizable(True)

        right_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; border-left: 1px solid {T.RULE}; background: {T.BG}; }}"
        )

        right_content = QWidget()

        right_content.setStyleSheet(f"background: {T.BG};")

        self._rv = QVBoxLayout(right_content)

        self._rv.setContentsMargins(14, 14, 14, 14)

        self._rv.setSpacing(10)

        right_scroll.setWidget(right_content)

        root.addWidget(right_scroll)

        self._build_right_panel()

        self._connect_signals()

        self._install_shortcuts()

        self.setAcceptDrops(True)

    # ── Right panel ─────────────────────────────────────────

    def _build_right_panel(self):

        rv = self._rv

        rv.addWidget(LogoBadge("POLY-X", "Visor"))

        rv.addSpacing(4)

        # ── Open image / folder ──

        lbl_open = QLabel(tr("Imagen"))

        lbl_open.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_open)

        row_open = QHBoxLayout()

        btn_img = QPushButton(tr("Imagen"))
        btn_img.setIcon(iconos.icon("imagenes", 15, T.INK2))

        btn_fol = QPushButton(tr("Carpeta"))
        btn_fol.setIcon(iconos.icon("dataset", 15, T.INK2))

        btn_img.clicked.connect(self._open_image)

        btn_fol.clicked.connect(self._open_folder)

        row_open.addWidget(btn_img)

        row_open.addWidget(btn_fol)

        rv.addLayout(row_open)

        self.lbl_img_name = QLabel(tr("Sin imagen"))

        self.lbl_img_name.setWordWrap(True)

        self.lbl_img_name.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        rv.addWidget(self.lbl_img_name)

        # Navigation (only shown with a folder)

        self.nav_frame = QFrame()

        self.nav_frame.setStyleSheet("QFrame { border: none; background: transparent; }")

        nav_lay = QHBoxLayout(self.nav_frame)

        nav_lay.setContentsMargins(0, 0, 0, 0)

        nav_lay.setSpacing(6)

        btn_prev = QPushButton("←")

        btn_next = QPushButton("→")

        btn_prev.setFixedWidth(36)

        btn_next.setFixedWidth(36)

        btn_prev.clicked.connect(self.state.prev_image)

        btn_next.clicked.connect(self.state.next_image)

        self.lbl_nav = QLabel("1 / 1")

        self.lbl_nav.setAlignment(Qt.AlignCenter)

        nav_lay.addWidget(btn_prev)

        nav_lay.addWidget(self.lbl_nav, 1)

        nav_lay.addWidget(btn_next)

        self.nav_frame.setVisible(False)

        rv.addWidget(self.nav_frame)

        rv.addWidget(HLine())

        # ── Model ──

        lbl_mod = QLabel(tr("Modelo"))

        lbl_mod.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_mod)

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

        rv.addWidget(QLabel(tr("Confianza mínima:")))

        self.spin_conf = QDoubleSpinBox()

        self.spin_conf.setRange(0.01, 0.99)

        self.spin_conf.setSingleStep(0.05)

        self.spin_conf.setValue(0.25)

        self.spin_conf.setDecimals(2)

        rv.addWidget(self.spin_conf)

        # Inference resolution: decisive with tiny particles. At 640 px

        # a 12 px particle in a 4096 photo collapses to ~2 px and nothing is

        # detected. It must be possible to raise it.

        rv.addWidget(QLabel(tr("Resolución de inferencia (px):")))

        self.spin_imgsz = QSpinBox()

        self.spin_imgsz.setRange(320, 8192)

        self.spin_imgsz.setSingleStep(32)

        self.spin_imgsz.setValue(2080)

        self.spin_imgsz.setToolTip(
            tr(
                "Lado mayor al que se redimensiona la imagen para inferir.\n"
                "Más alto = partículas más grandes para la red, pero más memoria.\n"
                "Se redondea al múltiplo de 32 más cercano."
            )
        )

        rv.addWidget(self.spin_imgsz)

        self.chk_gpu = QCheckBox(tr("Usar GPU si está disponible"))

        self.chk_gpu.setChecked(True)

        rv.addWidget(self.chk_gpu)

        self.btn_detect = QPushButton(tr("Detectar"))
        self.btn_detect.setIcon(iconos.icon("ejecutar", 15, T.INK2))

        self.btn_detect.setObjectName("primary")

        self.btn_detect.clicked.connect(self._detect)

        rv.addWidget(self.btn_detect)

        self.btn_load_txt = QPushButton(tr("Cargar etiquetas (.txt)"))
        self.btn_load_txt.setIcon(iconos.icon("texto", 15, T.INK2))

        self.btn_load_txt.setToolTip(
            tr(
                "Muestra las anotaciones YOLO que acompañan a la imagen.\n"
                "Sirve para revisar el conteo manual sobre la placa, con las\n"
                "tallas ya convertidas a µm por la calibración."
            )
        )

        self.btn_load_txt.clicked.connect(self._load_labels)

        rv.addWidget(self.btn_load_txt)

        self.btn_load_run = QPushButton(tr("Cargar predicciones de una corrida"))
        self.btn_load_run.setIcon(iconos.icon("dataset", 15, T.INK2))

        self.btn_load_run.setToolTip(
            tr(
                "Lee las predicciones que el Detector dejó en runs/detect_.../ "
                "para esta misma foto. Sirve para revisar una corrida ya cerrada, "
                "sin volver a pasar el modelo."
            )
        )
        self.btn_load_run.clicked.connect(self._run_load)

        rv.addWidget(self.btn_load_run)

        rv.addWidget(HLine())

        # ── Calibration ──

        lbl_calib = QLabel(tr("Calibración μm/píxel"))

        lbl_calib.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_calib)

        row_calib = QHBoxLayout()

        self.btn_line = QPushButton(tr("Línea"))
        self.btn_line.setIcon(iconos.icon("linea", 15, T.INK2))

        self.btn_circ = QPushButton(tr("Círculo"))
        self.btn_circ.setIcon(iconos.icon("circulo", 15, T.INK2))

        self.btn_line.setCheckable(True)

        self.btn_circ.setCheckable(True)

        self.btn_line.clicked.connect(lambda: self._start_calib("linea"))

        self.btn_circ.clicked.connect(lambda: self._start_calib("circulo"))

        row_calib.addWidget(self.btn_line)

        row_calib.addWidget(self.btn_circ)

        rv.addLayout(row_calib)

        self.lbl_calib_hint = QLabel(tr("Haz clic en la imagen para marcar\npuntos de referencia."))

        self.lbl_calib_hint.setStyleSheet(f"color: {T.INK3}; font-size: 8.5pt;")

        self.lbl_calib_hint.setVisible(False)

        rv.addWidget(self.lbl_calib_hint)

        btn_cancel_calib = QPushButton(tr("✕  Cancelar calibración"))

        btn_cancel_calib.setStyleSheet(
            f"QPushButton {{ color: {T.WARN_TX}; border-color: {T.WARN_TX}; }}"
            f"QPushButton:hover {{ background: #fff8c5; }}"
        )

        btn_cancel_calib.clicked.connect(self.state.cancel_calib)

        self.btn_cancel_calib = btn_cancel_calib

        btn_cancel_calib.setVisible(False)

        rv.addWidget(btn_cancel_calib)

        self.lbl_calib_result = QLabel(tr("Sin calibrar"))

        self.lbl_calib_result.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt;")

        rv.addWidget(self.lbl_calib_result)

        rv.addWidget(HLine())

        # ── Results ──

        lbl_res = QLabel(tr("Resultados"))

        lbl_res.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_res)

        rv.addWidget(QLabel(tr("Filtrar por clase:")))

        self.combo_filter = QComboBox()

        self.combo_filter.addItem(tr("Todas las clases"), None)

        self.combo_filter.currentIndexChanged.connect(self._on_filter_changed)

        rv.addWidget(self.combo_filter)

        self.lbl_det_count = QLabel(tr("0 detecciones"))

        self.lbl_det_count.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        rv.addWidget(self.lbl_det_count)

        self.lbl_morph = QLabel("")

        self.lbl_morph.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; font-weight: 600;")

        rv.addWidget(self.lbl_morph)

        # One row per particle, with its NUMBER in front: it is the same number

        # drawn on the image, so one can go from the row to the particle.

        self.table = QTableWidget(0, 6)

        self.table.setHorizontalHeaderLabels(["#", "Clase", "Tipo", "Largo μm", "Ancho μm", "Asp."])

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.table.verticalHeader().setDefaultSectionSize(22)

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        self.table.setSelectionMode(QTableWidget.SingleSelection)

        self.table.setAlternatingRowColors(True)

        self.table.setFixedHeight(200)

        self.table.setStyleSheet("font-size: 9pt;")

        self.table.itemSelectionChanged.connect(self._on_particle_chosen)

        rv.addWidget(self.table)

        # ── Selected particle details ──

        # Here it is shown WHAT was measured: the mask contour and, on top of it, the

        # Feret line or the geodesic path. A size that cannot be seen

        # measured cannot be verified, and that is what a reviewer asks for.

        self.lbl_record_tit = QLabel(tr("Selecciona una partícula para ver su medición"))

        self.lbl_record_tit.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        self.lbl_record_tit.setWordWrap(True)

        rv.addWidget(self.lbl_record_tit)

        self.lbl_record = QLabel()

        self.lbl_record.setAlignment(Qt.AlignCenter)

        self.lbl_record.setMinimumHeight(150)

        self.lbl_record.setStyleSheet(
            "background: #0d1117; border: 1px solid #d0d7de; border-radius: 6px;"
        )

        rv.addWidget(self.lbl_record)

        self.lbl_record_data = QLabel("")

        self.lbl_record_data.setWordWrap(True)

        self.lbl_record_data.setStyleSheet(
            f"color: {T.INK3}; font-size: 9pt; font-family: Consolas, monospace;"
        )

        rv.addWidget(self.lbl_record_data)

        rv.addWidget(HLine())

        # ── Save ──

        btn_save = QPushButton(tr("Guardar info actual"))
        btn_save.setIcon(iconos.icon("guardar", 15, T.INK2))

        btn_save.setObjectName("primary")

        btn_save.clicked.connect(self._save)

        rv.addWidget(btn_save)

        self.lbl_status = QLabel(tr("● Listo"))

        self.lbl_status.setAlignment(Qt.AlignCenter)

        self.lbl_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 9pt;")

        rv.addWidget(self.lbl_status)

        rv.addStretch(1)

    # ── Connections and shortcuts ────────────────────────────

    def _connect_signals(self):

        self.state.image_loaded.connect(self._on_image_loaded)

        self.state.detections_changed.connect(self._on_detections_changed)

        self.state.calib_changed.connect(self._on_calib_changed)

        self.state.calib_pts_changed.connect(self._on_calib_pts_changed)

        self.state.calib_ready.connect(self._on_calib_ready)

    def _install_shortcuts(self):

        QShortcut(QKeySequence("Ctrl+D"), self, self._detect)

        QShortcut(QKeySequence(Qt.Key_Left), self, self.state.prev_image)

        QShortcut(QKeySequence(Qt.Key_Right), self, self.state.next_image)

        QShortcut(QKeySequence(Qt.Key_Escape), self, self.state.cancel_calib)

    # ── Open files ───────────────────────────────────────────

    def _open_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", "", "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )

        if path:
            self.state.load_single(Path(path))

            self.nav_frame.setVisible(False)

    def _open_folder(self):

        folder = QFileDialog.getExistingDirectory(self, "Abrir carpeta de imágenes")

        if folder:
            self.state.load_folder(Path(folder))

            total = len(self.state.images)

            if total > 1:
                self.nav_frame.setVisible(True)

                self._update_nav()

    def _on_image_loaded(self, path: Path):

        self.lbl_img_name.setText(path.name)

        self._update_nav()

        self._update_status_bar()

        self._set_status("● Imagen cargada", T.OK)

    def _update_nav(self):

        total = len(self.state.images)

        idx = self.state.current_idx

        self.lbl_nav.setText(f"{idx + 1} / {total}")

    # ── Model and detection ──────────────────────────────────

    def _load_model(self):

        path, _ = QFileDialog.getOpenFileName(self, "Cargar modelo YOLO", "", "Modelo YOLO (*.pt)")

        if not path:
            return

        try:
            from polyx.core.detect.yolo import YoloModel

            self.state.model = YoloModel(path)

            self.lbl_model.setText(Path(path).name)

            self._set_status("● Modelo listo", T.OK)

        except Exception as e:
            QMessageBox.warning(self, tr("Error al cargar modelo"), str(e))

    def _detect(self):

        if not self.state.model:
            QMessageBox.warning(self, tr("Sin modelo"), tr("Carga un modelo .pt primero."))

            return

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, tr("Sin imagen"), tr("Abre una imagen primero."))

            return

        imgsz = int(round(self.spin_imgsz.value() / 32)) * 32  # multiple of stride

        device = "0" if (self.chk_gpu.isChecked() and _there_are_gpu()) else "cpu"

        try:
            self._set_status(f"● Detectando (imgsz {imgsz}, {device})…", T.ACCENT)

            QApplication.setOverrideCursor(Qt.WaitCursor)

            QApplication.processEvents()

            # Automatic tiling: the Viewer is used on the full plate, which is

            # exactly the case where a single pass makes the particles disappear.

            plan: dict = {}

            dets = self.state.model.predict_auto(
                str(img),
                conf=self.spin_conf.value(),
                imgsz=imgsz,
                device=device,
                tiling="auto",
                threshold_px=2000,
                record=plan,
            )

            self.state.detections = dets

            self.state.detections_changed.emit(dets)

            if plan.get("troceado"):
                p = plan["plan"]

                self._set_status(
                    f"● {len(dets)} detecciones (troceada en {p['n_tiles']} tiles "
                    f"de {p['tile']} px, imgsz {imgsz})",
                    T.OK,
                )

            else:
                self._set_status(f"● {len(dets)} detecciones (imgsz {imgsz})", T.OK)

        except Exception as e:
            # Without enough VRAM the failure is common when raising imgsz: it says

            # what to do instead of showing the raw exception.

            msg = str(e)

            if "out of memory" in msg.lower():
                msg = (
                    f"Sin memoria de GPU a imgsz {imgsz}.\n\n"
                    f"Baja la resolución de inferencia o desmarca «Usar GPU»."
                )

            QMessageBox.warning(self, tr("Error en detección"), msg)

            self._set_status("● Error en detección", T.ERR)

        finally:
            QApplication.restoreOverrideCursor()

    def _run_load(self):
        """Loads into the Viewer the predictions a Detector run left behind.

        It is needed because the Detector's `Revisar` button only works
        while the run is still in memory: closing it loses the results. On
        disk they do remain, in runs/detect_*/<modelo>/labels/, so a run from
        last week can be reviewed again without repeating it.

        The ORIGINAL PHOTO is opened, never the run's annotated PNG: it has
        the boxes painted on top and measuring on it would put the rectangle
        stroke inside the particle's mask.
        """

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, tr("Sin imagen"), tr("Abre primero la foto original."))

            return

        folder_2 = QFileDialog.getExistingDirectory(
            self,
            tr("Carpeta de la corrida (runs/detect_...)"),
            str(Path(__file__).resolve().parents[4] / "runs"),
        )

        if not folder_2:
            return

        root_2 = Path(folder_2)

        # The runner names each file NNNN_<photo name>.txt, with an

        # index in front so that two homonymous photos from different folders do not

        # collide. That is why it searches by suffix and not by exact name.

        candidates = sorted(root_2.rglob(f"*_{img.stem}.txt"))

        candidates = [c for c in candidates if c.parent.name == "labels"]

        if not candidates:
            QMessageBox.information(
                self,
                tr("Sin predicciones"),
                tr(
                    "En esa corrida no hay predicciones para {n}.\n\n"
                    "Se buscó en las subcarpetas 'labels'."
                ).format(n=img.name),
            )

            return

        if len(candidates) > 1:
            # Several models ran on the same photo; choose which one.

            from PySide6.QtWidgets import QInputDialog

            options = [c.parent.parent.name for c in candidates]

            chosen, ok = QInputDialog.getItem(
                self,
                tr("Varios modelos"),
                tr("Esta foto se analizó con varios modelos. ¿Cuál revisar?"),
                options,
                0,
                False,
            )

            if not ok:
                return

            candidates = [c for c, o in zip(candidates, options) if o == chosen]

        txt = candidates[0]

        try:
            import cv2
            import numpy as np

            from polyx.core.detect.yolo import read_yolo_txt

            bgr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)

            if bgr is None:
                raise OSError("no se pudo leer la imagen")

            height, width_2 = bgr.shape[:2]

            names = {i: n for i, n in enumerate(self._class_names(txt.parent))}

            dets = read_yolo_txt(txt, width_2, height, names)

        except Exception as e:
            QMessageBox.warning(self, tr("Error"), f"No se pudieron leer las predicciones: {e}")

            return

        if not dets:
            QMessageBox.information(
                self,
                tr("Sin predicciones"),
                tr("El archivo está vacío: esa foto no tuvo detecciones."),
            )

            return

        self.state.detections = dets

        self.state.detections_changed.emit(dets)

        self._set_status(
            tr("● {n} predicciones cargadas de {c}").format(
                n=len(dets), c=txt.parent.parent.parent.name
            ),
            T.OK,
        )

    def _load_labels(self):
        """Loads the YOLO .txt of the current image and shows it as detections.

        It allows the manual count to be reviewed on the full plate and the
        sizes in µm to be seen without going back to the Labeller.
        """

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, tr("Sin imagen"), tr("Abre una imagen primero."))

            return

        txt = img.with_suffix(".txt")

        if not txt.exists():
            QMessageBox.information(
                self,
                tr("Sin etiquetas"),
                f"No hay archivo de etiquetas junto a la imagen:\n{txt.name}",
            )

            return

        import cv2
        import numpy as np

        from polyx.core.detect.yolo import Detection

        arr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)

        if arr is None:
            QMessageBox.warning(self, tr("Error"), tr("No se pudo leer la imagen."))

            return

        height, width_2 = arr.shape[:2]

        names = self._class_names(txt.parent)

        dets, malformed = [], 0

        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()

            if len(parts) < 5:
                if line.strip():
                    malformed += 1

                continue

            try:
                k = int(parts[0])

                cx, cy, w, h = (float(v) for v in parts[1:5])

            except ValueError:
                malformed += 1

                continue

            dets.append(
                Detection(
                    class_id=k,
                    class_name=names[k] if 0 <= k < len(names) else str(k),
                    conf=1.0,  # manual annotation: total certainty
                    x1=(cx - w / 2) * width_2,
                    y1=(cy - h / 2) * height,
                    x2=(cx + w / 2) * width_2,
                    y2=(cy + h / 2) * height,
                )
            )

        self.state.detections = dets

        self.state.detections_changed.emit(dets)

        warning_2 = f"  ({malformed} línea(s) ilegible(s))" if malformed else ""

        self._set_status(
            f"● {len(dets)} etiquetas cargadas de {txt.name}{warning_2}",
            T.WARN if malformed else T.OK,
        )

    @staticmethod
    def _class_names(folder_2: Path) -> list[str]:
        """Reads classes.txt from the folder; if it is not there, it uses the study's classes."""

        f = folder_2 / "classes.txt"

        if f.exists():
            names = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]

            if names:
                return names

        return ["PET", "PP", "LDPE"]

    def _on_detections_changed(self, dets: list):

        self._measure_shapes(dets)

        self._rebuild_table(dets)

        self._update_filter_combo(dets)

        self.lbl_det_count.setText(f"{len(dets)} detección(es)")

        self._update_status_bar()

    def _measure_shapes(self, dets: list):
        """Numbers and measures the real shape of each particle.

        It is done here and not when drawing the table because measuring
        costs about 6 ms per particle and the table is rebuilt every time a
        filter changes.
        """

        self._bgr_current = None

        if not dets:
            self.lbl_morph.setText("")

            return

        path_2 = self.state.current_image

        if path_2 is None:
            return

        try:
            from polyx.core.calibration import read_image
            from polyx.core.morphology import apply_to_detection

            bgr = read_image(path_2)

        except Exception:
            bgr = None

        if bgr is None:
            return

        self._bgr_current = bgr

        um = self.state.um_per_px if self.state.um_per_px > 0 else None

        # Top to bottom and left to right: the same criterion the

        # Detector uses, so that the number means the same in both modules.

        for i, d in enumerate(sorted(dets, key=lambda z: (round(z.y1 / 40), z.x1)), 1):
            d.number = i

            apply_to_detection(d, bgr, um)

        fib = sum(1 for d in dets if d.morphotype == "fibra")

        sin = sum(1 for d in dets if d.morphotype is None)

        txt = f"{fib} fibra(s) · {len(dets) - fib - sin} fragmento(s)"

        if sin:
            txt += f" · {sin} sin medir"

        if not um:
            txt += "   (sin calibrar: tallas solo en píxeles)"

        self.lbl_morph.setText(txt)

    def _rebuild_table(self, dets: list):

        self.table.setRowCount(0)

        for det in sorted(dets, key=lambda z: z.number or 0):
            row = self.table.rowCount()

            self.table.insertRow(row)

            it = QTableWidgetItem(str(det.number or row + 1))

            # The Detection travels inside the item: when selecting the row one must

            # be able to get back to it without looking it up by index, which drifts

            # as soon as the class filter is applied.

            it.setData(Qt.UserRole, det)

            self.table.setItem(row, 0, it)

            self.table.setItem(row, 1, QTableWidgetItem(det.class_name))

            self.table.setItem(row, 2, QTableWidgetItem(det.morphotype or "—"))

            self.table.setItem(
                row, 3, QTableWidgetItem(f"{det.length_um:.0f}" if det.length_um else "—")
            )

            self.table.setItem(
                row, 4, QTableWidgetItem(f"{det.width_um:.0f}" if det.width_um else "—")
            )

            self.table.setItem(row, 5, QTableWidgetItem(f"{det.aspect:.1f}" if det.aspect else "—"))

    def _on_particle_chosen(self):
        """Draws the measurement of the selected particle in the table."""

        rows = self.table.selectionModel().selectedRows()

        if not rows or getattr(self, "_bgr_current", None) is None:
            return

        item = self.table.item(rows[0].row(), 0)

        det = item.data(Qt.UserRole) if item else None

        if det is None:
            return

        try:
            from polyx.core.morphology import draw_measurement

            img, m = draw_measurement(self._bgr_current, det, zoom=6)

        except Exception:
            img, m = None, None

        if img is None or m is None or not m.ok:
            self.lbl_record.setText(tr("No se pudo aislar esta partícula del fondo"))

            self.lbl_record_data.setText("")

            return

        from PySide6.QtGui import QImage, QPixmap

        height, width_2 = img.shape[:2]

        qim = QImage(img.data, width_2, height, 3 * width_2, QImage.Format_BGR888).copy()

        pix = QPixmap.fromImage(qim)

        available = self.lbl_record.width() - 12

        if available > 0 and pix.width() > available:
            pix = pix.scaledToWidth(available, Qt.SmoothTransformation)

        self.lbl_record.setPixmap(pix)

        self.lbl_record_tit.setText(
            f"Partícula #{det.number} · {det.class_name} · {det.morphotype or '—'}"
            "   —   izquierda: sin marcas · derecha: verde = contorno medido, "
            "amarillo = Feret, magenta = geodésico"
        )

        um = self.state.um_per_px

        lines = [
            f"medido por  {m.method}",
            f"largo       {m.length_px:8.1f} px"
            + (f"  x {um:.4f} um/px = {det.length_um:.0f} um" if um > 0 and det.length_um else ""),
            f"ancho       {m.width_px:8.1f} px"
            + (f"  = {det.width_um:.0f} um" if um > 0 and det.width_um else ""),
            f"Feret       {m.feret_px:8.1f} px",
            f"geodesico   {m.geodesic_px:8.1f} px",
            f"aspecto     {m.aspect:8.1f}   curvatura {m.curvature:.2f}",
        ]

        if m.warning:
            lines.append(f"aviso: {m.warning}")

        self.lbl_record_data.setText("\n".join(lines))

    def _update_filter_combo(self, dets: list):

        self.combo_filter.blockSignals(True)

        prev = self.combo_filter.currentData()

        self.combo_filter.clear()

        self.combo_filter.addItem(tr("Todas las clases"), None)

        for _n in sorted({d.class_name for d in dets}):
            self.combo_filter.addItem(_n, _n)

        idx = self.combo_filter.findData(prev)

        if idx >= 0:
            self.combo_filter.setCurrentIndex(idx)

        self.combo_filter.blockSignals(False)

    def _on_filter_changed(self, _idx: int):

        # The class name goes in userData: if it were the shown text, translating

        # `Todas las clases` would leave the filter checking against a string that

        # would cease to exist and nothing would be shown again.

        self.canvas.set_filter_class(self.combo_filter.currentData())

    # ── Calibration ──────────────────────────────────────────

    def _start_calib(self, mode: str):

        self.state.start_calib(mode)

        self.btn_line.setChecked(mode == "linea")

        self.btn_circ.setChecked(mode == "circulo")

        self.lbl_calib_hint.setVisible(True)

        self.btn_cancel_calib.setVisible(True)

        if mode == "linea":
            self.lbl_calib_hint.setText(tr("Haz clic en 2 puntos sobre\nuna referencia conocida."))

        else:
            self.lbl_calib_hint.setText(
                tr("Haz clic en 3 puntos del borde\nde un objeto circular conocido.")
            )

        self._set_status("● Modo calibración activo", T.WARN)

    def _on_calib_pts_changed(self, pts: list):

        if not pts:
            self.btn_line.setChecked(False)

            self.btn_circ.setChecked(False)

            self.lbl_calib_hint.setVisible(False)

            self.btn_cancel_calib.setVisible(False)

        mode = self.state.calib_mode

        needed = 2 if mode == "linea" else 3

        self.lbl_calib_hint.setText(
            f"Haz clic en {needed - len(pts)} punto(s) más." if pts else self.lbl_calib_hint.text()
        )

    def _on_calib_ready(self):
        """Enough points collected → ask the user for the real size.

        The positional arguments are mandatory: PySide6 does not accept
        ``min`` nor ``max`` as keywords in ``getDouble`` and it raised
        AttributeError, which left the calibration unusable.
        """

        mode = self.state.calib_mode

        if mode == "linea":
            val, ok = QInputDialog.getDouble(
                self,
                "Calibración — Línea",
                "Tamaño real de la línea marcada (μm):",
                1000.0,
                0.001,
                1_000_000.0,
                3,
            )

            if ok and val > 0:
                self.state.finish_calib_line(val)

            else:
                self.state.cancel_calib()

        elif mode == "circulo":
            # 100 mm by default: it is the diameter of the Petri dishes in the study

            val, ok = QInputDialog.getDouble(
                self,
                "Calibración — Círculo",
                "Diámetro real del objeto circular (μm):\nPlaca Petri de 100 mm → 100000",
                100_000.0,
                0.001,
                10_000_000.0,
                3,
            )

            if ok and val > 0:
                self.state.finish_calib_circle(val)

            else:
                self.state.cancel_calib()

    def _on_calib_changed(self, um_per_px: float, mode_name: str):

        self.lbl_calib_result.setText(f"{um_per_px:.4f} μm/px  ({mode_name})")

        self.lbl_calib_result.setStyleSheet(
            f"color: {T.OK_TX}; font-size: 9.5pt; font-weight: 600;"
        )

        self._set_status(f"● Calibración aplicada: {um_per_px:.4f} μm/px", T.OK)

        self._update_status_bar()

        # Refresh the table with the new μm units

        if self.state.detections:
            self._rebuild_table(self.state.detections)

    def _update_status_bar(self):

        img = self.state.current_image

        img_txt = img.name if img else "Sin imagen"

        if self.state.um_per_px > 0:
            calib_txt = f"{self.state.um_per_px:.4f} μm/px"

        else:
            calib_txt = "Sin calibración"

        n_det = len(self.state.detections)

        det_txt = f"  —  {n_det} det." if n_det else ""

        self.status_bar.setText(f"  {img_txt}  —  {calib_txt}{det_txt}")

    # ── Save ───────────────────────────────────────────────

    def _save(self):

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, tr("Sin imagen"), tr("No hay imagen abierta."))

            return

        # Choose the destination folder

        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta de salida")

        if not out_dir:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = Path(out_dir) / f"visor_{img.stem}_{ts}"

        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._save_annotated_image(img, save_dir)

            self._save_csv(save_dir)

            self._save_json(img, save_dir)

            self._set_status(f"● Guardado en {save_dir.name}", T.OK)

            QMessageBox.information(self, tr("Guardado"), f"Archivos guardados en:\n{save_dir}")

        except Exception as e:
            QMessageBox.warning(self, tr("Error al guardar"), str(e))

            self._set_status("● Error al guardar", T.ERR)

    def _save_annotated_image(self, img: Path, save_dir: Path):

        import cv2
        import numpy as np

        # imdecode and not imread: on Windows imread fails with paths that contain

        # accents or non-ASCII characters, and returns None without warning.

        frame = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return

        CLASS_COLORS_BGR = [
            (47, 52, 227),  # PET  → red
            (0, 140, 255),  # PP   → orange
            (0, 215, 255),  # LDPE → yellow
            (216, 180, 0),
            (228, 202, 72),
            (182, 119, 0),
            (138, 62, 2),
            (139, 45, 118),
        ]

        for det in self.state.detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            color = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            w_px = det.x2 - det.x1

            h_px = det.y2 - det.y1

            diam_px = math.sqrt(w_px * h_px)

            um = self.state.px_to_um(diam_px)

            label = (
                f"{det.class_name} {um:.1f}um"
                if um is not None
                else f"{det.class_name} {diam_px:.0f}px"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        out_path = save_dir / f"{img.stem}_anotada.png"

        ok, buf = cv2.imencode(".png", frame)

        if ok:
            buf.tofile(str(out_path))

    def _save_csv(self, save_dir: Path):

        import csv

        out = save_dir / "detecciones.csv"

        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            w.writerow(["clase", "conf", "x1", "y1", "x2", "y2", "diam_px", "diam_um", "area_um2"])

            for det in self.state.detections:
                wp = det.x2 - det.x1

                hp = det.y2 - det.y1

                diam_px = math.sqrt(wp * hp)

                um = self.state.px_to_um(diam_px)

                area_um2 = self.state.um_per_px**2 * wp * hp if self.state.um_per_px > 0 else ""

                w.writerow(
                    [
                        det.class_name,
                        f"{det.conf:.4f}",
                        f"{det.x1:.1f}",
                        f"{det.y1:.1f}",
                        f"{det.x2:.1f}",
                        f"{det.y2:.1f}",
                        f"{diam_px:.2f}",
                        f"{um:.2f}" if um is not None else "",
                        f"{area_um2:.2f}" if area_um2 != "" else "",
                    ]
                )

    def _save_json(self, img: Path, save_dir: Path):

        um = self.state.um_per_px

        summary = {
            "imagen": img.name,
            "timestamp": datetime.now().isoformat(),
            "um_per_px": um if um > 0 else None,
            "n_detecciones": len(self.state.detections),
            "por_clase": {},
        }

        for det in self.state.detections:
            entry = summary["por_clase"].setdefault(det.class_name, {"n": 0})

            entry["n"] += 1

        out = save_dir / "resumen.json"

        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Drag & Drop ───────────────────────────────────────────

    def dragEnterEvent(self, ev):

        if ev.mimeData().hasUrls():
            for u in ev.mimeData().urls():
                p = u.toLocalFile().lower()

                if p.endswith(".pt") or any(
                    p.endswith(e) for e in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
                ):
                    ev.acceptProposedAction()

                    return

        ev.ignore()

    def dropEvent(self, ev):

        for url in ev.mimeData().urls():
            path = Path(url.toLocalFile())

            if path.suffix.lower() == ".pt":
                try:
                    from polyx.core.detect.yolo import YoloModel

                    self.state.model = YoloModel(str(path))

                    self.lbl_model.setText(path.name)

                    self._set_status("● Modelo cargado", T.OK)

                except Exception as e:
                    self._set_status(f"● Error: {e}", T.ERR)

            elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
                self.state.load_single(path)

                self.nav_frame.setVisible(False)

        ev.acceptProposedAction()

    def _set_status(self, text: str, color: str = T.OK):

        self.lbl_status.setText(text)

        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt;")
