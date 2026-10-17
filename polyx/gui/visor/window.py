"""Ventana principal del Visor — canvas (izq.) | panel de control (der.)."""

from __future__ import annotations

import json

import math

from datetime import datetime

from pathlib import Path



from PySide6.QtCore import Qt

from PySide6.QtGui import QKeySequence, QShortcut

from PySide6.QtWidgets import (

    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton,

    QLabel, QFileDialog, QMessageBox, QInputDialog, QComboBox,

    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,

    QDoubleSpinBox, QProgressDialog, QSizePolicy, QSpinBox, QCheckBox,

    QApplication,

)



from polyx.gui import theme as T
from polyx.gui import marca
from polyx.gui import iconos

from polyx.gui.widgets import LogoBadge, HLine

from .state import VisorState

from .canvas import VisorCanvas

from polyx.model.i18n import tr





def _hay_gpu() -> bool:

    """CUDA disponible. Se consulta perezosamente: importar torch tarda."""

    try:

        import torch

        return bool(torch.cuda.is_available())

    except Exception:

        return False





class VisorWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        marca.titular(self, "visor")

        self.resize(1420, 900)

        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")



        self.state = VisorState()



        central = QWidget()

        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        root.setContentsMargins(0, 0, 0, 0)

        root.setSpacing(0)



        # Canvas (centro/izquierda)

        canvas_wrap = QWidget()

        canvas_wrap.setStyleSheet(f"background: {T.BG_SOFT};")

        cw_lay = QVBoxLayout(canvas_wrap)

        cw_lay.setContentsMargins(0, 0, 0, 0)

        cw_lay.setSpacing(0)



        self.canvas = VisorCanvas(self.state)

        cw_lay.addWidget(self.canvas, 1)



        # Barra inferior de estado/calibración

        self.status_bar = QLabel(tr("  Sin imagen  —  Sin calibración"))

        self.status_bar.setFixedHeight(28)

        self.status_bar.setStyleSheet(

            f"background: {T.BG}; border-top: 1px solid {T.RULE}; "

            f"color: {T.INK2}; font-size: 9.5pt; padding: 0 12px;"

        )

        cw_lay.addWidget(self.status_bar)



        root.addWidget(canvas_wrap, 1)



        # Panel derecho

        right_scroll = QScrollArea()

        right_scroll.setFixedWidth(280)

        right_scroll.setWidgetResizable(True)

        right_scroll.setStyleSheet(

            f"QScrollArea {{ border: none; border-left: 1px solid {T.RULE}; "

            f"background: {T.BG}; }}"

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



    # ── Panel derecho ─────────────────────────────────────────

    def _build_right_panel(self):

        rv = self._rv



        rv.addWidget(LogoBadge("POLY-X", "Visor"))

        rv.addSpacing(4)



        # ── Abrir imagen / carpeta ──

        lbl_open = QLabel(tr("Imagen"))

        lbl_open.setStyleSheet(f"font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_open)



        row_open = QHBoxLayout()

        btn_img = QPushButton(tr("Imagen"))
        btn_img.setIcon(iconos.icono("imagenes", 15, T.INK2))

        btn_fol = QPushButton(tr("Carpeta"))
        btn_fol.setIcon(iconos.icono("dataset", 15, T.INK2))

        btn_img.clicked.connect(self._open_image)

        btn_fol.clicked.connect(self._open_folder)

        row_open.addWidget(btn_img)

        row_open.addWidget(btn_fol)

        rv.addLayout(row_open)



        self.lbl_img_name = QLabel(tr("Sin imagen"))

        self.lbl_img_name.setWordWrap(True)

        self.lbl_img_name.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        rv.addWidget(self.lbl_img_name)



        # Navegación (visible solo con carpeta)

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



        # ── Modelo ──

        lbl_mod = QLabel(tr("Modelo"))

        lbl_mod.setStyleSheet(f"font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_mod)



        self.lbl_model = QLabel(tr("Sin modelo"))

        self.lbl_model.setWordWrap(True)

        self.lbl_model.setStyleSheet(

            f"color: {T.INK3}; font-size: 9pt; background: {T.BG_SOFT}; "

            f"border: 1px solid {T.RULE}; border-radius: 4px; padding: 4px;"

        )

        rv.addWidget(self.lbl_model)



        btn_load_model = QPushButton(tr("Cargar modelo…"))
        btn_load_model.setIcon(iconos.icono("dataset", 15, T.INK2))

        btn_load_model.clicked.connect(self._load_model)

        rv.addWidget(btn_load_model)



        rv.addWidget(QLabel(tr("Confianza mínima:")))

        self.spin_conf = QDoubleSpinBox()

        self.spin_conf.setRange(0.01, 0.99)

        self.spin_conf.setSingleStep(0.05)

        self.spin_conf.setValue(0.25)

        self.spin_conf.setDecimals(2)

        rv.addWidget(self.spin_conf)



        # Resolución de inferencia: decisiva con partículas diminutas. A 640 px

        # una partícula de 12 px en una foto de 4096 colapsa a ~2 px y no se

        # detecta nada. Debe poder subirse.

        rv.addWidget(QLabel(tr("Resolución de inferencia (px):")))

        self.spin_imgsz = QSpinBox()

        self.spin_imgsz.setRange(320, 8192)

        self.spin_imgsz.setSingleStep(32)

        self.spin_imgsz.setValue(2080)

        self.spin_imgsz.setToolTip(

            tr("Lado mayor al que se redimensiona la imagen para inferir.\n"

            "Más alto = partículas más grandes para la red, pero más memoria.\n"

            "Se redondea al múltiplo de 32 más cercano.")

        )

        rv.addWidget(self.spin_imgsz)



        self.chk_gpu = QCheckBox(tr("Usar GPU si está disponible"))

        self.chk_gpu.setChecked(True)

        rv.addWidget(self.chk_gpu)



        self.btn_detect = QPushButton(tr("Detectar"))
        self.btn_detect.setIcon(iconos.icono("ejecutar", 15, T.INK2))

        self.btn_detect.setObjectName("primary")

        self.btn_detect.clicked.connect(self._detect)

        rv.addWidget(self.btn_detect)



        self.btn_cargar_txt = QPushButton(tr("Cargar etiquetas (.txt)"))
        self.btn_cargar_txt.setIcon(iconos.icono("texto", 15, T.INK2))

        self.btn_cargar_txt.setToolTip(

            tr("Muestra las anotaciones YOLO que acompañan a la imagen.\n"

            "Sirve para revisar el conteo manual sobre la placa, con las\n"

            "tallas ya convertidas a µm por la calibración.")

        )

        self.btn_cargar_txt.clicked.connect(self._cargar_etiquetas)

        rv.addWidget(self.btn_cargar_txt)



        self.btn_cargar_run = QPushButton(tr("Cargar predicciones de una corrida"))
        self.btn_cargar_run.setIcon(iconos.icono("dataset", 15, T.INK2))

        self.btn_cargar_run.setToolTip(
            tr("Lee las predicciones que el Detector dejó en runs/detect_.../ "
            "para esta misma foto. Sirve para revisar una corrida ya cerrada, "
            "sin volver a pasar el modelo.")
        )
        self.btn_cargar_run.clicked.connect(self._cargar_de_corrida)

        rv.addWidget(self.btn_cargar_run)



        rv.addWidget(HLine())



        # ── Calibración ──

        lbl_calib = QLabel(tr("Calibración μm/píxel"))

        lbl_calib.setStyleSheet(f"font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_calib)



        row_calib = QHBoxLayout()

        self.btn_linea = QPushButton(tr("Línea"))
        self.btn_linea.setIcon(iconos.icono("linea", 15, T.INK2))

        self.btn_circ = QPushButton(tr("Círculo"))
        self.btn_circ.setIcon(iconos.icono("circulo", 15, T.INK2))

        self.btn_linea.setCheckable(True)

        self.btn_circ.setCheckable(True)

        self.btn_linea.clicked.connect(lambda: self._start_calib("linea"))

        self.btn_circ.clicked.connect(lambda: self._start_calib("circulo"))

        row_calib.addWidget(self.btn_linea)

        row_calib.addWidget(self.btn_circ)

        rv.addLayout(row_calib)



        self.lbl_calib_hint = QLabel(

            tr("Haz clic en la imagen para marcar\npuntos de referencia.")

        )

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



        # ── Resultados ──

        lbl_res = QLabel(tr("Resultados"))

        lbl_res.setStyleSheet(f"font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_res)



        rv.addWidget(QLabel(tr("Filtrar por clase:")))

        self.combo_filter = QComboBox()

        self.combo_filter.addItem(tr("Todas las clases"), None)

        self.combo_filter.currentIndexChanged.connect(self._on_filter_changed)

        rv.addWidget(self.combo_filter)



        self.lbl_det_count = QLabel(tr("0 detecciones"))

        self.lbl_det_count.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        rv.addWidget(self.lbl_det_count)



        self.lbl_morfo = QLabel("")

        self.lbl_morfo.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; font-weight: 600;")

        rv.addWidget(self.lbl_morfo)



        # Una fila por partícula, con su NÚMERO delante: es el mismo que lleva

        # dibujado en la imagen, así que se puede ir de la fila a la partícula.

        self.table = QTableWidget(0, 6)

        self.table.setHorizontalHeaderLabels(

            ["#", "Clase", "Tipo", "Largo μm", "Ancho μm", "Asp."])

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.table.verticalHeader().setDefaultSectionSize(22)

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        self.table.setSelectionMode(QTableWidget.SingleSelection)

        self.table.setAlternatingRowColors(True)

        self.table.setFixedHeight(200)

        self.table.setStyleSheet(f"font-size: 9pt;")

        self.table.itemSelectionChanged.connect(self._on_particula_elegida)

        rv.addWidget(self.table)



        # ── Ficha de la partícula seleccionada ──

        # Aquí se ve SOBRE QUÉ se midió: el contorno de la máscara y, encima, la

        # recta de Feret o el camino geodésico. Una talla que no se puede ver

        # medida no se puede verificar, y eso es lo que pide un revisor.

        self.lbl_ficha_tit = QLabel(tr("Selecciona una partícula para ver su medición"))

        self.lbl_ficha_tit.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        self.lbl_ficha_tit.setWordWrap(True)

        rv.addWidget(self.lbl_ficha_tit)



        self.lbl_ficha = QLabel()

        self.lbl_ficha.setAlignment(Qt.AlignCenter)

        self.lbl_ficha.setMinimumHeight(150)

        self.lbl_ficha.setStyleSheet(

            f"background: #0d1117; border: 1px solid #d0d7de; border-radius: 6px;")

        rv.addWidget(self.lbl_ficha)



        self.lbl_ficha_datos = QLabel("")

        self.lbl_ficha_datos.setWordWrap(True)

        self.lbl_ficha_datos.setStyleSheet(

            f"color: {T.INK3}; font-size: 9pt; font-family: Consolas, monospace;")

        rv.addWidget(self.lbl_ficha_datos)



        rv.addWidget(HLine())



        # ── Guardar ──

        btn_save = QPushButton(tr("Guardar info actual"))
        btn_save.setIcon(iconos.icono("guardar", 15, T.INK2))

        btn_save.setObjectName("primary")

        btn_save.clicked.connect(self._save)

        rv.addWidget(btn_save)



        self.lbl_status = QLabel(tr("● Listo"))

        self.lbl_status.setAlignment(Qt.AlignCenter)

        self.lbl_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 9pt;")

        rv.addWidget(self.lbl_status)



        rv.addStretch(1)



    # ── Conexiones y atajos ───────────────────────────────────

    def _connect_signals(self):

        self.state.image_loaded.connect(self._on_image_loaded)

        self.state.detections_changed.connect(self._on_detections_changed)

        self.state.calib_changed.connect(self._on_calib_changed)

        self.state.calib_pts_changed.connect(self._on_calib_pts_changed)

        self.state.calib_ready.connect(self._on_calib_ready)



    def _install_shortcuts(self):

        QShortcut(QKeySequence("Ctrl+D"), self, self._detect)

        QShortcut(QKeySequence(Qt.Key_Left),  self, self.state.prev_image)

        QShortcut(QKeySequence(Qt.Key_Right), self, self.state.next_image)

        QShortcut(QKeySequence(Qt.Key_Escape), self, self.state.cancel_calib)



    # ── Abrir archivos ────────────────────────────────────────

    def _open_image(self):

        path, _ = QFileDialog.getOpenFileName(

            self, "Abrir imagen", "",

            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"

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



    # ── Modelo y detección ────────────────────────────────────

    def _load_model(self):

        path, _ = QFileDialog.getOpenFileName(

            self, "Cargar modelo YOLO", "", "Modelo YOLO (*.pt)"

        )

        if not path:

            return

        try:

            from polyx.model.yolo_wrap import YoloModel

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

        imgsz = int(round(self.spin_imgsz.value() / 32)) * 32   # múltiplo de stride

        device = "0" if (self.chk_gpu.isChecked() and _hay_gpu()) else "cpu"

        try:

            self._set_status(f"● Detectando (imgsz {imgsz}, {device})…", T.ACCENT)

            QApplication.setOverrideCursor(Qt.WaitCursor)

            QApplication.processEvents()

            # Troceado automático: el Visor se usa sobre la placa completa, que es

            # justo el caso donde un pase único hace desaparecer las partículas.

            plan: dict = {}

            dets = self.state.model.predict_auto(

                str(img), conf=self.spin_conf.value(), imgsz=imgsz, device=device,

                troceo="auto", umbral_px=2000, registro=plan,

            )

            self.state.detections = dets

            self.state.detections_changed.emit(dets)

            if plan.get("troceado"):

                p = plan["plan"]

                self._set_status(

                    f"● {len(dets)} detecciones (troceada en {p['n_tiles']} tiles "

                    f"de {p['tile']} px, imgsz {imgsz})", T.OK)

            else:

                self._set_status(f"● {len(dets)} detecciones (imgsz {imgsz})", T.OK)

        except Exception as e:

            # Sin VRAM suficiente el fallo es habitual al subir imgsz: se dice

            # qué hacer en vez de mostrar la excepción cruda.

            msg = str(e)

            if "out of memory" in msg.lower():

                msg = (f"Sin memoria de GPU a imgsz {imgsz}.\n\n"

                       f"Baja la resolución de inferencia o desmarca «Usar GPU».")

            QMessageBox.warning(self, tr("Error en detección"), msg)

            self._set_status("● Error en detección", T.ERR)

        finally:

            QApplication.restoreOverrideCursor()



    def _cargar_de_corrida(self):

        """Carga en el Visor las predicciones que dejó una corrida del Detector.



        Hace falta porque el botón «Revisar» del Detector solo funciona mientras

        la corrida sigue en memoria: al cerrarlo se pierden los resultados. En

        disco sí quedan, en runs/detect_*/<modelo>/labels/, así que una corrida

        de la semana pasada se puede volver a revisar sin repetirla.



        Se abre la FOTO ORIGINAL, nunca el PNG anotado del run: ese lleva las

        cajas pintadas encima y medir sobre él metería el trazo del rectángulo

        dentro de la máscara de la partícula.

        """

        img = self.state.current_image

        if img is None:

            QMessageBox.warning(self, tr("Sin imagen"),

                                tr("Abre primero la foto original."))

            return

        carpeta = QFileDialog.getExistingDirectory(

            self, tr("Carpeta de la corrida (runs/detect_...)"),

            str(Path(__file__).resolve().parents[3] / "runs"))

        if not carpeta:

            return

        raiz = Path(carpeta)



        # El runner nombra cada archivo como NNNN_<nombre de la foto>.txt, con un

        # índice delante para que dos fotos homónimas de carpetas distintas no se

        # pisen. Por eso se busca por sufijo y no por nombre exacto.

        candidatos = sorted(raiz.rglob(f"*_{img.stem}.txt"))

        candidatos = [c for c in candidatos if c.parent.name == "labels"]

        if not candidatos:

            QMessageBox.information(

                self, tr("Sin predicciones"),

                tr("En esa corrida no hay predicciones para {n}.\n\n"

                   "Se buscó en las subcarpetas 'labels'.").format(n=img.name))

            return

        if len(candidatos) > 1:

            # Varios modelos corrieron sobre la misma foto; se elige cuál.

            from PySide6.QtWidgets import QInputDialog

            opciones = [c.parent.parent.name for c in candidatos]

            elegido, ok = QInputDialog.getItem(

                self, tr("Varios modelos"),

                tr("Esta foto se analizó con varios modelos. ¿Cuál revisar?"),

                opciones, 0, False)

            if not ok:

                return

            candidatos = [c for c, o in zip(candidatos, opciones) if o == elegido]



        txt = candidatos[0]

        try:

            from polyx.model.yolo_wrap import read_yolo_txt

            import cv2

            import numpy as np

            bgr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8),

                               cv2.IMREAD_COLOR)

            if bgr is None:

                raise OSError("no se pudo leer la imagen")

            alto, ancho = bgr.shape[:2]

            nombres = {i: n for i, n in enumerate(

                self._nombres_de_clase(txt.parent))}

            dets = read_yolo_txt(txt, ancho, alto, nombres)

        except Exception as e:

            QMessageBox.warning(self, tr("Error"),

                                f"No se pudieron leer las predicciones: {e}")

            return

        if not dets:

            QMessageBox.information(self, tr("Sin predicciones"),

                                    tr("El archivo está vacío: esa foto no tuvo detecciones."))

            return

        self.state.detections = dets

        self.state.detections_changed.emit(dets)

        self._set_status(

            tr("● {n} predicciones cargadas de {c}").format(

                n=len(dets), c=txt.parent.parent.parent.name), T.OK)



    def _cargar_etiquetas(self):

        """Carga el .txt YOLO de la imagen actual y lo muestra como detecciones.



        Permite revisar el conteo manual sobre la placa completa y ver las

        tallas en µm sin volver al Etiquetador.

        """

        img = self.state.current_image

        if img is None:

            QMessageBox.warning(self, tr("Sin imagen"), tr("Abre una imagen primero."))

            return

        txt = img.with_suffix(".txt")

        if not txt.exists():

            QMessageBox.information(

                self, tr("Sin etiquetas"),

                f"No hay archivo de etiquetas junto a la imagen:\n{txt.name}")

            return



        import cv2

        import numpy as np

        from polyx.model.yolo_wrap import Detection



        arr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)

        if arr is None:

            QMessageBox.warning(self, tr("Error"), tr("No se pudo leer la imagen."))

            return

        alto, ancho = arr.shape[:2]



        nombres = self._nombres_de_clase(txt.parent)

        dets, malformadas = [], 0

        for linea in txt.read_text(encoding="utf-8", errors="ignore").splitlines():

            partes = linea.split()

            if len(partes) < 5:

                if linea.strip():

                    malformadas += 1

                continue

            try:

                k = int(partes[0])

                cx, cy, w, h = (float(v) for v in partes[1:5])

            except ValueError:

                malformadas += 1

                continue

            dets.append(Detection(

                class_id=k,

                class_name=nombres[k] if 0 <= k < len(nombres) else str(k),

                conf=1.0,                       # anotación manual: certeza total

                x1=(cx - w / 2) * ancho, y1=(cy - h / 2) * alto,

                x2=(cx + w / 2) * ancho, y2=(cy + h / 2) * alto,

            ))



        self.state.detections = dets

        self.state.detections_changed.emit(dets)

        aviso = f"  ({malformadas} línea(s) ilegible(s))" if malformadas else ""

        self._set_status(f"● {len(dets)} etiquetas cargadas de {txt.name}{aviso}",

                         T.WARN if malformadas else T.OK)



    @staticmethod

    def _nombres_de_clase(carpeta: Path) -> list[str]:

        """Lee classes.txt de la carpeta; si no está, usa las clases del estudio."""

        f = carpeta / "classes.txt"

        if f.exists():

            nombres = [l.strip() for l in

                       f.read_text(encoding="utf-8").splitlines() if l.strip()]

            if nombres:

                return nombres

        return ["PET", "PP", "LDPE"]



    def _on_detections_changed(self, dets: list):

        self._medir_formas(dets)

        self._rebuild_table(dets)

        self._update_filter_combo(dets)

        self.lbl_det_count.setText(f"{len(dets)} detección(es)")

        self._update_status_bar()



    def _medir_formas(self, dets: list):

        """Numera y mide la forma real de cada partícula.



        Se hace aquí y no al dibujar la tabla porque medir cuesta unos 6 ms por

        partícula y la tabla se reconstruye cada vez que cambia un filtro.

        """

        self._bgr_actual = None

        if not dets:

            self.lbl_morfo.setText("")

            return

        ruta = self.state.current_image

        if ruta is None:

            return

        try:

            from polyx.model.calibracion import leer_imagen

            from polyx.model.morfologia import aplicar_a_deteccion

            bgr = leer_imagen(ruta)

        except Exception:

            bgr = None

        if bgr is None:

            return

        self._bgr_actual = bgr

        um = self.state.um_per_px if self.state.um_per_px > 0 else None

        # De arriba abajo y de izquierda a derecha: el mismo criterio que usa el

        # Detector, para que el número signifique lo mismo en los dos módulos.

        for i, d in enumerate(sorted(dets, key=lambda z: (round(z.y1 / 40), z.x1)), 1):

            d.numero = i

            aplicar_a_deteccion(d, bgr, um)

        fib = sum(1 for d in dets if d.morfotipo == "fibra")

        sin = sum(1 for d in dets if d.morfotipo is None)

        txt = f"{fib} fibra(s) · {len(dets) - fib - sin} fragmento(s)"

        if sin:

            txt += f" · {sin} sin medir"

        if not um:

            txt += "   (sin calibrar: tallas solo en píxeles)"

        self.lbl_morfo.setText(txt)



    def _rebuild_table(self, dets: list):

        self.table.setRowCount(0)

        for det in sorted(dets, key=lambda z: (z.numero or 0)):

            row = self.table.rowCount()

            self.table.insertRow(row)

            it = QTableWidgetItem(str(det.numero or row + 1))

            # La Detection viaja dentro del item: al seleccionar la fila hay que

            # poder volver a ella sin buscarla por índice, que se desincroniza

            # en cuanto se filtra por clase.

            it.setData(Qt.UserRole, det)

            self.table.setItem(row, 0, it)

            self.table.setItem(row, 1, QTableWidgetItem(det.class_name))

            self.table.setItem(row, 2, QTableWidgetItem(det.morfotipo or "—"))

            self.table.setItem(row, 3, QTableWidgetItem(

                f"{det.largo_um:.0f}" if det.largo_um else "—"))

            self.table.setItem(row, 4, QTableWidgetItem(

                f"{det.ancho_um:.0f}" if det.ancho_um else "—"))

            self.table.setItem(row, 5, QTableWidgetItem(

                f"{det.aspecto:.1f}" if det.aspecto else "—"))



    def _on_particula_elegida(self):

        """Dibuja la medición de la partícula seleccionada en la tabla."""

        filas = self.table.selectionModel().selectedRows()

        if not filas or getattr(self, "_bgr_actual", None) is None:

            return

        item = self.table.item(filas[0].row(), 0)

        det = item.data(Qt.UserRole) if item else None

        if det is None:

            return

        try:

            from polyx.model.morfologia import dibujar_medicion

            img, m = dibujar_medicion(self._bgr_actual, det, zoom=6)

        except Exception:

            img, m = None, None

        if img is None or m is None or not m.ok:

            self.lbl_ficha.setText(tr("No se pudo aislar esta partícula del fondo"))

            self.lbl_ficha_datos.setText("")

            return



        from PySide6.QtGui import QImage, QPixmap

        alto, ancho = img.shape[:2]

        qim = QImage(img.data, ancho, alto, 3 * ancho, QImage.Format_BGR888).copy()

        pix = QPixmap.fromImage(qim)

        disponible = self.lbl_ficha.width() - 12

        if disponible > 0 and pix.width() > disponible:

            pix = pix.scaledToWidth(disponible, Qt.SmoothTransformation)

        self.lbl_ficha.setPixmap(pix)

        self.lbl_ficha_tit.setText(

            f"Partícula #{det.numero} · {det.class_name} · {det.morfotipo or '—'}"

            "   —   izquierda: sin marcas · derecha: verde = contorno medido, "

            "amarillo = Feret, magenta = geodésico")

        um = self.state.um_per_px

        lineas = [

            f"medido por  {m.metodo}",

            f"largo       {m.largo_px:8.1f} px"

            + (f"  x {um:.4f} um/px = {det.largo_um:.0f} um" if um > 0 and det.largo_um else ""),

            f"ancho       {m.ancho_px:8.1f} px"

            + (f"  = {det.ancho_um:.0f} um" if um > 0 and det.ancho_um else ""),

            f"Feret       {m.feret_px:8.1f} px",

            f"geodesico   {m.geodesico_px:8.1f} px",

            f"aspecto     {m.aspecto:8.1f}   curvatura {m.curvatura:.2f}",

        ]

        if m.aviso:

            lineas.append(f"aviso: {m.aviso}")

        self.lbl_ficha_datos.setText("\n".join(lineas))



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

        # El nombre de clase va en userData: si fuera el texto visible, traducir

        # «Todas las clases» dejaria el filtro comparando contra una cadena que

        # ya no existe y no volveria a mostrar nada.

        self.canvas.set_filter_class(self.combo_filter.currentData())



    # ── Calibración ───────────────────────────────────────────

    def _start_calib(self, mode: str):

        self.state.start_calib(mode)

        self.btn_linea.setChecked(mode == "linea")

        self.btn_circ.setChecked(mode == "circulo")

        self.lbl_calib_hint.setVisible(True)

        self.btn_cancel_calib.setVisible(True)

        if mode == "linea":

            self.lbl_calib_hint.setText(

                tr("Haz clic en 2 puntos sobre\nuna referencia conocida.")

            )

        else:

            self.lbl_calib_hint.setText(

                tr("Haz clic en 3 puntos del borde\nde un objeto circular conocido.")

            )

        self._set_status("● Modo calibración activo", T.WARN)



    def _on_calib_pts_changed(self, pts: list):

        if not pts:

            self.btn_linea.setChecked(False)

            self.btn_circ.setChecked(False)

            self.lbl_calib_hint.setVisible(False)

            self.btn_cancel_calib.setVisible(False)

        mode = self.state.calib_mode

        needed = 2 if mode == "linea" else 3

        self.lbl_calib_hint.setText(

            f"Haz clic en {needed - len(pts)} punto(s) más."

            if pts else self.lbl_calib_hint.text()

        )



    def _on_calib_ready(self):

        """Suficientes puntos recogidos → pedir tamaño real al usuario.



        Los argumentos posicionales son obligatorios: PySide6 no acepta ``min``

        ni ``max`` como palabras clave en ``getDouble`` y lanzaba AttributeError,

        lo que dejaba la calibración inutilizable.

        """

        mode = self.state.calib_mode

        if mode == "linea":

            val, ok = QInputDialog.getDouble(

                self, "Calibración — Línea",

                "Tamaño real de la línea marcada (μm):",

                1000.0, 0.001, 1_000_000.0, 3

            )

            if ok and val > 0:

                self.state.finish_calib_linea(val)

            else:

                self.state.cancel_calib()



        elif mode == "circulo":

            # Por defecto 100 mm: es el diámetro de las placas Petri del estudio

            val, ok = QInputDialog.getDouble(

                self, "Calibración — Círculo",

                "Diámetro real del objeto circular (μm):\n"

                "Placa Petri de 100 mm → 100000",

                100_000.0, 0.001, 10_000_000.0, 3

            )

            if ok and val > 0:

                self.state.finish_calib_circulo(val)

            else:

                self.state.cancel_calib()



    def _on_calib_changed(self, um_per_px: float, mode_name: str):

        self.lbl_calib_result.setText(f"{um_per_px:.4f} μm/px  ({mode_name})")

        self.lbl_calib_result.setStyleSheet(f"color: {T.OK_TX}; font-size: 9.5pt; font-weight: 600;")

        self._set_status(f"● Calibración aplicada: {um_per_px:.4f} μm/px", T.OK)

        self._update_status_bar()

        # Refrescar tabla con nuevas unidades μm

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



    # ── Guardar ───────────────────────────────────────────────

    def _save(self):

        img = self.state.current_image

        if img is None:

            QMessageBox.warning(self, tr("Sin imagen"), tr("No hay imagen abierta."))

            return



        # Elegir carpeta destino

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

            QMessageBox.information(self, tr("Guardado"),

                f"Archivos guardados en:\n{save_dir}")

        except Exception as e:

            QMessageBox.warning(self, tr("Error al guardar"), str(e))

            self._set_status("● Error al guardar", T.ERR)



    def _save_annotated_image(self, img: Path, save_dir: Path):

        import cv2

        import numpy as np

        # imdecode y no imread: en Windows imread falla con rutas que llevan

        # acentos o caracteres no ASCII, y devuelve None sin avisar.

        frame = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is None:

            return



        CLASS_COLORS_BGR = [

            (47,  52, 227),   # PET  → rojo

            (0,  140, 255),   # PP   → naranjo

            (0,  215, 255),   # LDPE → amarillo

            (216, 180,   0),

            (228,  202,  72),

            (182, 119,   0),

            (138,  62,   2),

            (139,  45, 118),

        ]

        for det in self.state.detections:

            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            color = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            w_px = det.x2 - det.x1

            h_px = det.y2 - det.y1

            diam_px = math.sqrt(w_px * h_px)

            um = self.state.px_to_um(diam_px)

            label = (f"{det.class_name} {um:.1f}um"

                     if um is not None else f"{det.class_name} {diam_px:.0f}px")

            cv2.putText(frame, label, (x1, max(y1 - 6, 14)),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)



        out_path = save_dir / f"{img.stem}_anotada.png"

        ok, buf = cv2.imencode(".png", frame)

        if ok:

            buf.tofile(str(out_path))



    def _save_csv(self, save_dir: Path):

        import csv

        out = save_dir / "detecciones.csv"

        with open(out, "w", newline="", encoding="utf-8") as f:

            w = csv.writer(f)

            w.writerow(["clase", "conf", "x1", "y1", "x2", "y2",

                         "diam_px", "diam_um", "area_um2"])

            for det in self.state.detections:

                wp = det.x2 - det.x1

                hp = det.y2 - det.y1

                diam_px = math.sqrt(wp * hp)

                um = self.state.px_to_um(diam_px)

                area_um2 = (self.state.um_per_px ** 2 * wp * hp

                            if self.state.um_per_px > 0 else "")

                w.writerow([

                    det.class_name, f"{det.conf:.4f}",

                    f"{det.x1:.1f}", f"{det.y1:.1f}",

                    f"{det.x2:.1f}", f"{det.y2:.1f}",

                    f"{diam_px:.2f}",

                    f"{um:.2f}" if um is not None else "",

                    f"{area_um2:.2f}" if area_um2 != "" else "",

                ])



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

                if p.endswith(".pt") or any(p.endswith(e) for e in

                        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):

                    ev.acceptProposedAction()

                    return

        ev.ignore()



    def dropEvent(self, ev):

        for url in ev.mimeData().urls():

            path = Path(url.toLocalFile())

            if path.suffix.lower() == ".pt":

                try:

                    from polyx.model.yolo_wrap import YoloModel

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

