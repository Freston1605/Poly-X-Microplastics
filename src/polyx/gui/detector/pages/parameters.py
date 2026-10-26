"""Page 4 — Inference parameters + optical calibration + filters."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
)

from polyx.core.i18n import tr
from polyx.core.platform import IS_MAC, architecture_mac, label_devices
from polyx.gui import iconos
from polyx.gui import theme as T

from ._base import DetectorPage


def _hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
    l.setWordWrap(True)
    return l


# Quality profiles: they hide YOLO jargon behind a simple choice.
# `Máxima detección` targets tiny particles in high-resolution photos
# (if the GPU cannot handle the imgsz, the auto-fallback lowers the size on its own).
PRESETS = {
    "Rápido": {"conf": 0.25, "imgsz": 1280},
    "Equilibrado": {"conf": 0.15, "imgsz": 2560},
    "Máxima detección": {"conf": 0.10, "imgsz": 4096},
}


class _ProbeThread(QThread):
    """Probes in the background the maximum imgsz the GPU can take for the loaded model."""

    progress = Signal(int)
    done = Signal(int, dict)
    failed = Signal(str)

    def __init__(self, model, image_path, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.image_path = image_path
        self.device = device

    def run(self):
        try:
            mx, detail = self.model.probe_max_imgsz(
                self.image_path,
                device=self.device,
                progress=lambda sz: self.progress.emit(sz),
            )
            self.done.emit(mx, detail)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class ParametersPage(DetectorPage):
    STEP_N = 4
    STEP_TITLE = tr("Parámetros")
    STEP_DESCRIPTION = tr(
        "Configura confianza, IoU, calibración óptica y filtros. La calibración μm/px es "
        "opcional pero recomendada para reportes científicos."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Inference ──
        c1, l1 = self.card(tr("Inferencia"), "ajustes")

        # Quality profile (preset): applies conf + imgsz at once
        self._applying_preset = False
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_row.addWidget(QLabel(tr("Perfil:")))
        self.combo_preset = QComboBox()
        self.combo_preset.addItem(tr("Personalizado"))
        self.combo_preset.addItems(list(PRESETS.keys()))
        self.combo_preset.currentTextChanged.connect(self._apply_preset)
        self.combo_preset.setMinimumWidth(190)
        preset_row.addWidget(self.combo_preset)
        preset_row.addWidget(
            _hint(
                tr(
                    "Elige un perfil y listo. «Máxima detección» es el recomendado para "
                    "microplásticos pequeños en fotos de microscopía de alta resolución."
                )
            ),
            1,
        )
        l1.addLayout(preset_row)

        g = QGridLayout()
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(10)

        # Confidence
        g.addWidget(QLabel(tr("Confianza mín. (conf):")), 0, 0)
        self.sb_conf = QDoubleSpinBox()
        self.sb_conf.setRange(0.01, 1.0)
        self.sb_conf.setSingleStep(0.05)
        self.sb_conf.setDecimals(2)
        self.sb_conf.setValue(state.params.conf)
        self.sb_conf.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_conf, 0, 1)
        g.addWidget(
            _hint(tr("0.05–0.95. Más alta = menos detecciones (con falsos positivos).")), 0, 2
        )

        # IoU NMS
        g.addWidget(QLabel(tr("IoU NMS:")), 1, 0)
        self.sb_iou_nms = QDoubleSpinBox()
        self.sb_iou_nms.setRange(0.1, 0.9)
        self.sb_iou_nms.setSingleStep(0.05)
        self.sb_iou_nms.setDecimals(2)
        self.sb_iou_nms.setValue(state.params.iou_nms)
        self.sb_iou_nms.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_iou_nms, 1, 1)
        g.addWidget(_hint(tr("0.30–0.70. Suprime cajas superpuestas.")), 1, 2)

        # imgsz + `detectar máximo` button
        g.addWidget(QLabel(tr("Tamaño imagen (imgsz):")), 2, 0)
        imgsz_row = QHBoxLayout()
        imgsz_row.setSpacing(8)
        self.sb_imgsz = QSpinBox()
        self.sb_imgsz.setRange(320, 8192)
        self.sb_imgsz.setSingleStep(64)
        self.sb_imgsz.setValue(state.params.imgsz)
        self.sb_imgsz.valueChanged.connect(self._on_change)
        imgsz_row.addWidget(self.sb_imgsz)
        self.btn_probe = QPushButton(tr("Detectar máximo (GPU)"))
        self.btn_probe.setIcon(iconos.icon("buscar", 15, T.INK2))
        self.btn_probe.setCursor(Qt.PointingHandCursor)
        self.btn_probe.clicked.connect(self._probe_max_imgsz)
        imgsz_row.addWidget(self.btn_probe)
        imgsz_row.addStretch(1)
        g.addLayout(imgsz_row, 2, 1)
        g.addWidget(
            _hint(
                tr(
                    "Más grande = detecta partículas más pequeñas (clave para microplásticos), "
                    "pero más lento y usa más memoria GPU. Para fotos de alta resolución sube "
                    "a 4096+. El botón prueba el máximo que aguanta tu GPU con el modelo cargado."
                )
            ),
            2,
            2,
        )

        # device
        # The label and hint depend on the system: on a Mac there is no
        # "0" GPU to choose, and offering it only leads to the analysis failing.
        g.addWidget(QLabel(tr(label_devices())), 3, 0)
        self.ed_device = QLineEdit(state.params.device)
        self.ed_device.editingFinished.connect(self._on_change)
        g.addWidget(self.ed_device, 3, 1)
        if IS_MAC:
            clue_2 = (
                tr("'mps' = GPU del Mac (Apple Silicon). 'cpu' = procesador.")
                if architecture_mac() == "Apple Silicon"
                else tr("Este Mac es Intel: solo 'cpu'. MPS necesita Apple Silicon.")
            )
        else:
            clue_2 = tr("'0' = primera GPU. 'cpu' = CPU. '0,1' = GPU 0 y 1.")
        g.addWidget(_hint(clue_2), 3, 2)

        l1.addLayout(g)
        self.body.addWidget(c1)

        # ── Automatic tiling ──
        c5, l5 = self.card(tr("Troceado automático (fotos grandes)"), "piezas")
        l5.addWidget(
            _hint(
                tr(
                    "Una foto de placa completa entra a la red reescalada a imgsz: a 4096 px "
                    "reducidos a 2080, cada partícula encoge a la mitad y desaparece bajo el "
                    "stride de la red. Troceando, cada tile entra a resolución nativa y las "
                    "cajas vuelven a coordenadas de la foto completa, fusionadas con NMS "
                    "global — no hay que cortar nada a mano ni sumar los conteos después."
                )
            )
        )
        g5 = QGridLayout()
        g5.setHorizontalSpacing(24)
        g5.setVerticalSpacing(10)

        g5.addWidget(QLabel(tr("Cuándo trocear:")), 0, 0)
        self.combo_tiling = QComboBox()
        # The value goes in userData, not in the displayed text: if the text were the
        # value, translating the interface would break the check against `"siempre"` and
        # `"nunca"` in predict_auto(). The text is translated, the data does not change.
        for _value, _label in (
            ("auto", tr("automático")),
            ("siempre", tr("siempre")),
            ("nunca", tr("nunca")),
        ):
            self.combo_tiling.addItem(_label, _value)
        self.combo_tiling.setCurrentIndex(max(0, self.combo_tiling.findData(state.params.tiling)))
        self.combo_tiling.currentTextChanged.connect(self._on_change)
        g5.addWidget(self.combo_tiling, 0, 1)
        g5.addWidget(_hint(tr("«auto» decide sola mirando el lado mayor de cada foto.")), 0, 2)

        g5.addWidget(QLabel(tr("Umbral (lado mayor, px):")), 1, 0)
        self.sb_threshold = QSpinBox()
        self.sb_threshold.setRange(512, 20000)
        self.sb_threshold.setSingleStep(128)
        self.sb_threshold.setValue(state.params.tiling_threshold_px)
        self.sb_threshold.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_threshold, 1, 1)
        g5.addWidget(
            _hint(
                tr(
                    "Por encima de esto se trocea. 2000 deja pasar enteros los recortes del "
                    "estudio (1630 px) y trocea las placas completas (~4096 px)."
                )
            ),
            1,
            2,
        )

        g5.addWidget(QLabel(tr("Lado del tile (px):")), 2, 0)
        self.sb_tile = QSpinBox()
        self.sb_tile.setRange(0, 8192)
        self.sb_tile.setSingleStep(64)
        self.sb_tile.setValue(state.params.tiling_tile)
        self.sb_tile.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_tile, 2, 1)
        g5.addWidget(
            _hint(
                tr(
                    "0 = automático, min(umbral, imgsz). Conviene que coincida con el tamaño "
                    "de los recortes con que se entrenó el modelo: si el modelo vio "
                    "partículas a una escala y el tile las presenta a otra, el recall cae."
                )
            ),
            2,
            2,
        )

        g5.addWidget(QLabel(tr("Solape entre tiles:")), 3, 0)
        self.sb_overlap = QDoubleSpinBox()
        self.sb_overlap.setRange(0.0, 0.9)
        self.sb_overlap.setSingleStep(0.05)
        self.sb_overlap.setDecimals(2)
        self.sb_overlap.setValue(state.params.tiling_overlap)
        self.sb_overlap.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_overlap, 3, 1)
        g5.addWidget(
            _hint(
                tr(
                    "0.25 = cada tile pisa un cuarto del vecino. El solape es lo que evita "
                    "perder o duplicar la partícula que cae justo en la costura."
                )
            ),
            3,
            2,
        )

        l5.addLayout(g5)
        self.lbl_plan = QLabel("")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; border: none;")
        l5.addWidget(self.lbl_plan)
        self.body.addWidget(c5)
        self._refresh_plan()

        # ── Error analysis ──
        c2, l2 = self.card(tr("Análisis de errores (si hay GT)"), "diana")
        grid2 = QGridLayout()
        grid2.setHorizontalSpacing(24)
        grid2.setVerticalSpacing(10)
        grid2.addWidget(QLabel(tr("IoU para emparejar Verdaderos Positivos:")), 0, 0)
        self.sb_iou_tp = QDoubleSpinBox()
        self.sb_iou_tp.setRange(0.1, 0.95)
        self.sb_iou_tp.setSingleStep(0.05)
        self.sb_iou_tp.setDecimals(2)
        self.sb_iou_tp.setValue(state.params.iou_tp)
        self.sb_iou_tp.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_iou_tp, 0, 1)
        grid2.addWidget(_hint(tr("Estándar COCO: 0.50. Más estricto: 0.75.")), 0, 2)
        l2.addLayout(grid2)
        l2.addWidget(
            _hint(
                tr(
                    "Verdaderos Positivos: clase correcta + IoU ≥ umbral.  "
                    "Falsos Positivos: predicción sin GT cercano.  "
                    "Falsos Negativos: GT no detectado.  "
                    "Mal Clasificados: IoU alto pero clase distinta."
                )
            )
        )
        self.body.addWidget(c2)

        # ── Optical calibration ──
        c3, l3 = self.card(tr("Calibración óptica (tamaño de partícula)"), "visor")
        g3 = QGridLayout()
        g3.setHorizontalSpacing(24)
        g3.setVerticalSpacing(10)
        g3.addWidget(QLabel(tr("μm por píxel:")), 0, 0)
        self.sb_umpx = QDoubleSpinBox()
        # Up to 1000 and not 10: the plates in this study range from 31 to 49 µm/px, so
        # the previous cap silently clipped the real calibration and
        # left the value at 10 without warning about anything.
        self.sb_umpx.setRange(0.0, 1000.0)
        self.sb_umpx.setSingleStep(0.01)
        self.sb_umpx.setDecimals(4)
        self.sb_umpx.setValue(state.params.um_per_px)
        self.sb_umpx.valueChanged.connect(self._on_change)
        g3.addWidget(self.sb_umpx, 0, 1)
        g3.addWidget(
            _hint(tr("Respaldo: se usa solo si no se puede medir la placa. 0 = sin medición.")),
            0,
            2,
        )
        l3.addLayout(g3)

        # ── Automatic calibration against the dish ──
        self.chk_plate = QCheckBox(tr("Medir la placa en cada foto y usar su escala (recomendado)"))
        self.chk_plate.setChecked(state.params.measure_plate)
        self.chk_plate.setStyleSheet(f"color: {T.INK2}; border: none;")
        self.chk_plate.toggled.connect(self._on_change)
        l3.addWidget(self.chk_plate)

        g4 = QGridLayout()
        g4.setHorizontalSpacing(24)
        g4.setVerticalSpacing(10)
        g4.addWidget(QLabel(tr("Diámetro real de la placa (mm):")), 0, 0)
        self.sb_plate_mm = QDoubleSpinBox()
        self.sb_plate_mm.setRange(1.0, 1000.0)
        self.sb_plate_mm.setDecimals(2)
        self.sb_plate_mm.setValue(state.params.diameter_plate_mm)
        self.sb_plate_mm.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_plate_mm, 0, 1)
        g4.addWidget(
            _hint(tr("Diámetro externo nominal. La placa Petri del estudio mide 100 mm.")), 0, 2
        )

        g4.addWidget(QLabel(tr("Altura de la placa (mm):")), 1, 0)
        self.sb_height = QDoubleSpinBox()
        self.sb_height.setRange(0.0, 100.0)
        self.sb_height.setDecimals(1)
        self.sb_height.setValue(state.params.height_plate_mm)
        self.sb_height.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_height, 1, 1)
        g4.addWidget(
            _hint(tr("0 = no corregir. Una Petri de 100 mm suele medir 15 mm de alto.")), 1, 2
        )

        g4.addWidget(QLabel(tr("Distancia cámara → base (mm):")), 2, 0)
        self.sb_distance = QDoubleSpinBox()
        self.sb_distance.setRange(0.0, 5000.0)
        self.sb_distance.setDecimals(0)
        self.sb_distance.setValue(state.params.distance_camera_mm)
        self.sb_distance.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_distance, 2, 1)
        g4.addWidget(
            _hint(tr("0 = no corregir. Medir hasta el FONDO de la placa, no hasta el borde.")), 2, 2
        )
        l3.addLayout(g4)

        l3.addWidget(
            _hint(
                tr(
                    "No hay que marcar nada: el borde de la placa se encuentra solo. El radio se "
                    "obtiene ajustando un círculo al anillo muestreado en 720 direcciones, con "
                    "rechazo de atípicos; Hough solo aporta el centro aproximado porque su radio "
                    "llega a errar un 12 %. En los recortes la placa no aparece, así que ahí la "
                    "escala se hereda del índice de calibración que dejó el recorte."
                )
            )
        )
        l3.addWidget(
            _hint(
                tr(
                    "El borde de la placa está más cerca de la cámara que el fondo donde reposan "
                    "las partículas, de modo que se proyecta más grande y la escala sale pequeña: "
                    "sin corregir, TODAS las tallas quedan subestimadas. Con la placa a 15 mm y la "
                    "cámara a 100 mm son casi 16 puntos porcentuales; a 500 mm, un 3 %. Rellena los "
                    "dos campos de arriba para corregirlo."
                )
            )
        )
        l3.addWidget(
            _hint(
                tr(
                    "El detector calcula área y diámetro equivalente (de círculo con misma área) "
                    "para cada partícula detectada. Si hay calibración, también en μm y μm²."
                )
            )
        )
        self.body.addWidget(c3)

        # ── Size filter ──
        c4, l4 = self.card(tr("Filtro por tamaño (opcional)"), "visor")
        g4 = QGridLayout()
        g4.setHorizontalSpacing(24)
        g4.setVerticalSpacing(10)
        g4.addWidget(QLabel(tr("Tamaño mín (μm):")), 0, 0)
        self.sb_min = QDoubleSpinBox()
        self.sb_min.setRange(0.0, 10000.0)
        self.sb_min.setDecimals(2)
        self.sb_min.setValue(state.params.size_min_um)
        self.sb_min.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_min, 0, 1)

        g4.addWidget(QLabel(tr("Tamaño máx (μm):")), 0, 2)
        self.sb_max = QDoubleSpinBox()
        self.sb_max.setRange(0.0, 10000.0)
        self.sb_max.setDecimals(2)
        self.sb_max.setValue(state.params.size_max_um)
        self.sb_max.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_max, 0, 3)
        g4.addWidget(_hint(tr("0 = sin filtro. Aplica sobre el diámetro equivalente.")), 1, 0, 1, 4)
        l4.addLayout(g4)
        self.body.addWidget(c4)

        # ── How far the polymer holds ──
        c5, l5 = self.card(tr("Asignación de polímero (opcional)"), "ensayo")
        g5 = QGridLayout()
        g5.setHorizontalSpacing(24)
        g5.setVerticalSpacing(10)
        g5.addWidget(QLabel(tr("Confianza mínima para asignar:")), 0, 0)
        self.sb_assignment = QDoubleSpinBox()
        self.sb_assignment.setRange(0.0, 1.0)
        self.sb_assignment.setDecimals(2)
        self.sb_assignment.setSingleStep(0.05)
        self.sb_assignment.setValue(state.params.conf_assignment)
        self.sb_assignment.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_assignment, 0, 1)
        g5.addWidget(
            _hint(
                tr(
                    "0 = desactivado. Por debajo de este valor la partícula se cuenta igual, "
                    "pero se reporta como «no asignable» en vez de atribuirle un polímero. "
                    "El Nile Red responde a la polaridad del entorno, no a la identidad "
                    "química: separa bien el PET —poliéster— del resto, mientras que PP y "
                    "LDPE son las dos poliolefinas y se distinguen solo por brillo, que "
                    "depende de la exposición y del foco. Abstenerse es más honesto que "
                    "inventar la clase, y el informe declara qué porcentaje quedó sin asignar."
                )
            ),
            1,
            0,
            1,
            4,
        )
        l5.addLayout(g5)
        self.body.addWidget(c5)

    def _apply_preset(self, name: str):
        """Applies the chosen profile (conf + imgsz) to the controls."""
        cfg = PRESETS.get(name)
        if not cfg:
            return
        self._applying_preset = True
        try:
            self.sb_conf.setValue(cfg["conf"])
            self.sb_imgsz.setValue(cfg["imgsz"])
        finally:
            self._applying_preset = False

    def _probe_max_imgsz(self):
        """Launches the maximum-imgsz probe in the background over the active model/image."""
        models = self.state.active_models()
        if not models:
            QMessageBox.warning(
                self, tr("Sin modelo"), tr("Carga al menos un modelo en la pestaña Modelos.")
            )
            return
        if not self.state.images:
            QMessageBox.warning(
                self,
                tr("Sin imágenes"),
                tr("Selecciona imágenes en la pestaña Imágenes (se usa una para medir)."),
            )
            return
        slot = models[0]
        if slot.loaded is None:
            from polyx.core.detect.yolo import YoloModel

            slot.loaded = YoloModel(str(slot.path), alias=slot.alias)
        device = self.state.params.device
        self.btn_probe.setEnabled(False)
        self.btn_probe.setText(tr("Probando…"))
        self._probe = _ProbeThread(slot.loaded, str(self.state.images[0]), device, self)
        self._probe.progress.connect(lambda sz: self.btn_probe.setText(f"⏳ Probando {sz}px…"))
        self._probe.done.connect(self._on_probe_done)
        self._probe.failed.connect(self._on_probe_failed)
        self._probe.start()

    def _on_probe_done(self, max_ok: int, detail: dict):
        self.btn_probe.setEnabled(True)
        self.btn_probe.setText(tr("Detectar máximo (GPU)"))
        self.sb_imgsz.setValue(max_ok)
        oom = [str(k) for k, v in detail.items() if v == "oom"]
        msg = (
            f"Máximo seguro para «{self.state.active_models()[0].alias}»: "
            f"{max_ok} px.\n\nSe fijó imgsz = {max_ok}."
        )
        if oom:
            msg += f"\n(A {oom[0]} px se quedó sin memoria.)"
        if any(v == "cpu" for v in detail.values()):
            msg = (
                f"En CPU no hay tope de memoria GPU. Se fijó imgsz = {max_ok} "
                "pero será lento. Con GPU el detector es mucho más rápido."
            )
        QMessageBox.information(self, tr("imgsz máximo"), msg)

    def _on_probe_failed(self, err: str):
        self.btn_probe.setEnabled(True)
        self.btn_probe.setText(tr("Detectar máximo (GPU)"))
        QMessageBox.warning(self, tr("No se pudo medir"), err)

    def _on_change(self, *_):
        # Manual edit → the profile stops applying (returns to `"Personalizado"`)
        if not self._applying_preset and hasattr(self, "combo_preset"):
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentIndex(0)
            self.combo_preset.blockSignals(False)
        p = self.state.params
        p.conf = float(self.sb_conf.value())
        p.iou_nms = float(self.sb_iou_nms.value())
        p.iou_tp = float(self.sb_iou_tp.value())
        p.imgsz = int(self.sb_imgsz.value())
        p.device = self.ed_device.text().strip() or "0"
        p.um_per_px = float(self.sb_umpx.value())
        p.measure_plate = self.chk_plate.isChecked()
        p.diameter_plate_mm = float(self.sb_plate_mm.value())
        p.height_plate_mm = float(self.sb_height.value())
        p.distance_camera_mm = float(self.sb_distance.value())
        p.size_min_um = float(self.sb_min.value())
        p.size_max_um = float(self.sb_max.value())
        p.conf_assignment = float(self.sb_assignment.value())
        p.tiling = self.combo_tiling.currentData() or "auto"
        p.tiling_threshold_px = int(self.sb_threshold.value())
        p.tiling_tile = int(self.sb_tile.value())
        p.tiling_overlap = float(self.sb_overlap.value())
        self._refresh_plan()
        self.state.params_changed.emit()

    def _refresh_plan(self):
        """Shows what the tiling will do with the already selected images.

        Seeing the decision before launching the batch avoids the surprise of
        discovering two hours later that everything was inferred in one piece.
        """
        from polyx.core.detect.yolo import policy_tiling, size_image

        p = self.state.params
        if not self.state.images:
            self.lbl_plan.setText(tr("Selecciona imágenes para ver qué se trocearía."))
            return
        wh = size_image(self.state.images[0])
        if wh is None:
            self.lbl_plan.setText(tr("No se pudo leer el tamaño de la primera imagen."))
            return
        if p.tiling == "nunca":
            self.lbl_plan.setText(
                f"Primera imagen {wh[0]}×{wh[1]} px → un solo pase a imgsz {p.imgsz} "
                f"(troceado desactivado)."
            )
            return
        plan = policy_tiling(
            wh[0],
            wh[1],
            p.imgsz,
            threshold_px=0 if p.tiling == "siempre" else p.tiling_threshold_px,
            tile=p.tiling_tile,
            overlap=p.tiling_overlap,
        )
        if plan is None:
            self.lbl_plan.setText(
                f"Primera imagen {wh[0]}×{wh[1]} px → un solo pase: no llega al "
                f"umbral de {p.tiling_threshold_px} px."
            )
        else:
            scale = p.imgsz / plan["tile"]
            self.lbl_plan.setText(
                f"Primera imagen {wh[0]}×{wh[1]} px → {plan['n_tiles']} tiles de "
                f"{plan['tile']} px con {int(plan['overlap'] * 100)}% de solape, cada "
                f"uno a imgsz {p.imgsz} ({scale:.2f}× la resolución nativa del tile). "
                f"Tarda ~{plan['n_tiles']}× más que un pase único."
            )
