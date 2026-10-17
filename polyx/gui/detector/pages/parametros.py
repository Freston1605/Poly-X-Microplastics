"""Página 4 — Parámetros de inferencia + calibración óptica + filtros."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QLineEdit, QFrame, QPushButton, QMessageBox, QComboBox, QCheckBox,
)

from ._base import DetectorPage
from polyx.model.i18n import tr
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.model.plataforma import ES_MAC, arquitectura_mac, etiqueta_dispositivos


def _hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
    l.setWordWrap(True)
    return l


# Perfiles de calidad: ocultan la jerga YOLO tras una elección simple.
# "Máxima detección" apunta a partículas diminutas en fotos de alta resolución
# (si la GPU no aguanta el imgsz, el auto-fallback baja de tamaño solo).
PRESETS = {
    "Rápido": {"conf": 0.25, "imgsz": 1280},
    "Equilibrado": {"conf": 0.15, "imgsz": 2560},
    "Máxima detección": {"conf": 0.10, "imgsz": 4096},
}


class _ProbeThread(QThread):
    """Prueba en background el imgsz máximo que aguanta la GPU para el modelo cargado."""
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
                self.image_path, device=self.device,
                progress=lambda sz: self.progress.emit(sz),
            )
            self.done.emit(mx, detail)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class ParametrosPage(DetectorPage):
    STEP_N = 4
    STEP_TITLE = tr("Parámetros")
    STEP_DESCRIPTION = (
        tr("Configura confianza, IoU, calibración óptica y filtros. La calibración μm/px es "
        "opcional pero recomendada para reportes científicos.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Inferencia ──
        c1, l1 = self.card(tr("Inferencia"), "ajustes")

        # Perfil de calidad (preset): aplica conf + imgsz de una vez
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
        preset_row.addWidget(_hint(
            tr("Elige un perfil y listo. «Máxima detección» es el recomendado para "
            "microplásticos pequeños en fotos de microscopía de alta resolución.")
        ), 1)
        l1.addLayout(preset_row)

        g = QGridLayout()
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(10)

        # Confianza
        g.addWidget(QLabel(tr("Confianza mín. (conf):")), 0, 0)
        self.sb_conf = QDoubleSpinBox()
        self.sb_conf.setRange(0.01, 1.0); self.sb_conf.setSingleStep(0.05)
        self.sb_conf.setDecimals(2); self.sb_conf.setValue(state.params.conf)
        self.sb_conf.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_conf, 0, 1)
        g.addWidget(_hint(tr("0.05–0.95. Más alta = menos detecciones (con falsos positivos).")), 0, 2)

        # IoU NMS
        g.addWidget(QLabel(tr("IoU NMS:")), 1, 0)
        self.sb_iou_nms = QDoubleSpinBox()
        self.sb_iou_nms.setRange(0.1, 0.9); self.sb_iou_nms.setSingleStep(0.05)
        self.sb_iou_nms.setDecimals(2); self.sb_iou_nms.setValue(state.params.iou_nms)
        self.sb_iou_nms.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_iou_nms, 1, 1)
        g.addWidget(_hint(tr("0.30–0.70. Suprime cajas superpuestas.")), 1, 2)

        # imgsz + botón "detectar máximo"
        g.addWidget(QLabel(tr("Tamaño imagen (imgsz):")), 2, 0)
        imgsz_row = QHBoxLayout()
        imgsz_row.setSpacing(8)
        self.sb_imgsz = QSpinBox()
        self.sb_imgsz.setRange(320, 8192); self.sb_imgsz.setSingleStep(64)
        self.sb_imgsz.setValue(state.params.imgsz)
        self.sb_imgsz.valueChanged.connect(self._on_change)
        imgsz_row.addWidget(self.sb_imgsz)
        self.btn_probe = QPushButton(tr("Detectar máximo (GPU)"))
        self.btn_probe.setIcon(iconos.icono("buscar", 15, T.INK2))
        self.btn_probe.setCursor(Qt.PointingHandCursor)
        self.btn_probe.clicked.connect(self._probe_max_imgsz)
        imgsz_row.addWidget(self.btn_probe)
        imgsz_row.addStretch(1)
        g.addLayout(imgsz_row, 2, 1)
        g.addWidget(_hint(
            tr("Más grande = detecta partículas más pequeñas (clave para microplásticos), "
            "pero más lento y usa más memoria GPU. Para fotos de alta resolución sube "
            "a 4096+. El botón prueba el máximo que aguanta tu GPU con el modelo cargado.")
        ), 2, 2)

        # device
        # El rotulo y la pista dependen del sistema: en un Mac no hay ninguna
        # GPU "0" que elegir, y ofrecerla solo lleva a que el analisis falle.
        g.addWidget(QLabel(tr(etiqueta_dispositivos())), 3, 0)
        self.ed_device = QLineEdit(state.params.device)
        self.ed_device.editingFinished.connect(self._on_change)
        g.addWidget(self.ed_device, 3, 1)
        if ES_MAC:
            pista = (tr("'mps' = GPU del Mac (Apple Silicon). 'cpu' = procesador.")
                     if arquitectura_mac() == "Apple Silicon"
                     else tr("Este Mac es Intel: solo 'cpu'. MPS necesita Apple Silicon."))
        else:
            pista = tr("'0' = primera GPU. 'cpu' = CPU. '0,1' = GPU 0 y 1.")
        g.addWidget(_hint(pista), 3, 2)

        l1.addLayout(g)
        self.body.addWidget(c1)

        # ── Troceado automático ──
        c5, l5 = self.card(tr("Troceado automático (fotos grandes)"), "piezas")
        l5.addWidget(_hint(
            tr("Una foto de placa completa entra a la red reescalada a imgsz: a 4096 px "
            "reducidos a 2080, cada partícula encoge a la mitad y desaparece bajo el "
            "stride de la red. Troceando, cada tile entra a resolución nativa y las "
            "cajas vuelven a coordenadas de la foto completa, fusionadas con NMS "
            "global — no hay que cortar nada a mano ni sumar los conteos después.")
        ))
        g5 = QGridLayout()
        g5.setHorizontalSpacing(24); g5.setVerticalSpacing(10)

        g5.addWidget(QLabel(tr("Cuándo trocear:")), 0, 0)
        self.combo_troceo = QComboBox()
        # El valor va en userData y no en el texto visible: si el texto fuera el
        # valor, traducir la interfaz romperia la comparacion con "siempre" y
        # "nunca" en predict_auto(). El texto se traduce, el dato no cambia.
        for _valor, _rotulo in (("auto", tr("automático")),
                                ("siempre", tr("siempre")),
                                ("nunca", tr("nunca"))):
            self.combo_troceo.addItem(_rotulo, _valor)
        self.combo_troceo.setCurrentIndex(
            max(0, self.combo_troceo.findData(state.params.troceo)))
        self.combo_troceo.currentTextChanged.connect(self._on_change)
        g5.addWidget(self.combo_troceo, 0, 1)
        g5.addWidget(_hint(tr("«auto» decide sola mirando el lado mayor de cada foto.")), 0, 2)

        g5.addWidget(QLabel(tr("Umbral (lado mayor, px):")), 1, 0)
        self.sb_umbral = QSpinBox()
        self.sb_umbral.setRange(512, 20000); self.sb_umbral.setSingleStep(128)
        self.sb_umbral.setValue(state.params.troceo_umbral_px)
        self.sb_umbral.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_umbral, 1, 1)
        g5.addWidget(_hint(
            tr("Por encima de esto se trocea. 2000 deja pasar enteros los recortes del "
            "estudio (1630 px) y trocea las placas completas (~4096 px).")
        ), 1, 2)

        g5.addWidget(QLabel(tr("Lado del tile (px):")), 2, 0)
        self.sb_tile = QSpinBox()
        self.sb_tile.setRange(0, 8192); self.sb_tile.setSingleStep(64)
        self.sb_tile.setValue(state.params.troceo_tile)
        self.sb_tile.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_tile, 2, 1)
        g5.addWidget(_hint(
            tr("0 = automático, min(umbral, imgsz). Conviene que coincida con el tamaño "
            "de los recortes con que se entrenó el modelo: si el modelo vio "
            "partículas a una escala y el tile las presenta a otra, el recall cae.")
        ), 2, 2)

        g5.addWidget(QLabel(tr("Solape entre tiles:")), 3, 0)
        self.sb_overlap = QDoubleSpinBox()
        self.sb_overlap.setRange(0.0, 0.9); self.sb_overlap.setSingleStep(0.05)
        self.sb_overlap.setDecimals(2)
        self.sb_overlap.setValue(state.params.troceo_overlap)
        self.sb_overlap.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_overlap, 3, 1)
        g5.addWidget(_hint(
            tr("0.25 = cada tile pisa un cuarto del vecino. El solape es lo que evita "
            "perder o duplicar la partícula que cae justo en la costura.")
        ), 3, 2)

        l5.addLayout(g5)
        self.lbl_plan = QLabel("")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; border: none;")
        l5.addWidget(self.lbl_plan)
        self.body.addWidget(c5)
        self._refrescar_plan()

        # ── Análisis de errores ──
        c2, l2 = self.card(tr("Análisis de errores (si hay GT)"), "diana")
        g2 = QGridLayout()
        g2.setHorizontalSpacing(24); g2.setVerticalSpacing(10)
        g2.addWidget(QLabel(tr("IoU para emparejar Verdaderos Positivos:")), 0, 0)
        self.sb_iou_tp = QDoubleSpinBox()
        self.sb_iou_tp.setRange(0.1, 0.95); self.sb_iou_tp.setSingleStep(0.05)
        self.sb_iou_tp.setDecimals(2); self.sb_iou_tp.setValue(state.params.iou_tp)
        self.sb_iou_tp.valueChanged.connect(self._on_change)
        g2.addWidget(self.sb_iou_tp, 0, 1)
        g2.addWidget(_hint(tr("Estándar COCO: 0.50. Más estricto: 0.75.")), 0, 2)
        l2.addLayout(g2)
        l2.addWidget(_hint(
            tr("Verdaderos Positivos: clase correcta + IoU ≥ umbral.  "
            "Falsos Positivos: predicción sin GT cercano.  "
            "Falsos Negativos: GT no detectado.  "
            "Mal Clasificados: IoU alto pero clase distinta.")
        ))
        self.body.addWidget(c2)

        # ── Calibración óptica ──
        c3, l3 = self.card(tr("Calibración óptica (tamaño de partícula)"), "visor")
        g3 = QGridLayout()
        g3.setHorizontalSpacing(24); g3.setVerticalSpacing(10)
        g3.addWidget(QLabel(tr("μm por píxel:")), 0, 0)
        self.sb_umpx = QDoubleSpinBox()
        # Hasta 1000 y no 10: las placas de este estudio van de 31 a 49 µm/px, de
        # modo que el tope anterior recortaba en silencio la calibracion real y
        # dejaba el valor en 10 sin avisar de nada.
        self.sb_umpx.setRange(0.0, 1000.0); self.sb_umpx.setSingleStep(0.01)
        self.sb_umpx.setDecimals(4); self.sb_umpx.setValue(state.params.um_per_px)
        self.sb_umpx.valueChanged.connect(self._on_change)
        g3.addWidget(self.sb_umpx, 0, 1)
        g3.addWidget(_hint(
            tr("Respaldo: se usa solo si no se puede medir la placa. 0 = sin medición.")
        ), 0, 2)
        l3.addLayout(g3)

        # ── Calibración automática contra la placa ──
        self.chk_placa = QCheckBox(
            tr("Medir la placa en cada foto y usar su escala (recomendado)"))
        self.chk_placa.setChecked(state.params.medir_placa)
        self.chk_placa.setStyleSheet(f"color: {T.INK2}; border: none;")
        self.chk_placa.toggled.connect(self._on_change)
        l3.addWidget(self.chk_placa)

        g4 = QGridLayout()
        g4.setHorizontalSpacing(24); g4.setVerticalSpacing(10)
        g4.addWidget(QLabel(tr("Diámetro real de la placa (mm):")), 0, 0)
        self.sb_placa_mm = QDoubleSpinBox()
        self.sb_placa_mm.setRange(1.0, 1000.0); self.sb_placa_mm.setDecimals(2)
        self.sb_placa_mm.setValue(state.params.diametro_placa_mm)
        self.sb_placa_mm.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_placa_mm, 0, 1)
        g4.addWidget(_hint(tr("Diámetro externo nominal. La placa Petri del estudio mide 100 mm.")),
                     0, 2)

        g4.addWidget(QLabel(tr("Altura de la placa (mm):")), 1, 0)
        self.sb_altura = QDoubleSpinBox()
        self.sb_altura.setRange(0.0, 100.0); self.sb_altura.setDecimals(1)
        self.sb_altura.setValue(state.params.altura_placa_mm)
        self.sb_altura.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_altura, 1, 1)
        g4.addWidget(_hint(tr("0 = no corregir. Una Petri de 100 mm suele medir 15 mm de alto.")), 1, 2)

        g4.addWidget(QLabel(tr("Distancia cámara → base (mm):")), 2, 0)
        self.sb_distancia = QDoubleSpinBox()
        self.sb_distancia.setRange(0.0, 5000.0); self.sb_distancia.setDecimals(0)
        self.sb_distancia.setValue(state.params.distancia_camara_mm)
        self.sb_distancia.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_distancia, 2, 1)
        g4.addWidget(_hint(tr("0 = no corregir. Medir hasta el FONDO de la placa, no hasta el borde.")), 2, 2)
        l3.addLayout(g4)

        l3.addWidget(_hint(tr(
            "No hay que marcar nada: el borde de la placa se encuentra solo. El radio se "
            "obtiene ajustando un círculo al anillo muestreado en 720 direcciones, con "
            "rechazo de atípicos; Hough solo aporta el centro aproximado porque su radio "
            "llega a errar un 12 %. En los recortes la placa no aparece, así que ahí la "
            "escala se hereda del índice de calibración que dejó el recorte.")))
        l3.addWidget(_hint(tr(
            "El borde de la placa está más cerca de la cámara que el fondo donde reposan "
            "las partículas, de modo que se proyecta más grande y la escala sale pequeña: "
            "sin corregir, TODAS las tallas quedan subestimadas. Con la placa a 15 mm y la "
            "cámara a 100 mm son casi 16 puntos porcentuales; a 500 mm, un 3 %. Rellena los "
            "dos campos de arriba para corregirlo.")))
        l3.addWidget(_hint(
            tr("El detector calcula área y diámetro equivalente (de círculo con misma área) "
            "para cada partícula detectada. Si hay calibración, también en μm y μm².")
        ))
        self.body.addWidget(c3)

        # ── Filtro por tamaño ──
        c4, l4 = self.card(tr("Filtro por tamaño (opcional)"), "visor")
        g4 = QGridLayout()
        g4.setHorizontalSpacing(24); g4.setVerticalSpacing(10)
        g4.addWidget(QLabel(tr("Tamaño mín (μm):")), 0, 0)
        self.sb_min = QDoubleSpinBox(); self.sb_min.setRange(0.0, 10000.0)
        self.sb_min.setDecimals(2); self.sb_min.setValue(state.params.size_min_um)
        self.sb_min.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_min, 0, 1)

        g4.addWidget(QLabel(tr("Tamaño máx (μm):")), 0, 2)
        self.sb_max = QDoubleSpinBox(); self.sb_max.setRange(0.0, 10000.0)
        self.sb_max.setDecimals(2); self.sb_max.setValue(state.params.size_max_um)
        self.sb_max.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_max, 0, 3)
        g4.addWidget(_hint(tr("0 = sin filtro. Aplica sobre el diámetro equivalente.")), 1, 0, 1, 4)
        l4.addLayout(g4)
        self.body.addWidget(c4)

        # ── Hasta dónde se sostiene el polímero ──
        c5, l5 = self.card(tr("Asignación de polímero (opcional)"), "ensayo")
        g5 = QGridLayout()
        g5.setHorizontalSpacing(24); g5.setVerticalSpacing(10)
        g5.addWidget(QLabel(tr("Confianza mínima para asignar:")), 0, 0)
        self.sb_asignacion = QDoubleSpinBox(); self.sb_asignacion.setRange(0.0, 1.0)
        self.sb_asignacion.setDecimals(2); self.sb_asignacion.setSingleStep(0.05)
        self.sb_asignacion.setValue(state.params.conf_asignacion)
        self.sb_asignacion.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_asignacion, 0, 1)
        g5.addWidget(_hint(tr(
            "0 = desactivado. Por debajo de este valor la partícula se cuenta igual, "
            "pero se reporta como «no asignable» en vez de atribuirle un polímero. "
            "El Nile Red responde a la polaridad del entorno, no a la identidad "
            "química: separa bien el PET —poliéster— del resto, mientras que PP y "
            "LDPE son las dos poliolefinas y se distinguen solo por brillo, que "
            "depende de la exposición y del foco. Abstenerse es más honesto que "
            "inventar la clase, y el informe declara qué porcentaje quedó sin asignar.")),
            1, 0, 1, 4)
        l5.addLayout(g5)
        self.body.addWidget(c5)

    def _apply_preset(self, name: str):
        """Aplica el perfil elegido (conf + imgsz) a los controles."""
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
        """Lanza el probe del imgsz máximo en background sobre el modelo/imagen activos."""
        models = self.state.active_models()
        if not models:
            QMessageBox.warning(self, tr("Sin modelo"),
                                tr("Carga al menos un modelo en la pestaña Modelos."))
            return
        if not self.state.images:
            QMessageBox.warning(self, tr("Sin imágenes"),
                                tr("Selecciona imágenes en la pestaña Imágenes "
                                "(se usa una para medir)."))
            return
        slot = models[0]
        if slot.loaded is None:
            from polyx.model.yolo_wrap import YoloModel
            slot.loaded = YoloModel(str(slot.path), alias=slot.alias)
        device = self.state.params.device
        self.btn_probe.setEnabled(False)
        self.btn_probe.setText(tr("Probando…"))
        self._probe = _ProbeThread(slot.loaded, str(self.state.images[0]), device, self)
        self._probe.progress.connect(
            lambda sz: self.btn_probe.setText(f"⏳ Probando {sz}px…"))
        self._probe.done.connect(self._on_probe_done)
        self._probe.failed.connect(self._on_probe_failed)
        self._probe.start()

    def _on_probe_done(self, max_ok: int, detail: dict):
        self.btn_probe.setEnabled(True)
        self.btn_probe.setText(tr("Detectar máximo (GPU)"))
        self.sb_imgsz.setValue(max_ok)
        oom = [str(k) for k, v in detail.items() if v == "oom"]
        msg = (f"Máximo seguro para «{self.state.active_models()[0].alias}»: "
               f"{max_ok} px.\n\nSe fijó imgsz = {max_ok}.")
        if oom:
            msg += f"\n(A {oom[0]} px se quedó sin memoria.)"
        if any(v == "cpu" for v in detail.values()):
            msg = (f"En CPU no hay tope de memoria GPU. Se fijó imgsz = {max_ok} "
                   "pero será lento. Con GPU el detector es mucho más rápido.")
        QMessageBox.information(self, tr("imgsz máximo"), msg)

    def _on_probe_failed(self, err: str):
        self.btn_probe.setEnabled(True)
        self.btn_probe.setText(tr("Detectar máximo (GPU)"))
        QMessageBox.warning(self, tr("No se pudo medir"), err)

    def _on_change(self, *_):
        # Edición manual → el perfil deja de aplicar (vuelve a "Personalizado")
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
        p.medir_placa = self.chk_placa.isChecked()
        p.diametro_placa_mm = float(self.sb_placa_mm.value())
        p.altura_placa_mm = float(self.sb_altura.value())
        p.distancia_camara_mm = float(self.sb_distancia.value())
        p.size_min_um = float(self.sb_min.value())
        p.size_max_um = float(self.sb_max.value())
        p.conf_asignacion = float(self.sb_asignacion.value())
        p.troceo = self.combo_troceo.currentData() or "auto"
        p.troceo_umbral_px = int(self.sb_umbral.value())
        p.troceo_tile = int(self.sb_tile.value())
        p.troceo_overlap = float(self.sb_overlap.value())
        self._refrescar_plan()
        self.state.params_changed.emit()

    def _refrescar_plan(self):
        """Muestra qué hará el troceado con las imágenes ya seleccionadas.

        Ver la decisión antes de lanzar el lote evita la sorpresa de descubrir a
        las dos horas que se infirió todo de una pieza.
        """
        from polyx.model.yolo_wrap import tamano_imagen, politica_troceado
        p = self.state.params
        if not self.state.images:
            self.lbl_plan.setText(tr("Selecciona imágenes para ver qué se trocearía."))
            return
        wh = tamano_imagen(self.state.images[0])
        if wh is None:
            self.lbl_plan.setText(tr("No se pudo leer el tamaño de la primera imagen."))
            return
        if p.troceo == "nunca":
            self.lbl_plan.setText(
                f"Primera imagen {wh[0]}×{wh[1]} px → un solo pase a imgsz {p.imgsz} "
                f"(troceado desactivado)."
            )
            return
        plan = politica_troceado(
            wh[0], wh[1], p.imgsz,
            umbral_px=0 if p.troceo == "siempre" else p.troceo_umbral_px,
            tile=p.troceo_tile, overlap=p.troceo_overlap)
        if plan is None:
            self.lbl_plan.setText(
                f"Primera imagen {wh[0]}×{wh[1]} px → un solo pase: no llega al "
                f"umbral de {p.troceo_umbral_px} px."
            )
        else:
            escala = p.imgsz / plan["tile"]
            self.lbl_plan.setText(
                f"Primera imagen {wh[0]}×{wh[1]} px → {plan['n_tiles']} tiles de "
                f"{plan['tile']} px con {int(plan['overlap'] * 100)}% de solape, cada "
                f"uno a imgsz {p.imgsz} ({escala:.2f}× la resolución nativa del tile). "
                f"Tarda ~{plan['n_tiles']}× más que un pase único."
            )
