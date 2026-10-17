"""Página 1 — Modelo. Preset rápido + familia (v8/v11) + tamaño + pesos custom."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton,
    QFileDialog, QCheckBox,
)

from ._base import TrainerPage
from polyx.gui import theme as T
from ..state import PRESETS
from polyx.model.i18n import tr


SIZE_DESCRIPTIONS = [
    ("n", "Nano",   "El más liviano y rápido. Útil para probar.", T.OK),
    ("s", "Small",  "Buen balance velocidad/precisión.",          T.OK),
    ("m", "Medium", "RECOMENDADO. Precisión sólida en GPU media.", T.ACCENT),
    ("l", "Large",  "Más preciso, requiere GPU potente.",         T.WARN),
    ("x", "Extra",  "El más preciso, muy lento sin GPU buena.",   T.ERR),
]


class ModeloPage(TrainerPage):
    PAGE_ICON = "modelo"
    PAGE_TITLE = tr("Seleccionar modelo")
    PAGE_DESCRIPTION = (
        tr("Elige el preset según el caso, o personaliza familia y tamaño. El preset "
        "'Balanceado' es el recomendado para la mayoría de proyectos.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Preset rápido ──
        c1, l1 = self.card(tr("Preset rápido"), "rayo")
        info = QLabel(
            tr("Aplica un conjunto de parámetros probados con un clic. Cambiarlo después "
            "en 'Parámetros' marca como Personalizado.")
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(info)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel(tr("Preset:")))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS.keys()))
        self.combo_preset.setCurrentText(self.state.model.preset_name)
        self.combo_preset.currentTextChanged.connect(self._on_preset)
        self.combo_preset.setMinimumWidth(280)
        row.addWidget(self.combo_preset)
        row.addStretch(1)
        l1.addLayout(row)
        self.body.addWidget(c1)

        # ── Familia y tamaño ──
        c2, l2 = self.card(tr("Familia y tamaño"), "modelo")
        row2 = QHBoxLayout()
        row2.setSpacing(20)
        row2.addWidget(QLabel(tr("Familia:")))
        self.combo_family = QComboBox()
        self.combo_family.addItems(["YOLOv8", "YOLOv11"])
        self.combo_family.setCurrentText("YOLOv11" if self.state.model.family == "v11" else "YOLOv8")
        self.combo_family.currentTextChanged.connect(self._on_family)
        row2.addWidget(self.combo_family)

        row2.addWidget(QLabel(tr("Tamaño:")))
        self.combo_size = QComboBox()
        for code, _, _, _ in SIZE_DESCRIPTIONS:
            self.combo_size.addItem(code)
        self.combo_size.setCurrentText(self.state.model.size)
        self.combo_size.currentTextChanged.connect(self._on_size)
        row2.addWidget(self.combo_size)
        row2.addStretch(1)
        l2.addLayout(row2)

        # Descripciones
        for code, name, desc, color in SIZE_DESCRIPTIONS:
            line = QHBoxLayout()
            bullet = QLabel("•")
            bullet.setStyleSheet(f"color: {color}; font-weight: 700; border: none;")
            line.addWidget(bullet)
            text = QLabel(f"<b>{code}</b> → <b>{name}</b> — {tr(desc)}")
            text.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
            text.setTextFormat(Qt.RichText)
            line.addWidget(text, 1)
            l2.addLayout(line)

        # ── Comparar arquitecturas ──
        self.chk_comparar = QCheckBox(
            tr("Entrenar ambas arquitecturas (v8 y v11) con la misma configuración"))
        self.chk_comparar.setChecked(self.state.model.comparar_familias)
        self.chk_comparar.stateChanged.connect(self._on_comparar)
        self.chk_comparar.setStyleSheet(
            f"QCheckBox {{ font-weight: 600; color: {T.ACCENT_TX}; border: none; }}")
        l2.addWidget(self.chk_comparar)

        self.lbl_comparar = QLabel()
        self.lbl_comparar.setWordWrap(True)
        self.lbl_comparar.setStyleSheet(
            f"color: {T.INK3}; font-size: 10pt; border: none;")
        l2.addWidget(self.lbl_comparar)
        self._actualizar_aviso_comparar()

        # ── Pesos que produce el entrenamiento ──
        self.chk_dominio = QCheckBox(
            tr("Guardar también el mejor peso para el conjunto real"))
        self.chk_dominio.setChecked(self.state.params.elegir_por_dominio_real)
        self.chk_dominio.stateChanged.connect(self._on_dominio)
        self.chk_dominio.setStyleSheet(
            f"QCheckBox {{ font-weight: 600; color: {T.OK_TX}; border: none; }}")
        l2.addWidget(self.chk_dominio)

        aviso = QLabel(tr(
            "Al terminar quedan tres pesos en la carpeta del run: "
            "<b>best_sintetico.pt</b>, <b>best_real.pt</b> y <b>last.pt</b> "
            "(este último sirve para reanudar el entrenamiento)."))
        aviso.setWordWrap(True)
        aviso.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l2.addWidget(aviso)

        self.body.addWidget(c2)

        # ── Pesos personalizados ──
        c3, l3 = self.card(tr("Pesos personalizados (opcional)"), "paquete")
        info3 = QLabel(
            tr("Si tienes un .pt ya entrenado y quieres seguir desde ahí, selecciónalo. "
            "Deja vacío para usar el modelo base de la familia/tamaño elegidos.")
        )
        info3.setWordWrap(True)
        info3.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l3.addWidget(info3)

        row3 = QHBoxLayout()
        row3.setSpacing(8)
        self.ed_custom = QLineEdit()
        self.ed_custom.setPlaceholderText(tr("Ruta a un .pt para continuar el entrenamiento…"))
        self.ed_custom.editingFinished.connect(self._on_custom)
        row3.addWidget(self.ed_custom, 1)
        btn = QPushButton("…")
        btn.setFixedWidth(36)
        btn.clicked.connect(self._browse_custom)
        row3.addWidget(btn)
        btn_clr = QPushButton("✕")
        btn_clr.setFixedWidth(36)
        btn_clr.clicked.connect(lambda: (self.ed_custom.clear(), self._on_custom()))
        row3.addWidget(btn_clr)
        l3.addLayout(row3)
        self.body.addWidget(c3)

    def _on_preset(self, name: str):
        self.state.apply_preset(name)
        self.state.model.preset_name = name

    def _on_family(self, txt: str):
        self.state.model.family = "v11" if "11" in txt else "v8"
        self._actualizar_aviso_comparar()
        self.state.model_changed.emit()

    def _on_size(self, txt: str):
        self.state.model.size = txt
        self._actualizar_aviso_comparar()
        self.state.model_changed.emit()

    def _on_dominio(self):
        self.state.params.elegir_por_dominio_real = self.chk_dominio.isChecked()

    def _on_comparar(self):
        self.state.model.comparar_familias = self.chk_comparar.isChecked()
        # La familia elegida arriba queda sin efecto al comparar: se entrenan
        # las dos. Se desactiva el combo para que eso se vea, no se adivine.
        self.combo_family.setEnabled(not self.chk_comparar.isChecked())
        self._actualizar_aviso_comparar()
        self.state.model_changed.emit()

    def _actualizar_aviso_comparar(self):
        """Explica exactamente que se va a entrenar, con los nombres reales."""
        mdl = self.state.model
        if self.chk_comparar.isChecked():
            pesos = " y ".join(mdl.peso_de(f) for f in ("v8", "v11"))
            self.lbl_comparar.setText(tr(
                "Se entrenarán <b>{}</b> uno tras otro, con idénticos "
                "imgsz, batch, épocas, semilla y augmentación. Así la "
                "diferencia de métricas se puede atribuir a la arquitectura "
                "y no a los hiperparámetros.<br>"
                "Tarda <b>el doble</b>: van en secuencia porque comparten GPU. "
                "Al terminar, la comparación sale en el log y en la pestaña "
                "Comparar.").format(pesos))
        else:
            self.lbl_comparar.setText(tr(
                "Se entrenará solo <b>{}</b>. "
                "Marca la casilla para entrenar también la otra familia y "
                "compararlas en igualdad de condiciones.").format(
                    mdl.base_weights_name))

    def _on_custom(self):
        t = self.ed_custom.text().strip()
        self.state.model.custom_weights = Path(t) if t else None
        self.state.model_changed.emit()

    def _browse_custom(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar pesos .pt", "", "Pesos PyTorch (*.pt)"
        )
        if f:
            self.ed_custom.setText(f)
            self._on_custom()
