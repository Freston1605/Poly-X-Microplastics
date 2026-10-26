"""Page 1 — `Modelo`. Quick preset + family (v8/v11) + size + custom weights."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)

from polyx.core.i18n import tr
from polyx.gui import theme as T

from ..state import PRESETS
from ._base import TrainerPage

SIZE_DESCRIPTIONS = [
    ("n", "Nano",   "El más liviano y rápido. Útil para probar.", T.OK),
    ("s", "Small",  "Buen balance velocidad/precisión.",          T.OK),
    ("m", "Medium", "RECOMENDADO. Precisión sólida en GPU media.", T.ACCENT),
    ("l", "Large",  "Más preciso, requiere GPU potente.",         T.WARN),
    ("x", "Extra",  "El más preciso, muy lento sin GPU buena.",   T.ERR),
]


class ModelPage(TrainerPage):
    PAGE_ICON = "modelo"
    PAGE_TITLE = tr("Seleccionar modelo")
    PAGE_DESCRIPTION = (
        tr("Elige el preset según el caso, o personaliza familia y tamaño. El preset "
        "'Balanceado' es el recomendado para la mayoría de proyectos.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Quick preset ──
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

        # ── Family and size ──
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

        # Descriptions
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

        # ── Architecture check ──
        self.chk_compare = QCheckBox(
            tr("Entrenar ambas arquitecturas (v8 y v11) con la misma configuración"))
        self.chk_compare.setChecked(self.state.model.compare_families)
        self.chk_compare.stateChanged.connect(self._on_compare)
        self.chk_compare.setStyleSheet(
            f"QCheckBox {{ font-weight: 600; color: {T.ACCENT_TX}; border: none; }}")
        l2.addWidget(self.chk_compare)

        self.lbl_compare = QLabel()
        self.lbl_compare.setWordWrap(True)
        self.lbl_compare.setStyleSheet(
            f"color: {T.INK3}; font-size: 10pt; border: none;")
        l2.addWidget(self.lbl_compare)
        self._update_compare_notice()

        # ── Weights produced by training ──
        self.chk_domain = QCheckBox(
            tr("Guardar también el mejor peso para el conjunto real"))
        self.chk_domain.setChecked(self.state.params.choose_by_real_domain)
        self.chk_domain.stateChanged.connect(self._on_domain)
        self.chk_domain.setStyleSheet(
            f"QCheckBox {{ font-weight: 600; color: {T.OK_TX}; border: none; }}")
        l2.addWidget(self.chk_domain)

        warning = QLabel(tr(
            "Al terminar quedan tres pesos en la carpeta del run: "
            "<b>best_sintetico.pt</b>, <b>best_real.pt</b> y <b>last.pt</b> "
            "(este último sirve para reanudar el entrenamiento)."))
        warning.setWordWrap(True)
        warning.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l2.addWidget(warning)

        self.body.addWidget(c2)

        # ── Custom weights ──
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
        self._update_compare_notice()
        self.state.model_changed.emit()

    def _on_size(self, txt: str):
        self.state.model.size = txt
        self._update_compare_notice()
        self.state.model_changed.emit()

    def _on_domain(self):
        self.state.params.choose_by_real_domain = self.chk_domain.isChecked()

    def _on_compare(self):
        self.state.model.compare_families = self.chk_compare.isChecked()
        # The family selected above is ignored when checking: both are trained
        # both. The combo is disabled so that this is seen, not guessed.
        self.combo_family.setEnabled(not self.chk_compare.isChecked())
        self._update_compare_notice()
        self.state.model_changed.emit()

    def _update_compare_notice(self):
        """Explains exactly what is going to be trained, with the real names."""
        mdl = self.state.model
        if self.chk_compare.isChecked():
            weights = " y ".join(mdl.weight_of(f) for f in ("v8", "v11"))
            self.lbl_compare.setText(tr(
                "Se entrenarán <b>{}</b> uno tras otro, con idénticos "
                "imgsz, batch, épocas, semilla y augmentación. Así la "
                "diferencia de métricas se puede atribuir a la arquitectura "
                "y no a los hiperparámetros.<br>"
                "Tarda <b>el doble</b>: van en secuencia porque comparten GPU. "
                "Al terminar, la comparación sale en el log y en la pestaña "
                "Comparar.").format(weights))
        else:
            self.lbl_compare.setText(tr(
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
