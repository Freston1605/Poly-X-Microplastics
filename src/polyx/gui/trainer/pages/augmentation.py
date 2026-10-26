"""Page 4 — `Augmentación`. Presets (`Suave`/`Medio`/`Fuerte`) + manual sliders."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QComboBox, QSlider,
    QDoubleSpinBox,
)

from ._base import TrainerPage
from polyx.gui import theme as T
from ..state import AUG_LEVELS
from polyx.core.i18n import tr


def _slider_row(label: str, value: float, max_val: float, on_change) -> tuple[QHBoxLayout, QSlider, QLabel]:
    h = QHBoxLayout(); h.setSpacing(8)
    lbl = QLabel(label); lbl.setMinimumWidth(120)
    lbl.setStyleSheet(f"color: {T.INK2}; border: none;")
    h.addWidget(lbl)
    s = QSlider(Qt.Horizontal)
    s.setRange(0, 1000)
    s.setValue(int(value / max_val * 1000) if max_val else 0)
    h.addWidget(s, 1)
    val_lbl = QLabel(f"{value:.3f}")
    val_lbl.setMinimumWidth(60)
    val_lbl.setStyleSheet(f"color: {T.INK}; font-weight: 600; border: none;")
    h.addWidget(val_lbl)

    def _cb(v):
        real = v / 1000.0 * max_val
        val_lbl.setText(f"{real:.3f}")
        on_change(real)
    s.valueChanged.connect(_cb)
    return h, s, val_lbl


class AugmentationPage(TrainerPage):
    PAGE_ICON = "augmentacion"
    PAGE_TITLE = tr("Augmentación de datos")
    PAGE_DESCRIPTION = (
        tr("Aumenta artificialmente la variedad del dataset. Útil para datasets pequeños "
        "(< 500 imágenes). Demasiado aug = el modelo no converge; muy poco = sobreajuste.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Preset ──
        c1, l1 = self.card(tr("Nivel de augmentación"), "ajustes")
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(QLabel(tr("Nivel:")))
        self.combo = QComboBox()
        self.combo.addItems(list(AUG_LEVELS.keys()))
        self.combo.setCurrentText(state.aug.level)
        self.combo.currentTextChanged.connect(self._apply_level)
        self.combo.setMinimumWidth(180)
        row.addWidget(self.combo); row.addStretch(1)
        l1.addLayout(row)

        legend = QLabel(
            tr("<b>Ninguno</b>: sin transformaciones. &nbsp; "
            "<b>Suave</b>: solo flips y jitter HSV. &nbsp; "
            "<b>Medio</b> (recomendado): + mosaic, mixup ligero. &nbsp; "
            "<b>Fuerte</b>: + copy-paste agresivo.")
        )
        legend.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        legend.setWordWrap(True); legend.setTextFormat(Qt.RichText)
        l1.addWidget(legend)
        self.body.addWidget(c1)

        # ── Manual sliders ──
        c2, l2 = self.card(tr("Sliders manuales (avanzado)"), "ajustes")

        sets = [
            ("HSV-H (hue):",      "hsv_h",      0.1),
            ("HSV-S (saturación):", "hsv_s",    1.0),
            ("HSV-V (valor):",    "hsv_v",      1.0),
            ("Fliplr (horizontal):", "fliplr",  1.0),
            ("Mosaic:",           "mosaic",     1.0),
            ("Mixup:",            "mixup",      1.0),
            ("Copy-paste:",       "copy_paste", 1.0),
        ]
        self.sliders: dict[str, QSlider] = {}
        self.value_lbls: dict[str, QLabel] = {}
        for label, attr, mx in sets:
            row, s, vl = _slider_row(label, getattr(state.aug, attr), mx,
                                     lambda v, a=attr: self._on_slider(a, v))
            l2.addLayout(row)
            self.sliders[attr] = s
            self.value_lbls[attr] = vl
        self.body.addWidget(c2)

        # Hints
        c3, l3 = self.card(tr("Recomendaciones"), "idea")
        tips = QLabel(
            tr("• <b>Dataset pequeño (< 500 imgs)</b>: usa Fuerte. Más variedad sintética compensa pocos ejemplos.<br>"
            "• <b>Microplásticos PET/PP/LDPE</b>: <b>NO</b> uses HSV-H alto: los colores son la pista principal.<br>"
            "• <b>Copy-paste</b> funciona muy bien si el fondo es uniforme (como un filtro).<br>"
            "• Si el modelo no converge (mAP no sube), baja el nivel a Suave o Ninguno.")
        )
        tips.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none; line-height: 1.6;")
        tips.setTextFormat(Qt.RichText); tips.setWordWrap(True)
        l3.addWidget(tips)
        self.body.addWidget(c3)

        # Subscribe to external changes (apply preset from `Modelo`)
        self.state.aug_changed.connect(self._reload_from_state)

    def _reload_from_state(self):
        a = self.state.aug
        self.combo.blockSignals(True)
        self.combo.setCurrentText(a.level)
        self.combo.blockSignals(False)
        for attr, slider in self.sliders.items():
            mx = 0.1 if attr == "hsv_h" else 1.0
            val = getattr(a, attr)
            slider.blockSignals(True)
            slider.setValue(int(val / mx * 1000) if mx else 0)
            slider.blockSignals(False)
            self.value_lbls[attr].setText(f"{val:.3f}")

    def _apply_level(self, name: str):
        if name not in AUG_LEVELS: return
        d = AUG_LEVELS[name]
        a = self.state.aug
        a.level = name
        for k, v in d.items():
            setattr(a, k, v)
        self._reload_from_state()
        self.state.aug_changed.emit()

    def _on_slider(self, attr: str, value: float):
        setattr(self.state.aug, attr, value)
        # touching a slider manually switches to `Personalizado`
        # (the level is kept visually; it is not critical)
