"""Página base con encabezado numerado tipo manual."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame

from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.gui.widgets import StepHeader
from ..state import DetectorState


class DetectorPage(QWidget):
    """Página con header de paso numerado y contenido en un VBox."""

    STEP_N: int = 0
    STEP_TITLE: str = ""
    STEP_DESCRIPTION: str = ""

    def __init__(self, state: DetectorState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setStyleSheet(f"QWidget {{ background: {T.BG_SOFT}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(38, 28, 38, 32)
        outer.setSpacing(20)

        # Header del paso
        outer.addWidget(StepHeader(self.STEP_N, self.STEP_TITLE, self.STEP_DESCRIPTION))

        # Contenedor del cuerpo (tarjetas)
        self.body = QVBoxLayout()
        self.body.setSpacing(18)
        outer.addLayout(self.body)
        outer.addStretch(1)

    # ── helper para tarjetas ──
    def card(self, title: str, icono: str = "") -> tuple[QFrame, QVBoxLayout]:
        """Tarjeta con borde. `icono` es un nombre de core/iconos.py."""
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(20, 16, 20, 18)
        lay.setSpacing(12)

        if title:
            head = QHBoxLayout()
            head.setSpacing(8)
            if icono:
                # `icon` es el nombre de un dibujo de core/iconos.py, no un
                # emoji: asi la cabecera se ve igual en Windows y en macOS y
                # toma el color del tema en vez de venir con el suyo.
                ic = QLabel()
                ic.setPixmap(iconos.pixmap(icono, 17, T.INK2, 1.85))
                ic.setStyleSheet("border: none;")
                head.addWidget(ic)
            t = QLabel(title)
            t.setStyleSheet(
                f"color: {T.INK}; font-size: 13pt; font-weight: 600; border: none;"
            )
            head.addWidget(t)
            head.addStretch(1)
            lay.addLayout(head)

        return frame, lay
