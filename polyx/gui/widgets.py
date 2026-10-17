"""Widgets reutilizables (logo, headers numerados, tarjetas KPI)."""
from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPainterPath
from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy
)

from . import theme as T


class LogoBadge(QWidget):
    """Cuadrado azul redondeado con la 'P' blanca + texto al lado."""

    def __init__(self, title: str = "POLY-X", subtitle: str = "", size: int = 36):
        super().__init__()
        self._size = size
        self._title = title
        self._subtitle = subtitle
        self.setMinimumHeight(size + 12)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(10)

        self._badge = QLabel()
        self._badge.setFixedSize(size, size)
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border-radius: 6px; "
            f"font-size: {int(size*0.55)}pt; font-weight: 700;"
        )
        self._badge.setText("P")
        lay.addWidget(self._badge)

        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(
            f"color: {T.INK}; font-weight: 700; font-size: 12pt; letter-spacing: 0.5px;"
        )
        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        text_col.addWidget(self._title_lbl)
        if subtitle:
            text_col.addWidget(self._sub_lbl)
        lay.addLayout(text_col, 1)


class StepHeader(QWidget):
    """Encabezado tipo '① Título' con círculo azul + descripción debajo."""

    def __init__(self, number: int, title: str, description: str = ""):
        super().__init__()
        self._n = number
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(14)

        circle = QLabel(str(number))
        circle.setFixedSize(38, 38)
        circle.setAlignment(Qt.AlignCenter)
        circle.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border-radius: 19px; "
            f"font-size: 14pt; font-weight: 700;"
        )
        row.addWidget(circle)

        title_lbl = QLabel(title)
        title_lbl.setProperty("role", "h2")
        title_lbl.setStyleSheet(f"color: {T.INK}; font-size: 20pt; font-weight: 600;")
        row.addWidget(title_lbl, 1)

        lay.addLayout(row)

        if description:
            desc = QLabel(description)
            desc.setWordWrap(True)
            desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; margin-left: 52px;")
            lay.addWidget(desc)


class HLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setStyleSheet(f"color: {T.RULE_SOFT}; background: {T.RULE_SOFT}; max-height: 1px;")


class KPICard(QFrame):
    """Tarjeta KPI grande: título + valor numérico + barra de color inferior."""

    def __init__(self, label: str, color: str = T.ACCENT):
        super().__init__()
        self._color = color
        self.setMinimumHeight(110)
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(8)

        self.lbl_title = QLabel(label)
        self.lbl_title.setWordWrap(True)   # etiquetas largas (p. ej. "Verdaderos Positivos")
        self.lbl_title.setStyleSheet(
            f"color: {T.INK3}; font-size: 10pt; font-weight: 500; border: none;"
        )
        self.lbl_value = QLabel("—")
        self.lbl_value.setStyleSheet(
            f"color: {T.INK}; font-size: 22pt; font-weight: 700; border: none;"
        )
        bar = QFrame()
        bar.setFixedHeight(3)
        bar.setStyleSheet(f"background: {color}; border-radius: 2px; border: none;")

        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_value)
        lay.addStretch(1)
        lay.addWidget(bar)

    def set_value(self, value):
        if value is None:
            self.lbl_value.setText("—")
        else:
            self.lbl_value.setText(str(value))
