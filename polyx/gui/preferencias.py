"""Preferences dialog: theme, language and motion.

The three options apply in different ways, and the dialog says so
instead of leaving the user to find out:

* **Theme** — visible instantly in the launcher, which rebuilds itself.
  Modules already open keep theirs, because they are separate processes
  and their stylesheet carries the colors already embedded.
* **Language** — cannot be applied hot: the texts are fixed when each
  widget is built. Modules opened afterwards come out translated.
* **Motion** — immediate everywhere; only a repaint is needed.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

from . import theme as T
from polyx.core.i18n import LANGUAGES, language, set_language, tr


# ──────────────────────────────────────────────────────────────────────
def sample_theme(name_2: str, width_2: int = 104, height_2: int = 62) -> QPixmap:
    """Thumbnail of a theme: bar, card, text and button.

    A color square does not say how the program will look. This
    reproduces the real structure — a top bar, a surface with text and
    an accent button — which is what makes choosing possible without
    having to try all four.
    """
    p = T.PALETTES[name_2]
    px = QPixmap(width_2, height_2)
    px.fill(QColor(p["BG_SOFT"]))
    q = QPainter(px)
    q.setRenderHint(QPainter.Antialiasing)

    # top bar
    q.setPen(Qt.NoPen)
    q.setBrush(QColor(p["BG"]))
    q.drawRect(0, 0, width_2, 15)
    q.setPen(QColor(p["RULE"]))
    q.drawLine(0, 15, width_2, 15)
    q.setPen(Qt.NoPen)
    q.setBrush(QColor(p["ACCENT"]))
    q.drawRoundedRect(6, 4, 8, 8, 2, 2)
    q.setBrush(QColor(p["INK3"]))
    q.drawRoundedRect(19, 7, 26, 3, 1, 1)

    # card with two lines of text
    q.setBrush(QColor(p["BG"]))
    q.setPen(QColor(p["RULE"]))
    q.drawRoundedRect(7, 22, width_2 - 14, height_2 - 30, 4, 4)
    q.setPen(Qt.NoPen)
    q.setBrush(QColor(p["INK"]))
    q.drawRoundedRect(13, 29, 44, 4, 2, 2)
    q.setBrush(QColor(p["INK3"]))
    q.drawRoundedRect(13, 37, 60, 3, 1, 1)
    q.drawRoundedRect(13, 43, 34, 3, 1, 1)

    # accent button
    q.setBrush(QColor(p["ACCENT"]))
    q.drawRoundedRect(width_2 - 38, 40, 28, 12, 3, 3)
    q.end()
    return px


class _CardTheme(QFrame):
    """A theme option: its thumbnail, its name and its state."""

    chosen = Signal(str)

    def __init__(self, name_2: str, label_2: str, active_2: bool):
        super().__init__()
        self._name = name_2
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(label_2)
        self._paint_state(active_2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(9, 9, 9, 8)
        lay.setSpacing(7)

        view = QLabel()
        view.setPixmap(sample_theme(name_2))
        view.setStyleSheet("border: none;")
        lay.addWidget(view, 0, Qt.AlignCenter)

        title = QLabel(label_2)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color: {T.INK if active_2 else T.INK2}; font-size: 9.5pt; border: none;"
            f"font-weight: {'700' if active_2 else '500'};")
        lay.addWidget(title)

    def _paint_state(self, active_2: bool) -> None:
        edge = T.ACCENT if active_2 else T.RULE
        thickness = 2 if active_2 else 1
        self.setStyleSheet(
            f"_TarjetaTema {{ background: {T.BG}; border: {thickness}px solid {edge};"
            f" border-radius: 9px; }}"
            f"_TarjetaTema:hover {{ border-color: {T.ACCENT}; }}")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.chosen.emit(self._name)
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        # With the keyboard the grid is traversed with Tab and chosen with Enter or
        # space, just like any Qt control.
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.chosen.emit(self._name)
            return
        super().keyPressEvent(ev)


# ──────────────────────────────────────────────────────────────────────
class PreferencesDialog(QDialog):
    """Appearance preferences. The theme applies live when picked."""

    theme_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Preferencias"))
        self.setMinimumWidth(520)
        self._build()

    # ── Build ──────────────────────────────────────────────────────
    def _build(self) -> None:
        # It is rebuilt entirely when the theme changes: it is a small dialog and
        # rebuilding it is cheaper than touching every stylesheet.
        previous = self.layout()
        if previous is not None:
            QWidget().setLayout(previous)

        self.setStyleSheet(T.GLOBAL_QSS + f"QDialog {{ background: {T.BG_SOFT}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(6)

        root.addWidget(self._title(tr("Tema")))
        root.addWidget(self._clue(
            tr("Se aplica al instante aquí. Los módulos ya abiertos conservan el suyo.")))

        grid = QGridLayout()
        grid.setSpacing(10)
        current = T.theme()
        for i, name_2 in enumerate(T.PALETTES):
            card = _CardTheme(name_2, tr(T.NAMES_THEME[name_2]), name_2 == current)
            card.chosen.connect(self._choose_theme)
            grid.addWidget(card, i // 4, i % 4)
        root.addLayout(grid)
        root.addSpacing(14)

        root.addWidget(self._title(tr("Idioma")))
        root.addWidget(self._clue(
            tr("Los módulos que abras a partir de ahora salen en el idioma nuevo.")))
        row = QHBoxLayout()
        row.setSpacing(8)
        self._buttons_language: dict[str, QPushButton] = {}
        for cod, label_2 in LANGUAGES.items():
            b = QPushButton(label_2)
            b.setCheckable(True)
            b.setChecked(cod == language())
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumWidth(112)
            b.setStyleSheet(self._qss_toggle())
            b.clicked.connect(lambda _=False, c=cod: self._choose_language(c))
            self._buttons_language[cod] = b
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)
        root.addSpacing(16)

        root.addWidget(self._title(tr("Movimiento")))
        self._chk_anim = QCheckBox(tr("Animar el panel del microscopio"))
        self._chk_anim.setChecked(T.animations())
        self._chk_anim.setCursor(Qt.PointingHandCursor)
        self._chk_anim.setStyleSheet(
            f"QCheckBox {{ color: {T.INK2}; background: transparent; spacing: 8px; }}")
        self._chk_anim.toggled.connect(T.set_animations)
        root.addWidget(self._chk_anim)
        root.addWidget(self._clue(
            tr("Apágalo si el movimiento te molesta o si el equipo va justo. "
               "El panel queda en un fotograma fijo.")))

        root.addSpacing(18)
        pie = QHBoxLayout()
        pie.addStretch(1)
        close = QPushButton(tr("Cerrar"))
        close.setObjectName("primary")
        close.setCursor(Qt.PointingHandCursor)
        close.setDefault(True)
        close.clicked.connect(self.accept)
        pie.addWidget(close)
        root.addLayout(pie)

    # ── Pieces ─────────────────────────────────────────────────────
    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {T.INK}; font-size: 11.5pt; font-weight: 650;"
            f" border: none; background: transparent;")
        return lbl

    def _clue(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {T.INK3}; font-size: 9pt; border: none;"
            f" background: transparent; margin-bottom: 6px;")
        return lbl

    def _qss_toggle(self) -> str:
        return (
            f"QPushButton {{ background: {T.BG}; color: {T.INK2};"
            f" border: 1px solid {T.RULE}; border-radius: 6px; padding: 7px 14px; }}"
            f"QPushButton:hover {{ border-color: {T.ACCENT}; }}"
            f"QPushButton:checked {{ background: {T.ACCENT}; color: {T.ON_ACCENT};"
            f" border-color: {T.ACCENT}; font-weight: 600; }}")

    # ── Actions ────────────────────────────────────────────────────
    def _choose_theme(self, name_2: str) -> None:
        if name_2 == T.theme():
            return
        T.set_theme(name_2)
        self._build()          # the dialog itself is repainted with the new palette
        self.theme_changed.emit(name_2)

    def _choose_language(self, code: str) -> None:
        for cod, button_2 in self._buttons_language.items():
            button_2.setChecked(cod == code)
        if code != language():
            set_language(code)
