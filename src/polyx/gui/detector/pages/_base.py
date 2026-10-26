"""Base page with a numbered, manual-style header."""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from polyx.gui import iconos
from polyx.gui import theme as T
from polyx.gui.widgets import StepHeader

from ..state import DetectorState


class DetectorPage(QWidget):
    """Page with a numbered step header and content in a VBox."""

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

        # Step header
        outer.addWidget(StepHeader(self.STEP_N, self.STEP_TITLE, self.STEP_DESCRIPTION))

        # Body container (cards)
        self.body = QVBoxLayout()
        self.body.setSpacing(18)
        outer.addLayout(self.body)
        outer.addStretch(1)

    # ── card helper ──
    def card(self, title: str, icon: str = "") -> tuple[QFrame, QVBoxLayout]:
        """Card with a border. `icono` is a name from core/iconos.py."""
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
            if icon:
                # `icon` is the name of a drawing in polyx/gui/iconos.py, not an
                # emoji: this way the header looks the same on Windows and macOS and
                # takes the theme color instead of coming with its own.
                ic = QLabel()
                ic.setPixmap(iconos.pixmap(icon, 17, T.INK2, 1.85))
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
