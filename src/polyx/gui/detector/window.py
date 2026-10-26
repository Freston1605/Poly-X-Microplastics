"""Main window of the Detector — sidebar on the left + pages on the right."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from polyx.core.i18n import tr
from polyx.gui import iconos, marca
from polyx.gui import theme as T
from polyx.gui.widgets import LogoBadge

from .state import DetectorState

# ────────────────────────────────────────────────────────────────────
SIDEBAR_ITEMS = [
    # (icon name in polyx/gui/iconos.py, label)
    ("modelo", tr("Modelos")),
    ("imagenes", tr("Imágenes")),
    ("editar", tr("GT manual")),
    ("ajustes", tr("Parámetros")),
    ("ejecutar", tr("Ejecutar")),
    ("resultados", tr("Resultados")),
    ("errores", tr("Errores")),
    ("comparar", tr("Comparar")),
    ("reporte", tr("Reporte")),
]


# ────────────────────────────────────────────────────────────────────
class SidebarButton(QPushButton):
    """Sidebar item: vector icon + text. Pill-shaped while active.

    The icon is passed as a name and not as an emoji: an emoji is drawn by the
    system font, changes shape between Windows and macOS and arrives in a fixed
    color, so it cannot be tinted with the accent color when the page is active.
    """

    def __init__(self, icon_name: str, text: str):
        super().__init__(f"   {text}")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(38)
        self.setIcon(iconos.switchable_icon(
            icon_name, 17, T.INK3, T.ACCENT_D_TX))
        self.setIconSize(QSize(17, 17))
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {T.INK2};
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                text-align: left;
                font-size: 10.5pt;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {T.RULE_SOFT};
                color: {T.INK};
            }}
            QPushButton:checked {{
                background: {T.on_background(T.ACCENT, 0.16)};
                color: {T.ACCENT_D_TX};
                font-weight: 600;
            }}
            """
        )

# ────────────────────────────────────────────────────────────────────
class DetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        marca.titular(self, "detector")
        self.resize(1280, 820)
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")

        self.state = DetectorState()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border-right: 1px solid {T.RULE}; }}"
        )
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(14, 16, 14, 16)
        sb.setSpacing(4)

        # Logo at the top with subtitle "Detector v2"
        sb.addWidget(LogoBadge("POLY-X", "Detector v2"))
        sb.addSpacing(12)
        sep = QFrame(); sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {T.RULE_SOFT};")
        sb.addWidget(sep)
        sb.addSpacing(8)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.sidebar_buttons: list[SidebarButton] = []
        for icon, label in SIDEBAR_ITEMS:
            b = SidebarButton(icon, label)
            sb.addWidget(b)
            self.btn_group.addButton(b, len(self.sidebar_buttons))
            self.sidebar_buttons.append(b)

        sb.addStretch(1)

        # Sidebar footer: run status
        self.lbl_status = QLabel(tr("● Listo"))
        self.lbl_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 9pt; padding: 4px 8px;")
        sb.addWidget(self.lbl_status)

        # Authorship credit
        author = QLabel(tr("Diseñado por\nCristofher Ferrada\n2026"))
        author.setAlignment(Qt.AlignCenter)
        author.setStyleSheet(
            f"color: {T.INK3}; font-size: 8.5pt; padding: 8px; "
            f"border-top: 1px solid {T.RULE_SOFT}; margin-top: 6px;"
        )
        sb.addWidget(author)

        root.addWidget(sidebar)

        # ── Content area ──
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"QStackedWidget {{ background: {T.BG_SOFT}; }}")
        root.addWidget(self.stack, 1)

        # ── Create each page ──
        self._build_pages()

        # Connect navigation
        self.btn_group.idClicked.connect(self._on_sidebar_clicked)
        self.sidebar_buttons[0].setChecked(True)
        self.stack.setCurrentIndex(0)

        # Connect global signals to reflect state in the footer
        self.state.run_started.connect(lambda: self._set_status("● Ejecutando…", T.ACCENT))
        self.state.run_finished.connect(lambda: self._set_status("● Finalizado", T.OK))
        self.state.run_aborted.connect(lambda: self._set_status("● Detenido", T.WARN))

    def _build_pages(self):
        # Late import to avoid cycles
        from .pages.comparar import ComparePage
        from .pages.ejecutar import RunPage
        from .pages.errores import ErrorsPage
        from .pages.gt_manual import GTManualPage
        from .pages.imagenes import ImagesPage
        from .pages.modelos import ModelsPage
        from .pages.parametros import ParametersPage
        from .pages.reporte import ReportPage
        from .pages.resultados import ResultsPage

        pages = [
            ModelsPage(self.state),
            ImagesPage(self.state),
            GTManualPage(self.state),
            ParametersPage(self.state),
            RunPage(self.state),
            ResultsPage(self.state),
            ErrorsPage(self.state),
            ComparePage(self.state),
            ReportPage(self.state),
        ]
        for p in pages:
            wrapper = QScrollArea()
            wrapper.setWidgetResizable(True)
            wrapper.setStyleSheet("QScrollArea { border: none; background: transparent; }")
            wrapper.setWidget(p)
            self.stack.addWidget(wrapper)

    def _on_sidebar_clicked(self, idx: int):
        self.stack.setCurrentIndex(idx)

    def _set_status(self, text: str, color: str):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt; padding: 4px 8px;")
