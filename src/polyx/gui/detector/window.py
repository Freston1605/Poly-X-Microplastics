"""Main window of the Detector — sidebar on the left + pages on the right."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
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

from polyx.core.i18n import _, lazy
from polyx.gui import i18n_qt, iconos, marca
from polyx.gui import theme as T
from polyx.gui.widgets import LogoBadge

from .state import DetectorState

# ────────────────────────────────────────────────────────────────────
SIDEBAR_ITEMS = [
    # (icon name in polyx/gui/iconos.py, label)
    ("modelo", lazy("Models")),
    ("imagenes", lazy("Images")),
    ("editar", lazy("Manual GT")),
    ("ajustes", lazy("Parameters")),
    ("ejecutar", lazy("Run")),
    ("resultados", lazy("Results")),
    ("errores", lazy("Errors")),
    ("comparar", lazy("Compare")),
    ("reporte", lazy("Report")),
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
        self.setIcon(iconos.switchable_icon(icon_name, 17, T.INK3, T.ACCENT_D_TX))
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
        i18n_qt.activate()
        marca.titular(self, "detector")
        self.resize(1280, 820)

        self.state = DetectorState()
        self._retranslation_pending = False

        self._build_ui()

        # Connect global signals to reflect state in the footer. Connected once:
        # a language change rebuilds the widgets, not the window object.
        self.state.run_started.connect(lambda: self._set_status("● Ejecutando…", T.ACCENT))
        self.state.run_finished.connect(lambda: self._set_status("● Finalizado", T.OK))
        self.state.run_aborted.connect(lambda: self._set_status("● Detenido", T.WARN))
        self.state.run_finished.connect(self._apply_pending_retranslation)
        self.state.run_aborted.connect(self._apply_pending_retranslation)

    # ── Language ───────────────────────────────────────────────
    def changeEvent(self, event):
        if event.type() == QEvent.LanguageChange:
            self.retranslate()
        super().changeEvent(event)

    def retranslate(self) -> None:
        """Rebuild the pages so every text comes from the new catalog.

        A detection in progress owns a worker thread and the page that
        listens to it, so the rebuild waits for it to finish instead of
        orphaning the thread mid-run.
        """
        if self.state.is_running():
            self._retranslation_pending = True
            return
        index = self.stack.currentIndex()
        self._build_ui(index)
        marca.titular(self, "detector")
        self._refresh_pages()

    def _apply_pending_retranslation(self) -> None:
        if self._retranslation_pending:
            self._retranslation_pending = False
            self.retranslate()

    def _build_ui(self, index: int = 0) -> None:
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(f"QFrame {{ background: {T.BG}; border-right: 1px solid {T.RULE}; }}")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(14, 16, 14, 16)
        sb.setSpacing(4)

        # Logo at the top with subtitle "Detector v2"
        sb.addWidget(LogoBadge("POLY-X", "Detector v2"))
        sb.addSpacing(12)
        sep = QFrame()
        sep.setFixedHeight(1)
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
        self.lbl_status = QLabel(_("● Ready"))
        self.lbl_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 9pt; padding: 4px 8px;")
        sb.addWidget(self.lbl_status)

        # Authorship credit
        author = QLabel(_("Designed by\nCristofher Ferrada\n2026"))
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
        index = max(0, min(index, len(self.sidebar_buttons) - 1))
        self.sidebar_buttons[index].setChecked(True)
        self.stack.setCurrentIndex(index)

    def _refresh_pages(self) -> None:
        """Re-read the state so dynamic text is not left in the old language."""
        for position in range(self.stack.count()):
            page = self.stack.widget(position).widget()
            for name in ("refresh", "_refresh_table"):
                method = getattr(page, name, None)
                if callable(method):
                    method()

    def _build_pages(self):
        # Late import to avoid cycles
        from .pages.compare import ComparePage
        from .pages.errors import ErrorsPage
        from .pages.execute import RunPage
        from .pages.gt_manual import GTManualPage
        from .pages.images import ImagesPage
        from .pages.models import ModelsPage
        from .pages.parameters import ParametersPage
        from .pages.report import ReportPage
        from .pages.results import ResultsPage

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
