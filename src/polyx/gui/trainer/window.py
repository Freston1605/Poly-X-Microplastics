"""Main window of the Trainer. Sidebar with 9 items + pages on the right."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
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
)

from polyx.core.i18n import _, lazy
from polyx.gui import i18n_qt, iconos, marca
from polyx.gui import theme as T
from polyx.gui.widgets import LogoBadge, reset_central_widget

from .state import TrainerState

SIDEBAR_ITEMS = [
    # (icon name in polyx/gui/iconos.py, label)
    ("modelo", lazy("Model")),
    ("dataset", lazy("Dataset")),
    ("ajustes", lazy("Parameters")),
    ("augmentacion", lazy("Augmentation")),
    ("ejecutar", lazy("Train")),
    ("evaluar", lazy("Evaluate")),
    ("comparar", lazy("Compare")),
    ("exportar", lazy("Export")),
    ("reporte", lazy("Summary")),
]


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


class TrainerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        i18n_qt.activate()
        marca.titular(self, "entrenador")
        self.resize(1320, 840)

        self.state = TrainerState()
        self._retranslation_pending = False

        self._build_ui()

        # Status according to training state. Connected once: a language
        # change rebuilds the widgets, not the window object.
        self.state.train_started.connect(lambda: self._set_status("● Entrenando…", T.ACCENT))
        self.state.train_finished.connect(lambda _unused: self._set_status("● Finalizado", T.OK))
        self.state.train_aborted.connect(lambda: self._set_status("● Detenido", T.WARN))
        self.state.train_failed.connect(lambda _unused: self._set_status("● Error", T.ERR))
        self.state.train_finished.connect(lambda _unused: self._apply_pending_retranslation())
        self.state.train_aborted.connect(self._apply_pending_retranslation)
        self.state.train_failed.connect(lambda _unused: self._apply_pending_retranslation())

    # ── Language ───────────────────────────────────────────────
    def changeEvent(self, event):
        if event.type() == QEvent.LanguageChange:
            # Rebuild after the event returns: deleting the widget tree
            # inside changeEvent is not safe.
            QTimer.singleShot(0, self.retranslate)
        super().changeEvent(event)

    def retranslate(self) -> None:
        """Rebuild the pages so every text comes from the new catalog.

        A training in progress owns a worker thread and the page that
        listens to it, so the rebuild waits for it to finish instead of
        orphaning the thread mid-run.
        """
        if not i18n_qt.is_alive(self):
            return
        if self.state.is_running():
            self._retranslation_pending = True
            return
        index = self.stack.currentIndex()
        self._build_ui(index)
        marca.titular(self, "entrenador")
        self._refresh_pages()

    def _apply_pending_retranslation(self) -> None:
        if self._retranslation_pending:
            self._retranslation_pending = False
            self.retranslate()

    def _build_ui(self, index: int = 0) -> None:
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")

        central = reset_central_widget(self)
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

        sb.addWidget(LogoBadge("POLY-X", "YOLO Trainer"))
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

        # ── Content ──
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"QStackedWidget {{ background: {T.BG_SOFT}; }}")
        root.addWidget(self.stack, 1)

        self._build_pages()
        self.btn_group.idClicked.connect(self.stack.setCurrentIndex)
        index = max(0, min(index, len(self.sidebar_buttons) - 1))
        self.sidebar_buttons[index].setChecked(True)
        self.stack.setCurrentIndex(index)

    def _refresh_pages(self) -> None:
        """Re-read the state so dynamic text is not left in the old language."""
        for position in range(self.stack.count()):
            page = self.stack.widget(position).widget()
            refresh = getattr(page, "refresh", None)
            if callable(refresh):
                refresh()

    def _build_pages(self):
        from .pages.augmentation import AugmentationPage
        from .pages.compare import ComparePage
        from .pages.dataset import DatasetPage
        from .pages.evaluate import EvaluatePage
        from .pages.export import ExportPage
        from .pages.model import ModelPage
        from .pages.parameters import ParametersPage
        from .pages.summary import ReportPage
        from .pages.train import TrainPage

        pages = [
            ModelPage(self.state),
            DatasetPage(self.state),
            ParametersPage(self.state),
            AugmentationPage(self.state),
            TrainPage(self.state),
            EvaluatePage(self.state),
            ComparePage(self.state),
            ExportPage(self.state),
            ReportPage(self.state),
        ]
        for p in pages:
            wrapper = QScrollArea()
            wrapper.setWidgetResizable(True)
            wrapper.setStyleSheet("QScrollArea { border: none; background: transparent; }")
            wrapper.setWidget(p)
            self.stack.addWidget(wrapper)

    def _set_status(self, text: str, color: str):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt; padding: 4px 8px;")
