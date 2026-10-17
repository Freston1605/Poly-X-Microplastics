"""Ventana principal del Detector — sidebar a la izquierda + páginas a la derecha."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QFrame, QSizePolicy, QButtonGroup, QScrollArea,
)

from polyx.gui import theme as T
from polyx.gui import marca
from polyx.gui import iconos
from polyx.gui.widgets import LogoBadge
from .state import DetectorState
from polyx.model.i18n import tr


# ────────────────────────────────────────────────────────────────────
SIDEBAR_ITEMS = [
    # (nombre del icono en core/iconos.py, etiqueta)
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
    """Item del sidebar: icono vectorial + texto. Pastilla al estar activo.

    El icono se pasa como nombre y no como emoji: un emoji lo dibuja la fuente
    del sistema, cambia de forma entre Windows y macOS y llega en color fijo,
    de modo que no puede tenirse de acento cuando la pagina esta activa.
    """

    def __init__(self, icono_nombre: str, text: str):
        super().__init__(f"   {text}")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(38)
        self.setIcon(iconos.icono_conmutable(
            icono_nombre, 17, T.INK3, T.ACCENT_D_TX))
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
                background: {T.sobre_fondo(T.ACCENT, 0.16)};
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

        # Logo arriba con subtítulo "Detector v2"
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

        # Pie del sidebar: estado del run
        self.lbl_status = QLabel(tr("● Listo"))
        self.lbl_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 9pt; padding: 4px 8px;")
        sb.addWidget(self.lbl_status)

        # Crédito de autoría
        author = QLabel(tr("Diseñado por\nCristofher Ferrada\n2026"))
        author.setAlignment(Qt.AlignCenter)
        author.setStyleSheet(
            f"color: {T.INK3}; font-size: 8.5pt; padding: 8px; "
            f"border-top: 1px solid {T.RULE_SOFT}; margin-top: 6px;"
        )
        sb.addWidget(author)

        root.addWidget(sidebar)

        # ── Área de contenido ──
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"QStackedWidget {{ background: {T.BG_SOFT}; }}")
        root.addWidget(self.stack, 1)

        # ── Crear cada página ──
        self._build_pages()

        # Conectar navegación
        self.btn_group.idClicked.connect(self._on_sidebar_clicked)
        self.sidebar_buttons[0].setChecked(True)
        self.stack.setCurrentIndex(0)

        # Conectar señales globales para reflejar estado en pie
        self.state.run_started.connect(lambda: self._set_status("● Ejecutando…", T.ACCENT))
        self.state.run_finished.connect(lambda: self._set_status("● Finalizado", T.OK))
        self.state.run_aborted.connect(lambda: self._set_status("● Detenido", T.WARN))

    def _build_pages(self):
        # Importación tardía para evitar ciclos
        from .pages.modelos import ModelosPage
        from .pages.imagenes import ImagenesPage
        from .pages.gt_manual import GTManualPage
        from .pages.parametros import ParametrosPage
        from .pages.ejecutar import EjecutarPage
        from .pages.resultados import ResultadosPage
        from .pages.errores import ErroresPage
        from .pages.comparar import CompararPage
        from .pages.reporte import ReportePage

        pages = [
            ModelosPage(self.state),
            ImagenesPage(self.state),
            GTManualPage(self.state),
            ParametrosPage(self.state),
            EjecutarPage(self.state),
            ResultadosPage(self.state),
            ErroresPage(self.state),
            CompararPage(self.state),
            ReportePage(self.state),
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
