"""Module 1 — Launcher (main menu). Reproduces Figure 1 of the manual.

Hero with kicker, 'Poly-X analytics' title, subtitle, 4 KPI chips,
and 4 large module cards (Detector, Trainer, Labeler, Viewer).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from polyx import __version__
from polyx.core.i18n import language, tr
from polyx.core.platform import launch_updater, open_in_system
from polyx.gui import iconos, marca
from polyx.gui import theme as T
from polyx.gui.microscope_panel import MicroscopePanel
from polyx.gui.preferencias import PreferencesDialog
from polyx.gui.updates import UPDATER, SearchUpdates, can_update
from polyx.gui.widgets import HLine, LogoBadge

# ──────────────────────────────────────────────────────────────────────
# Instrumented startup: the launcher leaves a JSONL trace in `app.jsonl`.
# The observability imports are lazy, inside the functions,
# so that a failure in them neither blocks nor delays the window launch.
# ──────────────────────────────────────────────────────────────────────


def _register_boot() -> None:
    """Emits ``bootstrap.finish`` with version, python and system onto app.jsonl.

    Never blocks the launch: if HOME is read-only or the log directory
    cannot be created, the launcher opens as usual anyway.
    Observability must never be worse than having none.
    """
    try:
        from polyx.core import observability
        from polyx.core.platform import name_system

        log = observability.logger_for("app")
        observability.event(
            log,
            "bootstrap.finish",
            version=__version__,
            python=sys.version.split()[0],
            sistema=name_system(),
        )
    except Exception:  # noqa: BLE001, S110
        pass


def _spawn_module(dotted: str) -> bool:
    """Spawns the module in a separate process; True if it started, False if not.

    Returns instead of raising: that way the window decides how to warn
    the user and the observability drag (module.launch /
    module.launch.error in app.jsonl) cannot turn into a crash of the
    interface.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec(dotted)
        if spec is None:
            return False
        env = dict(os.environ)
        src_dir = str(Path(__file__).resolve().parents[3] / "src")
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", dotted],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=env,
        )
    except Exception as e:  # noqa: BLE001
        try:
            from polyx.core import observability

            observability.event(
                observability.logger_for("app"),
                "module.launch.error",
                level=logging.WARNING,
                modulo=dotted,
                error=str(e),
            )
        except Exception:  # noqa: BLE001, S110
            pass
        return False
    try:
        from polyx.core import observability

        observability.event(observability.logger_for("app"), "module.launch", modulo=dotted)
    except Exception:  # noqa: BLE001, S110
        pass
    return True


# ──────────────────────────────────────────────────────────────────────
# Hero panel: a microscope field with a detection in progress
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Top KPI chips (`3 / POLÍMEROS / PET·PP·LDPE`)
# ──────────────────────────────────────────────────────────────────────
class StatChip(QFrame):
    def __init__(self, big: str, label: str, sub: str, color: str):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        self.setMinimumHeight(96)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)

        l_big = QLabel(big)
        l_big.setStyleSheet(f"color: {color}; font-size: 26pt; font-weight: 700; border: none;")
        l_label = QLabel(label.upper())
        l_label.setStyleSheet(
            f"color: {T.INK2}; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.5px; border: none;"
        )
        l_sub = QLabel(sub)
        l_sub.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        l_sub.setWordWrap(True)

        lay.addWidget(l_big)
        lay.addWidget(l_label)
        lay.addWidget(l_sub)


# ──────────────────────────────────────────────────────────────────────
# Module card
# ──────────────────────────────────────────────────────────────────────
class ModuleCard(QFrame):
    """Card of a module: icon, title, description and open action.

    The whole card is the click area, not just the button. That is why
    it needs to say it is pressable — accent border and elevation on
    hover — and be reachable with the keyboard: without ``StrongFocus``
    the module grid was out of the Tab walk and existed only for the
    mouse.
    """

    def __init__(
        self,
        number: int,
        icon_name: str,
        title: str,
        description: str,
        accent: str,
        on_open,
        shortcut: str = "",
    ):
        super().__init__()
        self._on_open = on_open
        self._accent = accent
        self.setStyleSheet(self._qss(active_2=False))
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(title)
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(8)

        top = QHBoxLayout()
        tag = QLabel(f"{tr('MÓDULO')} 0{number}")
        tag.setStyleSheet(
            f"color: white; background: {accent}; padding: 3px 8px; "
            f"border-radius: 4px; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.2px; border: none;"
        )
        tag.setFixedHeight(20)
        tag.setAlignment(Qt.AlignCenter)
        top.addWidget(tag, 0, Qt.AlignLeft | Qt.AlignVCenter)
        top.addStretch(1)
        self._btn_open = QPushButton(tr("Abrir  →"))
        self._btn_open.setCursor(Qt.PointingHandCursor)
        self._btn_open.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {accent}; "
            f"border: none; font-weight: 600; padding: 4px 0; }}"
            f"QPushButton:hover {{ color: {T.ACCENT_D_TX}; }}"
        )
        self._btn_open.clicked.connect(self._open)
        top.addWidget(self._btn_open)
        lay.addLayout(top)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(iconos.pixmap(icon_name, 30, accent, 1.85))
        icon_lbl.setStyleSheet("border: none;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {T.INK}; font-size: 17pt; font-weight: 600; border: none;")
        title_row.addWidget(icon_lbl)
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)
        lay.addLayout(title_row)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        lay.addWidget(desc)
        lay.addStretch(1)

        self.setCursor(Qt.PointingHandCursor)
        if shortcut:
            self.setToolTip(f"{title}  ·  {shortcut}")

    # ── Visual state ───────────────────────────────────────────
    def _qss(self, active_2: bool) -> str:
        edge = self._accent if active_2 else T.RULE
        background = T.on_background(self._accent, 0.05) if active_2 else T.BG
        return (
            f"ModuleCard {{ background: {background}; border: 1px solid {edge};"
            f" border-radius: 10px; }}"
        )

    def enterEvent(self, ev):
        # Without this, a whole clickable card gave no signal of being clickable.
        self.setStyleSheet(self._qss(active_2=True))
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.setStyleSheet(self._qss(active_2=False))
        super().leaveEvent(ev)

    def focusInEvent(self, ev):
        self.setStyleSheet(self._qss(active_2=True))
        super().focusInEvent(ev)

    def focusOutEvent(self, ev):
        self.setStyleSheet(self._qss(active_2=False))
        super().focusOutEvent(ev)

    # ── Action ──────────────────────────────────────────────────
    def _open(self):
        """Opens the module and announces that it is opening.

        A module takes two or three seconds to appear, because its
        process imports torch before drawing anything. Without the
        signal, that silence reads as a click that did not land and
        people click again, so two windows end up open.
        """
        self._btn_open.setEnabled(False)
        self._btn_open.setText(tr("Abriendo…"))
        QTimer.singleShot(3200, self._restore_button)
        self._on_open()

    def _restore_button(self):
        self._btn_open.setEnabled(True)
        self._btn_open.setText(tr("Abrir  →"))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._open()
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._open()
            return
        super().keyPressEvent(ev)


# ──────────────────────────────────────────────────────────────────────
# Main Launcher window
# ──────────────────────────────────────────────────────────────────────
class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        marca.titular(self, "launcher")
        self.resize(1180, 820)

        # It is stored separately because the window is rebuilt when the theme changes
        # and the version notice cannot be lost in that rebuild.
        self._sha_version_new: str = ""

        self._search = SearchUpdates(self)
        self._search.there_is_a_new_version.connect(self._notify_version_new)
        # It is triggered with the window already drawn: startup takes priority.
        QTimer.singleShot(1200, self._search.lookfor)

        self._build_ui()

    def _build_ui(self):
        """Builds the whole window with the active palette.

        Rebuilding it is the simple way to apply a theme hot: the colors
        live embedded in each widget's stylesheet, so there is no way to
        repaint them without recreating them. In a window with four
        cards it costs milliseconds and no flicker is seen.
        """
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Topbar ──
        topbar = QFrame()
        topbar.setStyleSheet(f"background: {T.BG}; border-bottom: 1px solid {T.RULE};")
        topbar.setFixedHeight(58)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(24, 0, 24, 0)
        tb.setSpacing(12)
        tb.addWidget(LogoBadge("POLY-X", "Microplastics analytics suite"))
        tb.addStretch(1)

        # Preferences. The language used to live here as a loose dropdown; now
        # it shares a dialog with the theme and motion, which are the same
        # decision —how the program looks and behaves— and not three
        # controls scattered across the bar.
        self.btn_prefs = QPushButton()
        self.btn_prefs.setIcon(iconos.icon("ajustes", 18, T.INK2, 2.0))
        self.btn_prefs.setIconSize(QSize(18, 18))
        self.btn_prefs.setText(tr("Preferencias"))
        self.btn_prefs.setCursor(Qt.PointingHandCursor)
        self.btn_prefs.setToolTip(tr("Tema, idioma y movimiento"))
        self.btn_prefs.clicked.connect(self._open_preferences)
        tb.addWidget(self.btn_prefs)

        version_lbl = QLabel(f"v{__version__}")
        version_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        tb.addWidget(version_lbl)

        # New version notice. It starts hidden and only appears if the background
        # check confirms that GitHub is ahead of what is installed.
        self.btn_update = QPushButton(tr("Actualizar"))
        self.btn_update.setCursor(Qt.PointingHandCursor)
        self.btn_update.setStyleSheet(
            f"QPushButton {{ background: {T.WARN}; color: white; border: none; "
            f"border-radius: 4px; padding: 4px 10px; font-size: 9pt; "
            f"font-weight: 600; }}"
            f"QPushButton:hover {{ background: {T.ERR}; }}"
        )
        self.btn_update.setVisible(False)
        self.btn_update.clicked.connect(self._launch_updater)
        tb.addWidget(self.btn_update)
        # If the check has already answered before this rebuild, the
        # notice is put back instead of being lost.
        if self._sha_version_new:
            self._notify_version_new(self._sha_version_new)
        root.addWidget(topbar)

        # ── Central scroll ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        root.addWidget(scroll, 1)

        page = QWidget()
        scroll.setWidget(page)
        body = QVBoxLayout(page)
        body.setContentsMargins(48, 32, 48, 40)
        body.setSpacing(28)

        # ── Hero ──
        hero = QHBoxLayout()
        hero.setSpacing(36)

        left = QVBoxLayout()
        left.setSpacing(10)
        kicker = QLabel(tr("● PLATAFORMA DE ANÁLISIS"))
        kicker.setStyleSheet(
            f"color: {T.ACCENT_D_TX}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        left.addWidget(kicker)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        t1 = QLabel(tr("Poly-X"))
        t1.setStyleSheet(f"color: {T.INK}; font-size: 38pt; font-weight: 700; border: none;")
        t2 = QLabel(tr("analytics"))
        t2.setStyleSheet(f"color: {T.ACCENT_TX}; font-size: 38pt; font-weight: 300; border: none;")
        title_row.addWidget(t1)
        title_row.addWidget(t2)
        title_row.addStretch(1)
        left.addLayout(title_row)

        subtitle = QLabel(tr("Plataforma de detección y clasificación de microplásticos"))
        subtitle.setStyleSheet(f"color: {T.INK2}; font-size: 13pt; font-weight: 500; border: none;")
        left.addWidget(subtitle)

        descr = QLabel(
            tr(
                "Detección automatizada de PET, PP y LDPE por fluorescencia Nile Red (254 nm) "
                "e inteligencia artificial. Entrenamiento, etiquetado, detección y reporte en un mismo flujo."
            )
        )
        descr.setWordWrap(True)
        descr.setStyleSheet(f"color: {T.INK3}; font-size: 10.5pt; border: none;")
        left.addWidget(descr)

        # Authorship credit (just below the descriptor)
        author = QLabel(
            tr("Diseñado y desarrollado por <b>Cristofher Ferrada</b> · Dr (c) en Química, 2026")
        )
        author.setTextFormat(Qt.RichText)
        author.setStyleSheet(
            f"color: {T.ACCENT_D_TX}; font-size: 9.5pt; border: none; "
            f"background: {T.BG_SOFT}; padding: 6px 10px; border-radius: 4px;"
        )
        left.addWidget(author)
        left.addSpacing(4)

        # 4 chips
        chips = QHBoxLayout()
        chips.setSpacing(12)
        chips.addWidget(StatChip("3", tr("Polímeros"), "PET · PP · LDPE", T.ACCENT))
        chips.addWidget(StatChip("254", "nm", tr("fluorescencia Nile Red"), T.OK))
        chips.addWidget(StatChip("YOLO", "v8 / v11", "deep learning", T.VIO))
        chips.addWidget(StatChip("μm", tr("medición"), tr("calibración por píxel"), T.WARN))
        left.addLayout(chips)

        hero.addLayout(left, 3)
        hero.addWidget(MicroscopePanel(), 0, Qt.AlignTop)
        body.addLayout(hero)

        # ── `MÓDULOS` section ──
        mod_kicker = QLabel(tr("MÓDULOS"))
        mod_kicker.setStyleSheet(
            f"color: {T.INK2}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        body.addWidget(mod_kicker)

        grid = QGridLayout()
        grid.setSpacing(16)
        grid.addWidget(
            ModuleCard(
                1,
                "detector",
                tr("Detector"),
                tr(
                    "Analiza imágenes con un modelo .pt entrenado. Genera salidas anotadas, "
                    "CSV con centroides y diámetros, métricas globales e informe HTML de detección."
                ),
                T.ACCENT,
                self.open_detector,
                "Alt+1",
            ),
            0,
            0,
        )
        grid.addWidget(
            ModuleCard(
                2,
                "entrenador",
                tr("Entrenador"),
                tr(
                    "Entrena modelos YOLO v8 / v11. Curvas en vivo, recomendaciones automáticas "
                    "de calidad y comparación con runs anteriores."
                ),
                T.OK,
                self.open_trainer,
                "Alt+2",
            ),
            0,
            1,
        )
        grid.addWidget(
            ModuleCard(
                3,
                "etiquetador",
                tr("Etiquetador"),
                tr(
                    "Anota imágenes en formato YOLO. Soporta pre-anotación con un modelo "
                    "existente y atajos de teclado, ahorra ~80 % del tiempo manual."
                ),
                T.WARN,
                self.open_labeler,
                "Alt+3",
            ),
            1,
            0,
        )
        grid.addWidget(
            ModuleCard(
                4,
                "visor",
                tr("Visor"),
                tr(
                    "Inspección de una imagen a la vez con calibración interactiva μm/píxel "
                    "(línea o círculo) y medición precisa por partícula."
                ),
                T.VIO,
                self.open_viewer,
                "Alt+4",
            ),
            1,
            1,
        )
        body.addLayout(grid)

        # ── Footer ──
        body.addStretch(1)
        body.addWidget(HLine())
        foot = QHBoxLayout()
        st = QLabel(tr("● Listo"))
        st.setStyleSheet(f"color: {T.OK_TX}; font-size: 9.5pt; font-weight: 600;")
        foot.addWidget(st)
        foot.addSpacing(20)
        copyr = QLabel("© Cristofher Ferrada · Poly-X v" + __version__)
        copyr.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        foot.addWidget(copyr)
        foot.addStretch(1)
        readme_btn = QPushButton(tr("LÉAME"))
        readme_btn.setIcon(iconos.icon("texto", 15, T.INK2, 2.0))
        readme_btn.setCursor(Qt.PointingHandCursor)
        readme_btn.clicked.connect(self.open_readme)
        foot.addWidget(readme_btn)
        manual_btn = QPushButton(tr("Manual de usuario"))
        manual_btn.setIcon(iconos.icon("manual", 15, T.INK2, 2.0))
        manual_btn.setCursor(Qt.PointingHandCursor)
        manual_btn.clicked.connect(self.open_manual)
        foot.addWidget(manual_btn)
        body.addLayout(foot)

        # Keyboard shortcuts for the four modules. People who use the program
        # daily always open the same one, and reaching it with Alt+1 saves the trip to the
        # mouse every time.
        for tecla, action in (
            ("Alt+1", self.open_detector),
            ("Alt+2", self.open_trainer),
            ("Alt+3", self.open_labeler),
            ("Alt+4", self.open_viewer),
            ("Ctrl+,", self._open_preferences),
        ):
            QShortcut(QKeySequence(tecla), self, activated=action)

    # ── Preferences ────────────────────────────────────────────
    def _open_preferences(self):
        language_before = language()
        dlg = PreferencesDialog(self)
        dlg.theme_changed.connect(lambda _: self._build_ui())
        dlg.exec()
        if language() != language_before:
            # The language cannot be applied hot: each widget fixed its
            # text when it was built. The user is warned once, on close, and not on every
            # dialog click.
            QMessageBox.information(
                self,
                tr("Idioma"),
                tr(
                    "El idioma se aplicará al reabrir Poly-X. Los módulos que "
                    "abras desde ahora ya salen en el idioma nuevo."
                ),
            )

    # ── Updates ────────────────────────────────────────────────
    def _notify_version_new(self, sha_short: str):
        """Shows the update button when GitHub is ahead."""
        self._sha_version_new = sha_short
        self.btn_update.setText(tr("Actualizar") + f" · {sha_short}")
        self.btn_update.setToolTip(
            tr("Hay una versión nueva disponible en GitHub. Pulsa para descargarla e instalarla.")
        )
        self.btn_update.setVisible(True)

    def _launch_updater(self):
        if not can_update():
            QMessageBox.information(
                self,
                tr("Actualizar"),
                tr(
                    "No se encontró {} en la carpeta de instalación. Descarga "
                    "la versión nueva manualmente desde GitHub."
                ).format(UPDATER.name),
            )
            return
        if (
            QMessageBox.question(
                self,
                tr("Actualizar"),
                tr(
                    "Se descargará la versión nueva y se cerrará Poly-X.\n\n"
                    "Tus modelos, resultados y datos no se tocan. ¿Continuar?"
                ),
            )
            != QMessageBox.Yes
        ):
            return
        # The updater overwrites program files; Poly-X has to be released
        # first so it does not clash with the .py files that are in use.
        if not launch_updater(UPDATER):
            QMessageBox.warning(
                self,
                tr("Actualizar"),
                tr(
                    "No se pudo lanzar {}. Ejecútalo a mano desde la carpeta de instalación."
                ).format(UPDATER.name),
            )
            return
        self.close()

    # ── Launchers ──────────────────────────────────────────────
    def open_detector(self):
        self._launch_module("polyx.gui.detector")

    def open_trainer(self):
        self._launch_module("polyx.gui.trainer")

    def open_labeler(self):
        self._launch_module("polyx.gui.labeler")

    def open_viewer(self):
        self._launch_module("polyx.gui.viewer")

    def _launch_module(self, dotted: str, fallback: str = ""):
        """Spawns the module in a separate process so the launcher does not block.

        Delegates to _spawn_module, which does not raise: False means
        the module does not exist or that its process failed to start,
        and here the user has to be warned.
        """
        if not _spawn_module(dotted):
            self._toast(fallback or f"No se pudo abrir el módulo {dotted}")

    def open_manual(self):
        manual = Path(__file__).resolve().parents[3] / "docs" / "manual" / "Manual_PolyX.html"
        if manual.exists():
            import webbrowser

            webbrowser.open(manual.as_uri())
        else:
            self._toast("Manual_PolyX.html no encontrado")

    def open_readme(self):
        readme = Path(__file__).resolve().parents[3] / "LEEME.txt"
        if readme.exists():
            open_in_system(str(readme))
        else:
            self._toast("LEEME.txt no encontrado")

    def _toast(self, msg: str):
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(self, tr("Poly-X"), msg)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Before the first `show()`: the taskbar button is created
    # with whatever identity exists at that moment.
    marca.identify(app, "launcher")
    # Startup JSONL trace (`app.jsonl`); it never prevents the window from opening.
    _register_boot()
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
