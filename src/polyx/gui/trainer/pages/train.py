"""Page 5 — `Entrenar`. Control + tabs (`Métricas en vivo` · `Curvas` · `Log`)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from polyx.core.i18n import tr
from polyx.core.platform import open_in_system
from polyx.gui import iconos
from polyx.gui import theme as T
from polyx.gui.widgets import KPICard

from ..runner import TrainerRunner
from ..state import validate_training_config
from ._base import TrainerPage


class _HwProbeWorker(QThread):
    """Probes the hardware in the background. Returns a dict with torch info."""

    done = Signal(dict)

    def run(self):
        d = {"ok": False, "msg": "", "color": T.INK3}
        try:
            import torch

            d["torch_version"] = torch.__version__
            d["cuda_build"] = torch.version.cuda
            d["cuda_available"] = bool(torch.cuda.is_available())
            if d["cuda_available"]:
                d["device_name"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info(0)
                d["vram_free_gb"] = free / 1024**3
                d["vram_total_gb"] = total / 1024**3
                d["ok"] = True
        except Exception as e:  # noqa: BLE001
            d["error"] = str(e)
        self.done.emit(d)


class _LiveCurves(QFrame):
    """matplotlib 2x2 canvas with Box loss / mAP / Precision / Recall curves.

    matplotlib is imported lazily the first time data arrives.
    This avoids a ~5 s wait when opening the Trainer.
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(8, 8, 8, 8)
        self._placeholder = QLabel(
            tr("Las curvas aparecerán aquí en cuanto el entrenamiento avance la primera época.")
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {T.INK3}; font-size: 10pt; border: none; padding: 60px;"
        )
        self._lay.addWidget(self._placeholder)
        self._ok = False
        self.fig = None
        self.axes = None
        self.canvas = None

    def _lazy_init(self) -> bool:
        if self._ok or self.canvas is not None:
            return self._ok
        try:
            import matplotlib

            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.fig = Figure(figsize=(8, 5), tight_layout=True)
            self.axes = self.fig.subplots(2, 2)
            for ax in self.axes.flat:
                ax.grid(alpha=0.25)
            self.canvas = FigureCanvasQTAgg(self.fig)
            # Replace the placeholder with the canvas
            self._lay.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._lay.addWidget(self.canvas)
            self._ok = True
        except Exception as e:  # noqa: BLE001
            self._placeholder.setText(f"matplotlib no disponible: {e}")
        return self._ok

    def update_curves(self, history):
        if not history:
            return
        if not self._lazy_init():
            return
        epochs = [m.epoch for m in history]
        box = [m.box_loss for m in history]
        map50 = [m.map50 for m in history]
        prec = [m.precision for m in history]
        rec = [m.recall for m in history]
        ax = self.axes
        for a in ax.flat:
            a.clear()
            a.grid(alpha=0.25)
        ax[0, 0].plot(epochs, box, color=T.WARN, lw=1.6)
        ax[0, 0].set_title("Box loss", fontsize=10)
        ax[0, 1].plot(epochs, map50, color=T.ACCENT, lw=1.6)
        ax[0, 1].set_title("mAP@50", fontsize=10)
        ax[1, 0].plot(epochs, prec, color=T.VIO, lw=1.6)
        ax[1, 0].set_title("Precision", fontsize=10)
        ax[1, 1].plot(epochs, rec, color=T.OK, lw=1.6)
        ax[1, 1].set_title("Recall", fontsize=10)
        for a in ax.flat:
            a.set_xlabel("Época", fontsize=8)
            a.tick_params(labelsize=8)
        self.canvas.draw()


class TrainPage(TrainerPage):
    PAGE_ICON = "ejecutar"
    PAGE_TITLE = tr("Entrenar")
    PAGE_DESCRIPTION = tr(
        "Inicia el entrenamiento del modelo con los parámetros configurados. Verás "
        "curvas en vivo, métricas y log. Puedes detenerlo en cualquier momento."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.runner: TrainerRunner | None = None

        # ── Hardware indicator (always shown) ──
        hw_frame = QFrame()
        hw_frame.setStyleSheet(
            f"QFrame {{ background: {T.BG_SOFT}; border: 1px solid {T.RULE}; border-radius: 6px; }}"
        )
        hwl = QHBoxLayout(hw_frame)
        hwl.setContentsMargins(14, 8, 14, 8)
        hwl.setSpacing(12)
        self.lbl_hw = QLabel(tr("⌛  Detectando hardware…"))
        self.lbl_hw.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
        self.lbl_hw.setTextFormat(Qt.RichText)
        hwl.addWidget(self.lbl_hw, 1)
        btn_recheck = QPushButton()
        btn_recheck.setIcon(iconos.icon("recargar", 15, T.INK2))
        btn_recheck.setFixedWidth(36)
        btn_recheck.setToolTip(tr("Re-detectar"))
        btn_recheck.clicked.connect(self._refresh_hw)
        hwl.addWidget(btn_recheck)
        self.body.addWidget(hw_frame)
        # deferred detection so startup is not blocked
        from PySide6.QtCore import QTimer

        QTimer.singleShot(150, self._refresh_hw)

        # ── Control ──
        c1, l1 = self.card(tr("Control"), "ajustes")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start = QPushButton(tr("Iniciar entrenamiento"))
        self.btn_start.setIcon(iconos.icon("ejecutar", 15, T.ON_ACCENT))
        self.btn_start.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 9px 18px; font-weight: 600;"
        )
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._start)
        row.addWidget(self.btn_start)

        self.btn_stop = QPushButton(tr("Detener"))
        self.btn_stop.setIcon(iconos.icon("detener", 15, T.ON_ACCENT))
        self.btn_stop.setStyleSheet(
            f"background: {T.ERR}; color: white; border: none; "
            f"border-radius: 6px; padding: 9px 18px; font-weight: 600;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_stop)

        self.btn_open = QPushButton(tr("Abrir carpeta de resultados"))
        self.btn_open.setIcon(iconos.icon("dataset", 15, T.INK2))
        self.btn_open.clicked.connect(self._open_results)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        l1.addLayout(row)

        # Epoch progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat(tr("%v / %m épocas"))
        l1.addWidget(self.progress)
        self.body.addWidget(c1)

        # ── Tabs `Métricas`/`Curvas`/`Log` ──
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 8px 18px; margin-right: 2px;
                background: {T.BG_SOFT}; color: {T.INK2};
                border: 1px solid {T.RULE}; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-weight: 500;
            }}
            QTabBar::tab:selected {{ background: {T.BG}; color: {T.ACCENT_D_TX}; font-weight: 600; }}
            QTabWidget::pane {{ border: 1px solid {T.RULE}; border-radius: 6px; background: {T.BG}; }}
        """)

        # Tab 1: Live metrics (8 KPI cards)
        metrics = QFrame()
        metrics.setStyleSheet(f"background: {T.BG};")
        ml = QGridLayout(metrics)
        ml.setContentsMargins(16, 16, 16, 16)
        ml.setSpacing(12)
        self.kpi_epoch = KPICard(tr("Época actual"), T.ACCENT)
        self.kpi_map50 = KPICard("mAP@50", T.OK)
        self.kpi_map95 = KPICard("mAP@50-95", T.VIO)
        self.kpi_box = KPICard("Box loss", T.WARN)
        self.kpi_prec = KPICard("Precision", T.ACCENT_D)
        self.kpi_rec = KPICard("Recall", T.ACCENT)
        self.kpi_best = KPICard("Mejor mAP@50", T.OK)
        self.kpi_nomp = KPICard(tr("Sin mejora"), T.WARN)
        for w, pos in [
            (self.kpi_epoch, (0, 0)),
            (self.kpi_map50, (0, 1)),
            (self.kpi_map95, (0, 2)),
            (self.kpi_box, (0, 3)),
            (self.kpi_prec, (1, 0)),
            (self.kpi_rec, (1, 1)),
            (self.kpi_best, (1, 2)),
            (self.kpi_nomp, (1, 3)),
        ]:
            ml.addWidget(w, *pos)
        # Card `qué mirar`
        watch = QFrame()
        watch.setStyleSheet(
            f"QFrame {{ background: {T.BG_SOFT}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        wl = QVBoxLayout(watch)
        wl.setContentsMargins(16, 12, 16, 14)
        title = QLabel(tr("¿Qué mirar?"))
        title.setStyleSheet(f"color: {T.INK}; font-weight: 600; font-size: 11pt; border: none;")
        wl.addWidget(title)
        tips = QLabel(
            tr(
                "• <b>mAP@50</b> debe SUBIR y estabilizarse cerca del 80–95 % para un buen modelo.<br>"
                "• <b>Box loss</b> debe BAJAR de forma sostenida. Si oscila o sube, baja el lr0.<br>"
                "• Si <b>Sin mejora</b> llega a <b>Patience</b>, el entrenamiento se detendrá solo."
            )
        )
        tips.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none; line-height: 1.6;")
        tips.setTextFormat(Qt.RichText)
        tips.setWordWrap(True)
        wl.addWidget(tips)
        ml.addWidget(watch, 2, 0, 1, 4)
        self.tabs.addTab(metrics, tr("Métricas en vivo"))

        # Tab 2: `Curvas`
        self.curves = _LiveCurves()
        self.tabs.addTab(self.curves, tr("Curvas"))

        # Tab 3: Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            f"background: #0d1117; color: #c9d1d9; font-family: Consolas, monospace; "
            f"font-size: 9.5pt; border: 1px solid {T.RULE}; border-radius: 6px;"
        )
        self.log.setMinimumHeight(420)
        self.tabs.addTab(self.log, "Log")

        self.body.addWidget(self.tabs)

        # Subscribe
        self.state.train_epoch.connect(self._on_epoch)
        self.state.train_log.connect(self._on_log)
        self.state.train_finished.connect(self._on_finished_ok)
        self.state.train_aborted.connect(self._on_aborted)
        self.state.train_failed.connect(self._on_failed)
        self._hw_worker: _HwProbeWorker | None = None

    # ── Hardware detection in the background ─────────────────
    def _refresh_hw(self):
        if self._hw_worker is not None and self._hw_worker.isRunning():
            return
        self.lbl_hw.setText(tr("⌛  Detectando hardware…"))
        self._hw_worker = _HwProbeWorker()
        self._hw_worker.done.connect(self._on_hw_probe)
        self._hw_worker.start()

    def _on_hw_probe(self, d: dict):
        if d.get("ok"):
            self.lbl_hw.setText(
                f"<b style='color:{T.OK_TX};'>● ENTRENARÁ EN GPU</b> &nbsp; · &nbsp; "
                f"<b>{d.get('device_name', '?')}</b> &nbsp; · &nbsp; "
                f"VRAM libre {d.get('vram_free_gb', 0):.1f} / {d.get('vram_total_gb', 0):.1f} GB "
                f"&nbsp; · &nbsp; torch {d.get('torch_version', '?')} (CUDA {d.get('cuda_build', '?')})"
            )
        elif "error" in d:
            self.lbl_hw.setText(
                f"<b style='color:{T.ERR_TX};'>● ERROR</b> al sondear hardware: {d['error']}"
            )
        else:
            tv = d.get("torch_version", "?")
            cb = d.get("cuda_build")
            if cb is None:
                msg = (
                    f"PyTorch {tv} fue instalado <b>SIN CUDA</b> — entrenará en CPU "
                    f"(muy lento). Reinstala con SETUP.bat detectando GPU."
                )
            else:
                msg = (
                    f"PyTorch {tv} con CUDA {cb} pero <code>torch.cuda.is_available()</code> "
                    f"= False — revisa drivers NVIDIA / reinicia el equipo."
                )
            self.lbl_hw.setText(
                f"<b style='color:{T.WARN_TX};'>● ENTRENARÁ EN CPU</b> &nbsp; · &nbsp; {msg}"
            )

    # ──────────────────────────────────────────
    def _start(self):
        if self.state.is_running():
            return
        # ── Pre-flight gate (M003/T03) ─────────────────────────────
        # Validation runs BEFORE creating the runner: a failure midway through the
        # run loses the work and the GPU, and the user does not understand what happened.
        # It covers the previous `data.yaml` loaded check with a dialog
        # that also lists ALL the problems at once.
        ok, problems = validate_training_config(self.state)
        if not ok:
            detail = "\n".join(f"• {m}" for m in problems)
            QMessageBox.warning(
                self,
                tr("No se puede entrenar"),
                tr("Corrige estos problemas antes de iniciar el entrenamiento:\n\n") + detail,
            )
            return
        # Reset
        self.state.history.clear()
        self.state.best_map50 = 0.0
        self.state.best_epoch = 0
        self.state.epochs_no_improve = 0
        self.log.clear()
        for k in (
            self.kpi_epoch,
            self.kpi_map50,
            self.kpi_map95,
            self.kpi_box,
            self.kpi_prec,
            self.kpi_rec,
            self.kpi_best,
            self.kpi_nomp,
        ):
            k.set_value(None)
        self.progress.setRange(0, self.state.params.epochs)
        self.progress.setValue(0)

        self.state.set_running(True)
        self.state.train_started.emit()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.runner = TrainerRunner(self.state)
        self.runner.epoch_metrics.connect(self.state.train_epoch.emit)
        self.runner.log_line.connect(self.state.train_log.emit)
        self.runner.progress.connect(lambda ep, tot: self.progress.setValue(ep))
        self.runner.finished_ok.connect(self.state.train_finished.emit)
        self.runner.aborted.connect(self.state.train_aborted.emit)
        self.runner.failed.connect(self.state.train_failed.emit)
        self.runner.start()

    def _stop(self):
        if not self.state.is_running():
            return
        self.state.request_abort()
        self.state.train_log.emit("[INFO] Solicitando detener entrenamiento…")
        # Ultralytics lacks a clean interruption; the flag is checked in the callback

    def _open_results(self):
        d = self.state.run_dir
        if d and d.exists():
            open_in_system(str(d))
        else:
            # runs_train folder
            root = Path(__file__).resolve().parents[5] / "runs_train"
            if root.exists():
                open_in_system(str(root))
            else:
                QMessageBox.information(
                    self, tr("Sin resultados"), tr("Aún no se han generado runs.")
                )

    # ── Slots ────────────────────────────────────────────────
    def _on_epoch(self, em):
        st = self.state
        self.kpi_epoch.set_value(f"{em.epoch} / {st.params.epochs}")
        self.kpi_map50.set_value(f"{em.map50:.3f}")
        self.kpi_map95.set_value(f"{em.map50_95:.3f}")
        self.kpi_box.set_value(f"{em.box_loss:.3f}")
        self.kpi_prec.set_value(f"{em.precision:.3f}")
        self.kpi_rec.set_value(f"{em.recall:.3f}")
        self.kpi_best.set_value(f"{st.best_map50:.3f}  (ep {st.best_epoch})")
        self.kpi_nomp.set_value(f"{st.epochs_no_improve} / {st.params.patience}")
        self.curves.update_curves(st.history)

    def _on_log(self, line: str):
        self.log.appendPlainText(line)

    def _on_finished_ok(self, best: str):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.state.train_log.emit(f"[OK] Best: {best}")

    def _on_aborted(self):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_failed(self, msg: str):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, tr("Falló el entrenamiento"), msg)
