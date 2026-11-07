"""Page 5 — `Ejecutar`. Starts the inference in the background and shows previews."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from polyx.core.i18n import _, lazy, ngettext
from polyx.core.platform import open_in_system
from polyx.gui import iconos
from polyx.gui import theme as T

from ..runner import DetectorRunner
from ._base import DetectorPage


class _PreviewSlot(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        self.setMinimumHeight(220)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(6)
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet(
            f"color: {T.INK2}; font-weight: 600; font-size: 10pt; border: none;"
        )
        self.preview = QLabel(_("(waiting for image)"))
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            f"background: {T.BG_SOFT}; color: {T.MUTED}; border-radius: 6px; "
            f"min-height: 180px; border: none;"
        )
        self.preview.setScaledContents(False)
        lay.addWidget(self.lbl_title)
        lay.addWidget(self.preview, 1)

    def set_image_bytes(self, data: bytes | None, name: str = ""):
        if not data:
            return
        ba = QByteArray(data)
        pm = QPixmap()
        pm.loadFromData(ba, "PNG")
        if pm.isNull():
            return
        # Scale to the available width
        scaled = pm.scaled(
            self.preview.width() or 320,
            self.preview.height() or 200,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)
        if name:
            self.lbl_title.setText(f"{self.lbl_title.text().split(' — ')[0]} — {name}")


class RunPage(DetectorPage):
    STEP_N = 5
    STEP_TITLE = lazy("Run")
    STEP_DESCRIPTION = lazy(
        "Starts inference. You will see live progress, previews of annotated images, and you can cancel at any time."
    )

    def refresh(self):
        """When entering the page, recompute what the tiling will do with the batch."""
        self._refresh_warning_tiling()

    def _refresh_warning_tiling(self):
        """Summarizes whether the photos will be tiled, and how much longer it will take."""
        from polyx.core.detect.yolo import policy_tiling, size_image

        p = self.state.params
        if not self.state.images:
            self.lbl_tiling.setText("")
            return

        wh = size_image(self.state.images[0])
        if wh is None:
            self.lbl_tiling.setText("")
            return

        if p.tiling == "nunca":
            self.lbl_tiling.setText(
                _(
                    "Tiling disabled: each photo enters whole at imgsz %(imgsz)s. In large photos, small particles may vanish when rescaled."
                )
                % {"imgsz": p.imgsz}
            )
            return

        plan = policy_tiling(
            wh[0],
            wh[1],
            p.imgsz,
            threshold_px=0 if p.tiling == "siempre" else p.tiling_threshold_px,
            tile=p.tiling_tile,
            overlap=p.tiling_overlap,
        )

        if plan is None:
            self.lbl_tiling.setText(
                _(
                    "The photos (%(width)s×%(height)s px) do not reach the %(threshold)s px threshold: they are analysed in one piece."
                )
                % {"width": wh[0], "height": wh[1], "threshold": p.tiling_threshold_px}
            )
        else:
            self.lbl_tiling.setText(
                ngettext(
                    "⚠ Large photos (%(width)s×%(height)s px): they will be analysed as %(count)d tile of %(tile)s px with %(overlap)d%% overlap. It will take about %(count)d× longer.<br>Tiling is only so the model can detect: results and the report are delivered on the complete photo.",
                    "⚠ Large photos (%(width)s×%(height)s px): they will be analysed as %(count)d tiles of %(tile)s px with %(overlap)d%% overlap. It will take about %(count)d× longer.<br>Tiling is only so the model can detect: results and the report are delivered on the complete photo.",
                    plan["n_tiles"],
                )
                % {
                    "width": wh[0],
                    "height": wh[1],
                    "count": plan["n_tiles"],
                    "tile": plan["tile"],
                    "overlap": int(plan["overlap"] * 100),
                }
            )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.runner: DetectorRunner | None = None

        # ── Control ──
        c1, l1 = self.card(_("Control"), "ejecutar")

        # Tiling warning. It used to live only on the `Parametros` tab, where nobody
        # saw it; here it appears right before pressing Start, which is when it matters
        # to know whether the batch will take N times longer than expected.
        self.lbl_tiling = QLabel()
        self.lbl_tiling.setWordWrap(True)
        self.lbl_tiling.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; border: none;")
        l1.addWidget(self.lbl_tiling)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start = QPushButton(_("Start detection"))
        self.btn_start.setIcon(iconos.icon("ejecutar", 15, T.ON_ACCENT))
        self.btn_start.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._start)
        row.addWidget(self.btn_start)

        self.btn_stop = QPushButton(_("Stop"))
        self.btn_stop.setIcon(iconos.icon("detener", 15, T.ON_ACCENT))
        self.btn_stop.setStyleSheet(
            f"background: {T.ERR}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_stop)

        self.btn_review = QPushButton(_("Review full screen"))
        self.btn_review.setIcon(iconos.icon("ver", 15, T.ON_ACCENT))
        self.btn_review.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_review.setCursor(Qt.PointingHandCursor)
        self.btn_review.setEnabled(False)
        self.btn_review.clicked.connect(self._open_review)
        row.addWidget(self.btn_review)

        self.btn_open = QPushButton(_("Open results folder"))
        self.btn_open.setIcon(iconos.icon("dataset", 15, T.INK2))
        self.btn_open.clicked.connect(self._open_results)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        l1.addLayout(row)
        self.body.addWidget(c1)

        # ── Progress ──
        c2, l2 = self.card(_("Progress"), "espera")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%  —  %v / %m")
        l2.addWidget(self.progress)
        self.lbl_progress_info = QLabel(_("No run yet."))
        self.lbl_progress_info.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l2.addWidget(self.lbl_progress_info)
        self.body.addWidget(c2)

        # ── Preview ──
        c3, l3 = self.card(_("Live preview"), "ver")
        sub = QLabel(_("Last image processed by each model:"))
        sub.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l3.addWidget(sub)

        grid = QHBoxLayout()
        grid.setSpacing(12)
        self.previews = [
            _PreviewSlot(_("Model %(n)s") % {"n": 1}),
            _PreviewSlot(_("Model %(n)s") % {"n": 2}),
            _PreviewSlot(_("Model %(n)s") % {"n": 3}),
        ]
        for p in self.previews:
            grid.addWidget(p, 1)
        l3.addLayout(grid)
        self.body.addWidget(c3)

    # ──────────────────────────────────────────
    def _start(self):
        if self.state.is_running():
            return
        active = self.state.active_models()
        if not active:
            QMessageBox.warning(
                self,
                _("Missing models"),
                _("Load at least one .pt model in the Models tab."),
            )
            return
        if not self.state.images:
            QMessageBox.warning(self, _("No images"), _("Select images in the Images tab."))
            return
        # Update each preview title with the model alias
        for i, slot in enumerate(active):
            if i < len(self.previews):
                self.previews[i].lbl_title.setText(slot.alias)

        self.state.set_running(True)
        self.state.run_started.emit()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        self.runner = DetectorRunner(self.state)
        self.runner.progress.connect(self._on_progress)
        self.runner.image_done.connect(self._on_image_done)
        self.runner.finished_ok.connect(self._on_finished_ok)
        self.runner.aborted.connect(self._on_aborted)
        self.runner.failed.connect(self._on_failed)
        self.runner.start()

    def _stop(self):
        if not self.state.is_running():
            return
        self.state.request_abort()
        self.lbl_progress_info.setText(_("Stopping…"))

    def _open_results(self):
        d = self.state.run_dir
        if d and d.exists():
            open_in_system(str(d))
        else:
            QMessageBox.information(self, _("No results"), _("No run has been executed yet."))

    def _open_review(self):
        if not any(self.state.results.values()):
            QMessageBox.information(self, _("No results"), _("Run a detection first."))
            return
        from ..review_dialog import ReviewDialog

        dlg = ReviewDialog(self.state, self)
        dlg.showMaximized()
        dlg.exec()

    # ── runner slots ──
    def _on_progress(self, done: int, total: int, name: str):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.state.run_progress.emit(done, total, name)
        self.lbl_progress_info.setText(
            _("Processing %(name)s  (%(done)d/%(total)d)")
            % {"name": name, "done": done, "total": total}
        )

    def _on_image_done(self, model_idx: int, res):
        active = self.state.active_models()
        # determine which preview it corresponds to
        try:
            slot = next(s for s in active if self.state.model_slots.index(s) == model_idx)
            pi = active.index(slot)
        except Exception:  # noqa: BLE001
            return
        if pi < len(self.previews):
            self.previews[pi].set_image_bytes(res.annotated_png, name=Path(res.image_path).name)
        self.state.run_image_done.emit(model_idx, res)

    def _on_finished_ok(self):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_review.setEnabled(True)
        self.lbl_progress_info.setText(
            _("Ready. Press “Review full screen” to inspect and correct.")
        )
        self.state.run_finished.emit()

    def _on_aborted(self):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if any(self.state.results.values()):
            self.btn_review.setEnabled(True)
        self.lbl_progress_info.setText(_("Stopped by the user."))
        self.state.run_aborted.emit()

    def _on_failed(self, msg: str):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress_info.setText(_("Error: %(message)s") % {"message": msg})
        QMessageBox.critical(self, _("The run failed"), msg)
