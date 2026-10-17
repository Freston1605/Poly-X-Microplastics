"""Página 5 — Ejecutar. Inicia la inferencia en background y muestra previews."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QProgressBar, QFrame,
    QGridLayout, QFileDialog, QMessageBox,
)

from ._base import DetectorPage
from polyx.gui import theme as T
from polyx.gui import iconos
from ..runner import DetectorRunner
from polyx.model.i18n import tr
from polyx.model.plataforma import abrir_en_el_sistema


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
        self.preview = QLabel(tr("(esperando imagen)"))
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
        if pm.isNull(): return
        # Escalar al ancho disponible
        scaled = pm.scaled(self.preview.width() or 320, self.preview.height() or 200,
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)
        if name:
            self.lbl_title.setText(f"{self.lbl_title.text().split(' — ')[0]} — {name}")


class EjecutarPage(DetectorPage):
    STEP_N = 5
    STEP_TITLE = tr("Ejecutar")
    STEP_DESCRIPTION = (
        tr("Inicia la inferencia. Verás progreso en vivo, previews de imágenes anotadas "
        "y podrás cancelar en cualquier momento.")
    )

    def refresh(self):
        """Al entrar a la pagina, recalcular que hara el troceado con el lote."""
        self._refrescar_aviso_troceo()

    def _refrescar_aviso_troceo(self):
        """Resume si las fotos se van a trocear, y cuanto mas va a tardar."""
        from polyx.model.yolo_wrap import tamano_imagen, politica_troceado
        p = self.state.params
        if not self.state.images:
            self.lbl_troceo.setText("")
            return

        wh = tamano_imagen(self.state.images[0])
        if wh is None:
            self.lbl_troceo.setText("")
            return

        if p.troceo == "nunca":
            self.lbl_troceo.setText(tr(
                "Troceado desactivado: cada foto entra completa a imgsz {}. "
                "En fotos grandes las partículas pequeñas pueden desaparecer al "
                "reescalar.").format(p.imgsz))
            return

        plan = politica_troceado(
            wh[0], wh[1], p.imgsz,
            umbral_px=0 if p.troceo == "siempre" else p.troceo_umbral_px,
            tile=p.troceo_tile, overlap=p.troceo_overlap)

        if plan is None:
            self.lbl_troceo.setText(tr(
                "Las fotos ({}×{} px) no llegan al umbral de "
                "{} px: se analizan de una pieza.").format(
                    wh[0], wh[1], p.troceo_umbral_px))
        else:
            self.lbl_troceo.setText(tr(
                "⚠ Fotos grandes ({}×{} px): se analizarán en "
                "{} recortes de {} px con "
                "{}% de solape. Va a tardar unas "
                "{}× más.<br>"
                "El recorte es solo para que el modelo pueda detectar: los "
                "resultados y el informe se entregan sobre la foto completa.").format(
                    wh[0], wh[1], plan['n_tiles'], plan['tile'],
                    int(plan['overlap'] * 100), plan['n_tiles']))

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.runner: DetectorRunner | None = None

        # ── Control ──
        c1, l1 = self.card(tr("Control"), "ejecutar")

        # Aviso de troceado. Vivia solo en la pestana Parametros, donde nadie lo
        # veia; aqui aparece justo antes de pulsar Iniciar, que es cuando importa
        # saber si el lote va a tardar N veces mas de lo esperado.
        self.lbl_troceo = QLabel()
        self.lbl_troceo.setWordWrap(True)
        self.lbl_troceo.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; border: none;")
        l1.addWidget(self.lbl_troceo)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start = QPushButton(tr("Iniciar detección"))
        self.btn_start.setIcon(iconos.icono("ejecutar", 15, T.ON_ACCENT))
        self.btn_start.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._start)
        row.addWidget(self.btn_start)

        self.btn_stop = QPushButton(tr("Detener"))
        self.btn_stop.setIcon(iconos.icono("detener", 15, T.ON_ACCENT))
        self.btn_stop.setStyleSheet(
            f"background: {T.ERR}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_stop)

        self.btn_review = QPushButton(tr("Revisar en pantalla grande"))
        self.btn_review.setIcon(iconos.icono("ver", 15, T.ON_ACCENT))
        self.btn_review.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_review.setCursor(Qt.PointingHandCursor)
        self.btn_review.setEnabled(False)
        self.btn_review.clicked.connect(self._open_review)
        row.addWidget(self.btn_review)

        self.btn_open = QPushButton(tr("Abrir carpeta de resultados"))
        self.btn_open.setIcon(iconos.icono("dataset", 15, T.INK2))
        self.btn_open.clicked.connect(self._open_results)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        l1.addLayout(row)
        self.body.addWidget(c1)

        # ── Progreso ──
        c2, l2 = self.card(tr("Progreso"), "espera")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%  —  %v / %m")
        l2.addWidget(self.progress)
        self.lbl_progress_info = QLabel(tr("Sin ejecución todavía."))
        self.lbl_progress_info.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l2.addWidget(self.lbl_progress_info)
        self.body.addWidget(c2)

        # ── Preview ──
        c3, l3 = self.card(tr("Preview en vivo"), "ver")
        sub = QLabel(tr("Última imagen procesada por cada modelo:"))
        sub.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l3.addWidget(sub)

        grid = QHBoxLayout()
        grid.setSpacing(12)
        self.previews = [
            _PreviewSlot(tr("Modelo {n}").format(n=1)),
            _PreviewSlot(tr("Modelo {n}").format(n=2)),
            _PreviewSlot(tr("Modelo {n}").format(n=3)),
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
            QMessageBox.warning(self, tr("Faltan modelos"), tr("Carga al menos un modelo .pt en la pestaña Modelos."))
            return
        if not self.state.images:
            QMessageBox.warning(self, tr("Sin imágenes"), tr("Selecciona imágenes en la pestaña Imágenes."))
            return
        # Actualizar título de cada preview con alias del modelo
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
        if not self.state.is_running(): return
        self.state.request_abort()
        self.lbl_progress_info.setText(tr("Deteniendo…"))

    def _open_results(self):
        d = self.state.run_dir
        if d and d.exists():
            abrir_en_el_sistema(str(d))
        else:
            QMessageBox.information(self, tr("Sin resultados"), tr("Aún no se ha ejecutado ninguna corrida."))

    def _open_review(self):
        if not any(self.state.results.values()):
            QMessageBox.information(self, tr("Sin resultados"),
                                    tr("Ejecuta una detección primero."))
            return
        from ..review_dialog import ReviewDialog
        dlg = ReviewDialog(self.state, self)
        dlg.showMaximized()
        dlg.exec()

    # ── slots del runner ──
    def _on_progress(self, done: int, total: int, name: str):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.state.run_progress.emit(done, total, name)
        self.lbl_progress_info.setText(f"Procesando {name}  ({done}/{total})")

    def _on_image_done(self, model_idx: int, res):
        active = self.state.active_models()
        # determinar a qué preview corresponde
        try:
            slot = next(s for s in active if self.state.model_slots.index(s) == model_idx)
            pi = active.index(slot)
        except Exception:
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
            tr("Listo. Pulsa «Revisar en pantalla grande» para inspeccionar y corregir."))
        self.state.run_finished.emit()

    def _on_aborted(self):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if any(self.state.results.values()):
            self.btn_review.setEnabled(True)
        self.lbl_progress_info.setText(tr("Detenido por el usuario."))
        self.state.run_aborted.emit()

    def _on_failed(self, msg: str):
        self.state.set_running(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress_info.setText(f"Error: {msg}")
        QMessageBox.critical(self, tr("Falló la ejecución"), msg)
