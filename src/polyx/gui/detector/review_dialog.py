"""Full-screen review viewer.

Shows each image with the model's PREDICTIONS as editable boxes. The user
reviews at full size, decides whether the detection is good (✓ `Buena` /
✗ `Mala`), corrects the boxes (draw/move/delete/reclassify) and saves the
corrections as YOLO .txt.

Reuses ``AnnotCanvas`` (the same powerful editor as the manual GT page).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from polyx.core.detect.yolo import Detection
from polyx.core.i18n import tr
from polyx.gui import iconos
from polyx.gui import theme as T

from .pages.gt_manual import DEFAULT_CLASSES, AnnotCanvas


class ReviewDialog(QDialog):
    """Maximized dialog to review and correct the detections of a run."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle(tr("Revisión de detecciones — Poly-X"))
        self.setModal(False)
        self._current_idx = -1
        self._results: List = []     # ImageResult of the active model

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Top bar ──
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(QLabel(tr("Modelo:")))
        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(160)
        self.combo_model.currentIndexChanged.connect(self._on_model_change)
        top.addWidget(self.combo_model)

        top.addWidget(QLabel(tr("Clase (para nuevas cajas):")))
        self.combo_class = QComboBox()
        self.combo_class.addItems(DEFAULT_CLASSES)
        self.combo_class.currentIndexChanged.connect(
            lambda i: self.canvas.set_active_class(i))
        top.addWidget(self.combo_class)

        btn_fit = QPushButton(tr("Ajustar (F)"))
        btn_fit.clicked.connect(lambda: self.canvas.zoom_fit())
        top.addWidget(btn_fit)
        btn_100 = QPushButton("100 %")
        btn_100.clicked.connect(lambda: self.canvas.zoom_100())
        top.addWidget(btn_100)
        top.addStretch(1)
        self.lbl_counts = QLabel("—")
        self.lbl_counts.setStyleSheet(
            f"color: {T.INK2}; font-weight: 600; font-size: 11pt; border: none;")
        top.addWidget(self.lbl_counts)
        root.addLayout(top)

        # ── Body: list + canvas ──
        body = QHBoxLayout()
        body.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(6)
        lbl_files = QLabel(tr("Imágenes"))
        lbl_files.setStyleSheet(f"color: {T.INK2}; font-weight: 600; border: none;")
        left.addWidget(lbl_files)
        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.setMaximumWidth(300)
        self.list.currentRowChanged.connect(self._on_list_row)
        left.addWidget(self.list, 1)
        body.addLayout(left)

        self.canvas = AnnotCanvas()
        self.canvas.set_class_names(DEFAULT_CLASSES)
        self.canvas.setMinimumSize(QSize(760, 540))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self.canvas, 1)
        root.addLayout(body, 1)

        # ── Canvas status ──
        self.lbl_status = QLabel(tr("Click y arrastra para dibujar · rueda: zoom · "
                                 "Espacio+arrastre: mover · Supr: borrar caja."))
        self.lbl_status.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        self.canvas.status_changed.connect(self.lbl_status.setText)
        root.addWidget(self.lbl_status)

        # ── Button row ──
        btns = QHBoxLayout()
        btns.setSpacing(8)
        b_prev = QPushButton(tr("←  Anterior"))
        b_prev.clicked.connect(self._go_prev)
        btns.addWidget(b_prev)
        b_next = QPushButton(tr("Siguiente  →"))
        b_next.clicked.connect(self._go_next)
        btns.addWidget(b_next)

        btns.addSpacing(20)
        self.btn_good = QPushButton(tr("✓  Buena"))
        self.btn_good.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 18px; font-weight: 700;")
        self.btn_good.setCursor(Qt.PointingHandCursor)
        self.btn_good.clicked.connect(lambda: self._set_verdict("buena"))
        btns.addWidget(self.btn_good)
        self.btn_bad = QPushButton(tr("✗  Mala"))
        self.btn_bad.setStyleSheet(
            f"background: {T.ERR}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 18px; font-weight: 700;")
        self.btn_bad.setCursor(Qt.PointingHandCursor)
        self.btn_bad.clicked.connect(lambda: self._set_verdict("mala"))
        btns.addWidget(self.btn_bad)

        btns.addStretch(1)
        b_dataset = QPushButton(tr("Enviar al dataset de reentrenamiento"))
        b_dataset.setIcon(iconos.icon("exportar", 15, T.INK2))
        b_dataset.setToolTip(
            tr("Copia esta imagen y sus cajas corregidas a dataset_correcciones/ "
            "para mejorar el modelo con fine-tuning (active learning)."))
        b_dataset.setCursor(Qt.PointingHandCursor)
        b_dataset.clicked.connect(self._send_to_dataset)
        btns.addWidget(b_dataset)
        b_save = QPushButton(tr("Guardar correcciones (.txt YOLO)"))
        b_save.setIcon(iconos.icon("guardar", 15, T.ON_ACCENT))
        b_save.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;")
        b_save.setCursor(Qt.PointingHandCursor)
        b_save.clicked.connect(self._save_current)
        btns.addWidget(b_save)
        root.addLayout(btns)

        # Shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._go_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._go_next)

        self._populate_models()

    # ──────────────────────────────────────────────────────────────
    def _populate_models(self):
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self._model_indices = []
        for mi, rs in self.state.results.items():
            if not rs:
                continue
            alias = self.state.model_slots[mi].alias
            self.combo_model.addItem(alias)
            self._model_indices.append(mi)
        self.combo_model.blockSignals(False)
        if self._model_indices:
            self._on_model_change(0)

    def _on_model_change(self, combo_idx: int):
        if combo_idx < 0 or combo_idx >= len(self._model_indices):
            return
        mi = self._model_indices[combo_idx]
        self._results = list(self.state.results.get(mi, []))
        self._refresh_list()
        if self._results:
            self.list.setCurrentRow(0)

    def _refresh_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        good = bad = 0
        for res in self._results:
            name = Path(res.image_path).name
            mark = ""
            if res.verdict == "buena":
                mark = "✓ "; good += 1
            elif res.verdict == "mala":
                mark = "✗ "; bad += 1
            it = QListWidgetItem(f"{mark}{name}")
            if res.verdict == "buena":
                it.setForeground(QColor(T.OK))
            elif res.verdict == "mala":
                it.setForeground(QColor(T.ERR))
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.setWindowTitle(
            f"Revisión de detecciones — {len(self._results)} imágenes "
            f"(✓{good} ✗{bad}) — Poly-X")

    def _on_list_row(self, row: int):
        # auto-save the verdict/boxes in memory of the previous image is already done
        if row < 0 or row >= len(self._results):
            return
        self._current_idx = row
        res = self._results[row]
        path = Path(res.image_path)
        # load predictions as editable boxes
        existing: List[Detection] = list(res.predictions)
        self.canvas.load_image(path, existing)
        self._update_counts(res)

    def _update_counts(self, res):
        parts = [f"Predicciones: {len(res.predictions)}"]
        if res.has_gt:
            parts.append(f"GT: {len(res.gt)}")
            parts.append(f"VP: {res.tp}")
            parts.append(f"FP: {res.fp}")
            parts.append(f"FN: {res.fn}")
        v = {"buena": "✓ Buena", "mala": "✗ Mala"}.get(res.verdict, "sin revisar")
        parts.append(f"·  {v}")
        self.lbl_counts.setText("   ".join(parts))

    def _set_verdict(self, verdict: str):
        if not (0 <= self._current_idx < len(self._results)):
            return
        self._results[self._current_idx].verdict = verdict
        # update the list mark without losing the selection
        row = self._current_idx
        self._refresh_list()
        self.list.blockSignals(True)
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self._update_counts(self._results[row])
        # advance automatically to the next one
        self._go_next()

    def _go_prev(self):
        if self.list.count() == 0:
            return
        self.list.setCurrentRow(max(0, self.list.currentRow() - 1))
        self.canvas.setFocus()

    def _go_next(self):
        if self.list.count() == 0:
            return
        self.list.setCurrentRow(
            min(self.list.count() - 1, self.list.currentRow() + 1))
        self.canvas.setFocus()

    # ── Saving corrections ──
    @staticmethod
    def _yolo_lines(dets: List[Detection], W: int, H: int) -> List[str]:
        lines = []
        for d in dets:
            cx = (d.x1 + d.x2) / 2 / W
            cy = (d.y1 + d.y2) / 2 / H
            w = (d.x2 - d.x1) / W
            h = (d.y2 - d.y1) / H
            lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines

    def _current_image_and_dets(self):
        """(path, dets, W, H) of the current image, or None if something fails."""
        if not (0 <= self._current_idx < len(self._results)):
            return None
        res = self._results[self._current_idx]
        path = Path(res.image_path)
        dets = self.canvas.detections()
        img = QImage(str(path))
        W, H = img.width(), img.height()
        if W == 0 or H == 0:
            QMessageBox.warning(self, tr("Error"), tr("No se pudo leer la imagen."))
            return None
        return path, dets, W, H

    def _save_current(self):
        cur = self._current_image_and_dets()
        if cur is None:
            return
        path, dets, W, H = cur
        # destination: run folder, _corrected suffix
        out_dir = Path(self.state.run_dir) if self.state.run_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_corrected.txt"
        out_path.write_text("\n".join(self._yolo_lines(dets, W, H)), encoding="utf-8")
        cls_path = out_dir / "classes.txt"
        if not cls_path.exists():
            cls_path.write_text("\n".join(DEFAULT_CLASSES), encoding="utf-8")
        self.lbl_status.setText(f"✓ Guardado: {out_path}  ({len(dets)} cajas)")

    def _send_to_dataset(self):
        """Active learning: accumulates image + corrected boxes to retrain."""
        cur = self._current_image_and_dets()
        if cur is None:
            return
        path, dets, W, H = cur
        if not dets:
            QMessageBox.information(
                self, tr("Dataset"),
                tr("No hay cajas en esta imagen. Corrige o dibuja antes de enviarla."))
            return
        root = Path(__file__).resolve().parents[4] / "dataset_correcciones"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, root / "images" / path.name)
        (root / "labels" / f"{path.stem}.txt").write_text(
            "\n".join(self._yolo_lines(dets, W, H)), encoding="utf-8")
        cls_path = root / "classes.txt"
        if not cls_path.exists():
            cls_path.write_text("\n".join(DEFAULT_CLASSES), encoding="utf-8")
        n = len(list((root / "labels").glob("*.txt")))
        self.lbl_status.setText(
            f"Enviada al dataset de reentrenamiento — {n} imagen(es) "
            f"acumuladas en dataset_correcciones/. Cuando juntes ~50, haz "
            f"fine-tuning en el Entrenador.")
