"""Page 8 — Comparing models against each other (summary table + per-image)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
)

from ._base import DetectorPage
from polyx.gui import theme as T
from polyx.model.metrics import LABEL_TP, LABEL_FP, LABEL_FN
from polyx.model.i18n import tr


def _safe_div(a, b): return a / b if b else 0.0


class ComparePage(DetectorPage):
    STEP_N = 8
    STEP_TITLE = tr("Comparar")
    STEP_DESCRIPTION = (
        tr("Si cargaste más de un modelo, aquí ves tabla resumen con todas las métricas "
        "lado a lado y cuántas detecciones hizo cada uno por imagen.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        c1, l1 = self.card(tr("Resumen por modelo"), "resultados")
        self.tbl_summary = QTableWidget(0, 8)
        self.tbl_summary.setHorizontalHeaderLabels(
            ["Modelo", "Imágenes", "Detecciones", "Conf media",
             LABEL_TP, LABEL_FP, LABEL_FN, "F1"]
        )
        self.tbl_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_summary.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_summary.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_summary.setAlternatingRowColors(True)
        self.tbl_summary.setMinimumHeight(180)
        l1.addWidget(self.tbl_summary)
        self.body.addWidget(c1)

        c2, l2 = self.card(tr("Detecciones por imagen"), "imagenes")
        self.tbl_per_image = QTableWidget(0, 1)
        self.tbl_per_image.setMinimumHeight(280)
        self.tbl_per_image.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_per_image.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_per_image.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_per_image.setAlternatingRowColors(True)
        l2.addWidget(self.tbl_per_image)
        self.body.addWidget(c2)

        self.state.run_finished.connect(self.refresh)

    def refresh(self):
        state = self.state
        active_indices = [i for i, s in enumerate(state.model_slots) if s.path is not None]

        # Summary
        self.tbl_summary.setRowCount(len(active_indices))
        for r, mi in enumerate(active_indices):
            slot = state.model_slots[mi]
            rs = state.results.get(mi, [])
            n_img = len({x.image_path for x in rs})
            n_det = sum(len(x.predictions) for x in rs)
            confs = [p.conf for x in rs for p in x.predictions]
            tp = sum(x.tp for x in rs)
            fp = sum(x.fp for x in rs)
            fn = sum(x.fn for x in rs)
            prec = _safe_div(tp, tp + fp); rec = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * prec * rec, prec + rec)
            cm = sum(confs)/len(confs) if confs else 0
            self.tbl_summary.setItem(r, 0, QTableWidgetItem(slot.alias))
            for col, v in enumerate([n_img, n_det, f"{cm:.3f}", tp, fp, fn,
                                     f"{f1:.3f}" if any(x.has_gt for x in rs) else "—"], start=1):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl_summary.setItem(r, col, it)

        # Per image: one column per model
        if not active_indices:
            self.tbl_per_image.setColumnCount(1); self.tbl_per_image.setRowCount(0); return
        image_set = sorted({x.image_path for mi in active_indices for x in state.results.get(mi, [])})
        cols = ["Imagen"] + [state.model_slots[mi].alias for mi in active_indices]
        self.tbl_per_image.setColumnCount(len(cols))
        self.tbl_per_image.setHorizontalHeaderLabels(cols)
        self.tbl_per_image.setRowCount(len(image_set))

        # Index results by (model_idx, image_path) -> number of predictions
        idx_map = {}
        for mi in active_indices:
            for r in state.results.get(mi, []):
                idx_map[(mi, r.image_path)] = len(r.predictions)
        for r, p in enumerate(image_set):
            self.tbl_per_image.setItem(r, 0, QTableWidgetItem(p.name))
            for c, mi in enumerate(active_indices, start=1):
                it = QTableWidgetItem(str(idx_map.get((mi, p), 0)))
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl_per_image.setItem(r, c, it)
