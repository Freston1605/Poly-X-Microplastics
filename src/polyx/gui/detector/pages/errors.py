"""Page 7 — `Errores` (False Positives / False Negatives / Misclassified).

Filterable list + preview. Internally the short codes FP/FN/MISCLS are still
used for the logic; the user is shown the full names.
"""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton,
)

from ._base import DetectorPage
from polyx.gui import theme as T
from polyx.core.quality.metrics import match_image, LABEL_FP, LABEL_FN, LABEL_MISCLS
from polyx.core.i18n import tr
from polyx.core.platform import open_in_system


class ErrorsPage(DetectorPage):
    STEP_N = 7
    STEP_TITLE = tr("Errores")
    STEP_DESCRIPTION = (
        tr("Lista de cajas problemáticas (solo si hay Ground Truth). "
        "Falsos Positivos: detección sin GT cercano. Falsos Negativos: GT no detectado. "
        "Mal Clasificados: bien localizado, mala clase.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Filters ──
        c1, l1 = self.card(tr("Filtros"), "buscar")
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(QLabel(tr("Tipo:")))
        self.combo_type = QComboBox()
        # Displayed text (full name) + internal data (short code)
        self.combo_type.addItem(tr("Todos"), "ALL")
        self.combo_type.addItem(LABEL_FP, "FP")
        self.combo_type.addItem(LABEL_FN, "FN")
        self.combo_type.addItem(LABEL_MISCLS, "MISCLS")
        self.combo_type.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.combo_type)

        row.addWidget(QLabel(tr("Modelo:")))
        self.combo_model = QComboBox()
        self.combo_model.addItem(tr("Todos"), -1)
        self.combo_model.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.combo_model)

        row.addStretch(1)
        self.lbl_count = QLabel(tr("0 errores"))
        self.lbl_count.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        row.addWidget(self.lbl_count)
        l1.addLayout(row)
        self.body.addWidget(c1)

        # ── Error table ──
        c2, l2 = self.card(tr("Cajas con error"), "errores")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Modelo", "Imagen", "Tipo", "Clase GT", "Clase Pred", "Conf"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(280)
        self.table.cellDoubleClicked.connect(self._open_image)
        l2.addWidget(self.table)
        self.body.addWidget(c2)

        # Connect
        self.state.run_finished.connect(self.refresh)
        self.state.models_changed.connect(self._refresh_model_combo)

    def _refresh_model_combo(self):
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self.combo_model.addItem("Todos", -1)
        for i, slot in enumerate(self.state.model_slots):
            if slot.path is not None:
                self.combo_model.addItem(slot.alias, i)
        self.combo_model.blockSignals(False)

    def refresh(self):
        state = self.state
        # currentData() returns the internal code (ALL/FP/FN/MISCLS), not the text
        type_code = self.combo_type.currentData()
        if type_code is None:
            type_code = "ALL"
        model_filter = self.combo_model.currentData()
        rows = []
        iou_tp = state.params.iou_tp

        for mi, rs in state.results.items():
            if model_filter is not None and model_filter != -1 and mi != model_filter:
                continue
            alias = state.model_slots[mi].alias
            for r in rs:
                if not r.has_gt: continue
                m = match_image(r.predictions, r.gt, iou_thr=iou_tp)
                # False Positives
                if type_code in ("ALL", "FP"):
                    for pi in m.fp_idx:
                        d = r.predictions[pi]
                        rows.append((alias, r.image_path, "FP", "—", d.class_name, f"{d.conf:.2f}"))
                # False Negatives
                if type_code in ("ALL", "FN"):
                    for gi in m.fn_idx:
                        d = r.gt[gi]
                        rows.append((alias, r.image_path, "FN", d.class_name, "—", "—"))
                # Misclassified
                if type_code in ("ALL", "MISCLS"):
                    for pi, gi in m.miscls_pairs:
                        rows.append((
                            alias, r.image_path, "MISCLS",
                            r.gt[gi].class_name, r.predictions[pi].class_name,
                            f"{r.predictions[pi].conf:.2f}"
                        ))

        self.lbl_count.setText(f"{len(rows)} error{'es' if len(rows)!=1 else ''}")
        self.table.setRowCount(len(rows))
        badge_colors = {"FP": T.WARN, "FN": T.ERR, "MISCLS": T.VIO}
        label_by_code = {"FP": LABEL_FP, "FN": LABEL_FN, "MISCLS": LABEL_MISCLS}
        from PySide6.QtGui import QBrush, QColor
        for i, (alias, p, code, cgt, cpr, cf) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(alias))
            self.table.setItem(i, 1, QTableWidgetItem(p.name))
            it = QTableWidgetItem(label_by_code.get(code, code))
            it.setTextAlignment(Qt.AlignCenter)
            it.setForeground(QBrush(QColor(badge_colors.get(code, T.INK))))
            self.table.setItem(i, 2, it)
            self.table.setItem(i, 3, QTableWidgetItem(cgt))
            self.table.setItem(i, 4, QTableWidgetItem(cpr))
            self.table.setItem(i, 5, QTableWidgetItem(cf))
            self.table.item(i, 0).setData(Qt.UserRole, alias)
            self.table.item(i, 1).setData(Qt.UserRole, str(p))

    def _open_image(self, row: int, col: int):
        if row < 0 or self.state.run_dir is None: return
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))
        annot = self.state.run_dir / alias / f"{img.stem}_annot.png"
        if annot.exists():
            open_in_system(str(annot))
