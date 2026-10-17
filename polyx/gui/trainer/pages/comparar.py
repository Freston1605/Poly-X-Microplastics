"""Página 7 — Comparar. Tabla de todos los runs anteriores con sus métricas."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView,
)

from ._base import TrainerPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.model.i18n import tr
from polyx.model.plataforma import abrir_en_el_sistema


def _runs_root() -> Path:
    return Path(__file__).resolve().parents[4] / "runs_train"


def _read_results_csv(run_dir: Path) -> dict | None:
    csv = run_dir / "results.csv"
    if not csv.exists(): return None
    try:
        lines = csv.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2: return None
        header = [c.strip() for c in lines[0].split(",")]
        last = [c.strip() for c in lines[-1].split(",")]
        d = dict(zip(header, last))
        # extraer las que nos interesan (tolerar nombres v8/v11)
        def pick(*keys):
            for k in keys:
                if k in d:
                    try: return float(d[k])
                    except ValueError: pass
            return None
        return {
            "epochs": pick("epoch") or 0,
            "mAP50": pick("metrics/mAP50(B)", "metrics/mAP_0.5"),
            "mAP50-95": pick("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95"),
            "precision": pick("metrics/precision(B)", "metrics/precision"),
            "recall": pick("metrics/recall(B)", "metrics/recall"),
            "box_loss": pick("train/box_loss"),
        }
    except Exception:
        return None


class CompararPage(TrainerPage):
    PAGE_ICON = "resultados"
    PAGE_TITLE = tr("Comparar runs")
    PAGE_DESCRIPTION = (
        tr("Tabla con todos los entrenamientos anteriores en runs_train/. Útil para comparar "
        "configuraciones y elegir el mejor modelo.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        c1, l1 = self.card(tr("Runs disponibles"), "dataset")
        row = QHBoxLayout()
        btn = QPushButton(tr("Refrescar lista"))
        btn.setIcon(iconos.icono("recargar", 15, T.INK2))
        btn.clicked.connect(self.refresh)
        row.addWidget(btn)
        btn2 = QPushButton(tr("Abrir carpeta runs_train/"))
        btn2.setIcon(iconos.icono("dataset", 15, T.INK2))
        btn2.clicked.connect(self._open_root)
        row.addWidget(btn2)
        row.addStretch(1)
        l1.addLayout(row)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Run", "Épocas", "mAP@50", "mAP@50-95", "Precision", "Recall", "Box loss"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(360)
        self.table.cellDoubleClicked.connect(self._open_run)
        l1.addWidget(self.table)
        self.body.addWidget(c1)

        self.refresh()

    def refresh(self):
        root = _runs_root()
        if not root.exists():
            self.table.setRowCount(0); return
        runs = sorted([d for d in root.iterdir() if d.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        self.table.setRowCount(len(runs))
        for r, d in enumerate(runs):
            self.table.setItem(r, 0, QTableWidgetItem(d.name))
            metrics = _read_results_csv(d)
            if metrics:
                self.table.setItem(r, 1, QTableWidgetItem(f"{int(metrics.get('epochs') or 0)}"))
                for col, k in enumerate(["mAP50", "mAP50-95", "precision", "recall", "box_loss"], start=2):
                    v = metrics.get(k)
                    self.table.setItem(r, col, QTableWidgetItem(f"{v:.3f}" if v is not None else "—"))
            else:
                for col in range(1, 7):
                    self.table.setItem(r, col, QTableWidgetItem("—"))
            self.table.item(r, 0).setData(Qt.UserRole, str(d))

    def _open_run(self, row: int, col: int):
        path = self.table.item(row, 0).data(Qt.UserRole)
        if path and Path(path).exists():
            abrir_en_el_sistema(path)

    def _open_root(self):
        root = _runs_root()
        root.mkdir(parents=True, exist_ok=True)
        abrir_en_el_sistema(str(root))
