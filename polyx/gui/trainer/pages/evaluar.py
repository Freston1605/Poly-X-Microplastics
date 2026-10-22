"""Page 6 — `Evaluar`. Validation of any .pt over a YOLO dataset."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QPlainTextEdit,
)

from ._base import TrainerPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.gui.widgets import KPICard
from polyx.ml.i18n import tr


class _EvalWorker(QThread):
    log_line = Signal(str)
    metrics_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, weights: str, data_yaml: str, imgsz: int, device: str, parent=None):
        super().__init__(parent)
        self.weights = weights; self.data_yaml = data_yaml
        self.imgsz = imgsz; self.device = device

    def run(self):
        try:
            from ultralytics import YOLO
            self.log_line.emit(f"[INFO] Cargando {self.weights}")
            m = YOLO(self.weights)
            self.log_line.emit(f"[INFO] Validando sobre {self.data_yaml} (imgsz={self.imgsz})")
            res = m.val(data=self.data_yaml, imgsz=self.imgsz, device=self.device, verbose=True)
            d = {
                "mAP50": float(getattr(res.box, "map50", 0.0)),
                "mAP50-95": float(getattr(res.box, "map", 0.0)),
                "precision": float(getattr(res.box, "mp", 0.0)),
                "recall": float(getattr(res.box, "mr", 0.0)),
            }
            self.metrics_ready.emit(d)
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class EvaluatePage(TrainerPage):
    PAGE_ICON = "evaluar"
    PAGE_TITLE = tr("Evaluar")
    PAGE_DESCRIPTION = (
        tr("Valida cualquier modelo .pt sobre un dataset YOLO (data.yaml). Útil para "
        "comprobar la calidad de un best.pt sobre un test set distinto al de entrenamiento.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.worker: _EvalWorker | None = None

        c1, l1 = self.card(tr("Configuración"), "ajustes")
        row1 = QHBoxLayout(); row1.setSpacing(8)
        row1.addWidget(QLabel(tr("Modelo .pt:")))
        self.ed_w = QLineEdit(); row1.addWidget(self.ed_w, 1)
        b1 = QPushButton("…"); b1.setFixedWidth(36); b1.clicked.connect(self._browse_w)
        row1.addWidget(b1)
        l1.addLayout(row1)

        row2 = QHBoxLayout(); row2.setSpacing(8)
        row2.addWidget(QLabel(tr("data.yaml:")))
        self.ed_y = QLineEdit()
        if state.dataset.yaml_path:
            self.ed_y.setText(str(state.dataset.yaml_path))
        row2.addWidget(self.ed_y, 1)
        b2 = QPushButton("…"); b2.setFixedWidth(36); b2.clicked.connect(self._browse_y)
        row2.addWidget(b2)
        l1.addLayout(row2)

        row3 = QHBoxLayout(); row3.setSpacing(20)
        row3.addWidget(QLabel(tr("imgsz:")))
        self.ed_imgsz = QLineEdit(str(state.params.imgsz)); self.ed_imgsz.setMaximumWidth(80)
        row3.addWidget(self.ed_imgsz)
        row3.addWidget(QLabel(tr("device:")))
        self.ed_dev = QLineEdit(state.params.device); self.ed_dev.setMaximumWidth(80)
        row3.addWidget(self.ed_dev)
        row3.addStretch(1)
        l1.addLayout(row3)

        btn = QPushButton(tr("Iniciar validación"))
        btn.setIcon(iconos.icon("ejecutar", 15, T.ON_ACCENT))
        btn.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self._run)
        l1.addWidget(btn, 0, Qt.AlignLeft)
        self.body.addWidget(c1)

        # Results
        c2, l2 = self.card(tr("Resultados"), "resultados")
        g = QGridLayout(); g.setSpacing(12)
        self.kpi_map50 = KPICard("mAP@50", T.ACCENT)
        self.kpi_map95 = KPICard("mAP@50-95", T.VIO)
        self.kpi_prec  = KPICard("Precision", T.OK)
        self.kpi_rec   = KPICard("Recall", T.WARN)
        g.addWidget(self.kpi_map50, 0, 0); g.addWidget(self.kpi_map95, 0, 1)
        g.addWidget(self.kpi_prec, 0, 2);  g.addWidget(self.kpi_rec, 0, 3)
        l2.addLayout(g)
        self.body.addWidget(c2)

        c3, l3 = self.card("Log", "registro")
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet(
            f"background: #0d1117; color: #c9d1d9; font-family: Consolas, monospace; "
            f"font-size: 9.5pt; border: 1px solid {T.RULE}; border-radius: 6px;"
        )
        self.log.setMinimumHeight(220)
        l3.addWidget(self.log)
        self.body.addWidget(c3)

    def _browse_w(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar .pt", "", "Pesos (*.pt)")
        if f: self.ed_w.setText(f)

    def _browse_y(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar data.yaml", "", "YAML (*.yaml *.yml)")
        if f: self.ed_y.setText(f)

    def _run(self):
        w = self.ed_w.text().strip(); y = self.ed_y.text().strip()
        if not w or not Path(w).exists():
            QMessageBox.warning(self, tr("Falta modelo"), tr("Selecciona un .pt válido.")); return
        if not y or not Path(y).exists():
            QMessageBox.warning(self, tr("Falta dataset"), tr("Selecciona un data.yaml válido.")); return
        try: imgsz = int(self.ed_imgsz.text())
        except ValueError: imgsz = 640
        self.log.clear()
        self.worker = _EvalWorker(w, y, imgsz, self.ed_dev.text().strip() or "0")
        self.worker.log_line.connect(self.log.appendPlainText)
        self.worker.metrics_ready.connect(self._on_metrics)
        self.worker.failed.connect(lambda m: QMessageBox.critical(self, tr("Falló"), m))
        self.worker.start()

    def _on_metrics(self, d: dict):
        self.kpi_map50.set_value(f"{d['mAP50']:.3f}")
        self.kpi_map95.set_value(f"{d['mAP50-95']:.3f}")
        self.kpi_prec.set_value(f"{d['precision']:.3f}")
        self.kpi_rec.set_value(f"{d['recall']:.3f}")
        self.log.appendPlainText(f"[OK] Validación completada.")
