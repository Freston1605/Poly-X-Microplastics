"""Page 8 — `Exportar`. Converts the model to ONNX, TensorRT, TFLite, CoreML."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QComboBox, QCheckBox, QMessageBox, QPlainTextEdit,
)

from ._base import TrainerPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.model.i18n import tr


FORMATS = [
    ("ONNX",      "onnx",        "Más portable. Para CPU o GPU vía onnxruntime."),
    ("TorchScript", "torchscript", "Modelo nativo de PyTorch."),
    ("TensorRT",  "engine",      "Solo NVIDIA. Inferencia más rápida en producción."),
    ("OpenVINO",  "openvino",    "Intel CPU/iGPU."),
    ("TFLite",    "tflite",      "Mobile/Edge devices."),
    ("CoreML",    "coreml",      "Apple Silicon (M1/M2/M3) y iOS."),
]


class _ExportWorker(QThread):
    log_line = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, weights: str, fmt: str, imgsz: int, half: bool, parent=None):
        super().__init__(parent)
        self.weights = weights; self.fmt = fmt
        self.imgsz = imgsz; self.half = half

    def run(self):
        try:
            from ultralytics import YOLO
            self.log_line.emit(f"[INFO] Cargando {self.weights}")
            m = YOLO(self.weights)
            self.log_line.emit(f"[INFO] Exportando a {self.fmt} (imgsz={self.imgsz}, half={self.half})")
            out = m.export(format=self.fmt, imgsz=self.imgsz, half=self.half)
            self.finished_ok.emit(str(out))
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class ExportPage(TrainerPage):
    PAGE_ICON = "exportar"
    PAGE_TITLE = tr("Exportar modelo")
    PAGE_DESCRIPTION = (
        tr("Convierte el modelo entrenado a otros formatos para producción: ONNX (portable), "
        "TensorRT (NVIDIA rápido), TFLite (móvil), CoreML (Apple), etc.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.worker: _ExportWorker | None = None

        c1, l1 = self.card(tr("Modelo a exportar"), "paquete")
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Pesos .pt:")))
        self.ed_w = QLineEdit()
        # autofill with best.pt from the last run
        if state.run_dir:
            best = Path(state.run_dir) / "weights" / "best.pt"
            if best.exists(): self.ed_w.setText(str(best))
        row.addWidget(self.ed_w, 1)
        b = QPushButton("…"); b.setFixedWidth(36); b.clicked.connect(self._browse)
        row.addWidget(b)
        l1.addLayout(row)
        self.body.addWidget(c1)

        c2, l2 = self.card(tr("Formato y opciones"), "ajustes")
        g = QGridLayout(); g.setHorizontalSpacing(20); g.setVerticalSpacing(10)
        g.addWidget(QLabel(tr("Formato:")), 0, 0)
        self.combo = QComboBox()
        for name, code, desc in FORMATS:
            self.combo.addItem(f"{name}  ({code})", code)
        self.combo.currentIndexChanged.connect(self._update_desc)
        g.addWidget(self.combo, 0, 1)

        g.addWidget(QLabel(tr("imgsz:")), 1, 0)
        self.ed_imgsz = QLineEdit(str(state.params.imgsz)); self.ed_imgsz.setMaximumWidth(100)
        g.addWidget(self.ed_imgsz, 1, 1)
        self.cb_half = QCheckBox(tr("FP16 (half precision)")); self.cb_half.setChecked(True)
        g.addWidget(self.cb_half, 2, 1)
        l2.addLayout(g)

        self.lbl_desc = QLabel("")
        self.lbl_desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        self.lbl_desc.setWordWrap(True)
        l2.addWidget(self.lbl_desc)

        btn = QPushButton(tr("Exportar"))
        btn.setIcon(iconos.icon("exportar", 15, T.ON_ACCENT))
        btn.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self._run)
        l2.addWidget(btn, 0, Qt.AlignLeft)
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

        self._update_desc()

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar .pt", "", "Pesos (*.pt)")
        if f: self.ed_w.setText(f)

    def _update_desc(self):
        idx = self.combo.currentIndex()
        # The description is translated here and not in FORMATS because that list also fixes
        # the internal code of the format, which is never translated.
        self.lbl_desc.setText(tr(FORMATS[idx][2]))

    def _run(self):
        w = self.ed_w.text().strip()
        if not w or not Path(w).exists():
            QMessageBox.warning(self, tr("Falta modelo"), tr("Selecciona un .pt válido.")); return
        try: imgsz = int(self.ed_imgsz.text())
        except ValueError: imgsz = 640
        fmt = self.combo.currentData()
        self.log.clear()
        self.worker = _ExportWorker(w, fmt, imgsz, self.cb_half.isChecked())
        self.worker.log_line.connect(self.log.appendPlainText)
        self.worker.finished_ok.connect(lambda p: (
            self.log.appendPlainText(f"[OK] Exportado: {p}"),
            QMessageBox.information(self, tr("Exportación lista"), f"Archivo generado:\n{p}")
        ))
        self.worker.failed.connect(lambda m: QMessageBox.critical(self, tr("Falló"), m))
        self.worker.start()
