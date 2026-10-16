"""Página 9 — Informe HTML del entrenamiento.

Genera un .html autocontenido con curvas, métricas finales, parámetros usados
y galería de figuras producidas por Ultralytics (results.png, confusion_matrix,
PR_curve, F1_curve, etc.).
"""
from __future__ import annotations
import base64
import html
import webbrowser
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QCheckBox, QFileDialog,
    QMessageBox, QComboBox,
)

from ._base import TrainerPage
from ...core import theme as T
from ...core import iconos
from ...core.i18n import tr


def _runs_root() -> Path:
    return Path(__file__).resolve().parents[3] / "runs_train"


def _b64_image(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""


def _read_results_csv_full(run_dir: Path) -> tuple[list[str], list[list[float]]]:
    csv = run_dir / "results.csv"
    if not csv.exists(): return [], []
    lines = csv.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2: return [], []
    header = [c.strip() for c in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        try:
            rows.append([float(c.strip()) for c in line.split(",")])
        except Exception:
            continue
    return header, rows


CSS = f"""
*{{box-sizing:border-box;}}
body{{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:{T.DOC.INK};
background:{T.DOC.BG};margin:0;padding:0;line-height:1.55;}}
.container{{max-width:1100px;margin:0 auto;padding:36px 44px 80px;}}
header.cover{{border-bottom:2px solid {T.DOC.INK};padding-bottom:22px;margin-bottom:28px;}}
.kicker{{font-size:10pt;letter-spacing:.14em;text-transform:uppercase;
color:{T.DOC.ACCENT_D};font-weight:600;margin-bottom:8px;}}
h1{{font-size:26pt;margin:4px 0 8px;}}
h2{{font-size:18pt;margin-top:34px;padding-bottom:6px;border-bottom:1px solid {T.DOC.RULE};}}
h3{{font-size:13pt;margin-top:20px;color:{T.DOC.INK2};}}
.meta{{font-size:10pt;color:{T.DOC.INK3};}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0;}}
.kpi{{background:{T.DOC.BG};border:1px solid {T.DOC.RULE};border-radius:8px;padding:14px 16px;}}
.kpi .l{{font-size:9pt;color:{T.DOC.INK3};font-weight:600;letter-spacing:1.4px;text-transform:uppercase;}}
.kpi .v{{font-size:22pt;font-weight:700;color:{T.DOC.INK};}}
.kpi .b{{height:3px;border-radius:2px;margin-top:8px;background:{T.DOC.ACCENT};}}
table.data{{width:100%;border-collapse:collapse;font-size:10pt;margin:12px 0;}}
table.data th{{background:{T.DOC.BG_SOFT};text-align:left;padding:7px 9px;border-bottom:1px solid {T.DOC.RULE};}}
table.data td{{padding:6px 9px;border-bottom:1px solid {T.DOC.RULE_SOFT};}}
.fig{{margin:18px 0;text-align:center;}}
.fig img{{max-width:100%;border:1px solid {T.DOC.RULE};border-radius:6px;}}
.caption{{font-size:9pt;color:{T.DOC.INK3};margin-top:6px;}}
"""


def build_training_report(run_dir: Path, out_path: Path, include_refs: bool = True) -> Path:
    """Construye el reporte HTML del entrenamiento a partir de la carpeta del run."""
    if not run_dir.exists():
        raise RuntimeError(f"No existe {run_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header, rows = _read_results_csv_full(run_dir)
    metrics_html = ""
    if header and rows:
        # Buscar columnas clave en la última fila
        def col(*names):
            for n in names:
                if n in header:
                    idx = header.index(n)
                    # Una ultima fila truncada (corrida cortada a mitad) tiene
                    # menos celdas que el encabezado: sin esta guarda el
                    # indice se sale del rango y el reporte revienta.
                    if idx < len(rows[-1]):
                        return rows[-1][idx]
                    return None
            return None
        last_epoch = int(col("epoch") or len(rows))
        map50 = col("metrics/mAP50(B)", "metrics/mAP_0.5")
        map95 = col("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95")
        prec  = col("metrics/precision(B)", "metrics/precision")
        rec   = col("metrics/recall(B)", "metrics/recall")
        kpi_html = f"""
        <div class='kpis'>
          <div class='kpi'><div class='l'>Épocas</div><div class='v'>{last_epoch}</div><div class='b'></div></div>
          <div class='kpi'><div class='l'>mAP@50</div><div class='v'>{f"{map50:.3f}" if map50 is not None else "—"}</div><div class='b' style='background:{T.DOC.OK}'></div></div>
          <div class='kpi'><div class='l'>mAP@50-95</div><div class='v'>{f"{map95:.3f}" if map95 is not None else "—"}</div><div class='b' style='background:{T.DOC.VIO}'></div></div>
          <div class='kpi'><div class='l'>P / R</div><div class='v'>{f"{prec:.2f}/{rec:.2f}" if prec is not None and rec is not None else "—"}</div><div class='b' style='background:{T.DOC.WARN}'></div></div>
        </div>
        """
        metrics_html = kpi_html

    # Figuras producidas por Ultralytics
    figs = []
    for name, label in [
        ("results.png",          "Curvas de entrenamiento (loss + métricas)"),
        ("confusion_matrix.png", "Matriz de confusión"),
        ("PR_curve.png",         "Curva Precision–Recall"),
        ("F1_curve.png",         "Curva F1 vs confianza"),
        ("P_curve.png",          "Curva Precision vs confianza"),
        ("R_curve.png",          "Curva Recall vs confianza"),
        ("labels.jpg",           "Distribución de etiquetas en train"),
        ("val_batch0_pred.jpg",  "Predicciones del batch 0 de validación"),
    ]:
        p = run_dir / name
        if p.exists():
            b64 = _b64_image(p)
            if b64:
                figs.append((label, b64))

    figs_html = ""
    for label, b64 in figs:
        figs_html += (
            f"<div class='fig'><img src='data:image/png;base64,{b64}'/>"
            f"<div class='caption'>{html.escape(label)}</div></div>"
        )

    refs = ""
    if include_refs:
        refs = """
        <h2>Referencias</h2>
        <ol>
          <li>Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024). Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. <em>PLoS ONE</em> 19(12): e0314520.</li>
          <li>Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024). Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in <em>Merluccius gayi</em> fish by Nile Red fluorescent staining and image analysis. <em>J. Chil. Chem. Soc.</em> 69(1): 6082-6085.</li>
          <li>Jocher G, Chaurasia A, Qiu J (2023). Ultralytics YOLOv8/v11.</li>
        </ol>
        """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang='es'><head><meta charset='utf-8'>
<title>Reporte de entrenamiento · {html.escape(run_dir.name)}</title>
<style>{CSS}</style>
</head><body><div class='container'>
<header class='cover'>
  <div class='kicker'>Poly-X · Reporte de entrenamiento</div>
  <h1>{html.escape(run_dir.name)}</h1>
  <p class='meta'><strong>Carpeta:</strong> {html.escape(str(run_dir))} &middot; <strong>Generado:</strong> {now}</p>
</header>

<h2>1. Métricas finales</h2>
{metrics_html or "<p>Sin results.csv legible.</p>"}

<h2>2. Curvas y figuras</h2>
{figs_html or "<p>No se encontraron figuras en el run.</p>"}

{refs}

<p style='margin-top:60px;color:{T.DOC.INK3};font-size:9.5pt;border-top:1px solid {T.DOC.RULE};padding-top:14px;'>
© Cristofher Ferrada · Poly-X · Reporte autogenerado.</p>
</div></body></html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path


class _ReportWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, run_dir: Path, out_path: Path, include_refs: bool, parent=None):
        super().__init__(parent)
        self.run_dir = run_dir; self.out_path = out_path; self.include_refs = include_refs

    def run(self):
        try:
            p = build_training_report(self.run_dir, self.out_path, self.include_refs)
            self.finished_ok.emit(str(p))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class InformePage(TrainerPage):
    PAGE_ICON = "reporte"
    PAGE_TITLE = tr("Informe del entrenamiento")
    PAGE_DESCRIPTION = (
        tr("Genera un informe HTML autocontenido con las curvas y métricas del run "
        "elegido. Listo para convertir a PDF (Ctrl+P en el navegador).")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.worker: _ReportWorker | None = None

        c1, l1 = self.card(tr("Selección de run"), "dataset")
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(QLabel(tr("Run:")))
        self.combo = QComboBox(); self.combo.setMinimumWidth(360)
        row.addWidget(self.combo, 1)
        btn = QPushButton("🔄"); btn.setFixedWidth(36); btn.clicked.connect(self._refresh)
        row.addWidget(btn)
        l1.addLayout(row)
        self.chk_refs = QCheckBox(tr("Incluir referencias bibliográficas"))
        self.chk_refs.setChecked(True)
        l1.addWidget(self.chk_refs)
        self.body.addWidget(c1)

        c2, l2 = self.card(tr("Generar"), "texto")
        rr = QHBoxLayout(); rr.setSpacing(8)
        b1 = QPushButton(tr("Generar reporte HTML"))
        b1.setIcon(iconos.icono("reporte", 15, T.ON_ACCENT))
        b1.setStyleSheet(
            f"background: {T.DOC.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        b1.setCursor(Qt.PointingHandCursor)
        b1.clicked.connect(self._generate)
        rr.addWidget(b1)
        b2 = QPushButton(tr("Guardar como…"))
        b2.setIcon(iconos.icono("guardar", 15, T.INK2))
        b2.clicked.connect(self._save_as)
        rr.addWidget(b2)
        rr.addStretch(1)
        l2.addLayout(rr)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"color: {T.DOC.INK3}; font-size: 9.5pt; border: none;")
        l2.addWidget(self.lbl_status)
        self.body.addWidget(c2)

        self._refresh()

    def _refresh(self):
        self.combo.clear()
        root = _runs_root()
        if not root.exists(): return
        runs = sorted([d for d in root.iterdir() if d.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for d in runs:
            self.combo.addItem(d.name, str(d))
        # auto-seleccionar el último run del state si existe
        if self.state.run_dir:
            idx = self.combo.findData(str(self.state.run_dir))
            if idx >= 0: self.combo.setCurrentIndex(idx)

    def _current_run(self) -> Path | None:
        s = self.combo.currentData()
        return Path(s) if s else None

    def _generate(self):
        run = self._current_run()
        if not run:
            QMessageBox.warning(self, tr("Sin run"), tr("No hay runs disponibles.")); return
        out = run / "reporte_entrenamiento.html"
        self._run_worker(run, out)

    def _save_as(self):
        run = self._current_run()
        if not run:
            QMessageBox.warning(self, tr("Sin run"), tr("No hay runs disponibles.")); return
        f, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte", str(run / "reporte_entrenamiento.html"), "HTML (*.html)"
        )
        if f:
            self._run_worker(run, Path(f))

    def _run_worker(self, run: Path, out: Path):
        self.lbl_status.setText(tr("Generando informe…"))
        self.worker = _ReportWorker(run, out, self.chk_refs.isChecked())
        self.worker.finished_ok.connect(self._on_ok)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_ok(self, p: str):
        self.lbl_status.setText(f"✓ Generado: {p}")
        try: webbrowser.open(Path(p).as_uri())
        except Exception: pass

    def _on_fail(self, msg: str):
        self.lbl_status.setText(tr("✗ Error"))
        QMessageBox.critical(self, tr("Falló"), msg)
