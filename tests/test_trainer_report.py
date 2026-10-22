"""build_training_report end-to-end tests.

``build_training_report`` (``polyx/gui/trainer/pages/informe.py``) is the deliverable
that proves a training run succeeded: it turns a runner-structured run directory
(Ultralytics ``results.csv`` + figure PNGs + ``weights/best.pt``) into a
self-contained HTML report with KPIs, embedded base64 figures and bibliography.

These tests build synthetic run directories under ``tmp_path`` and exercise the
real report pipeline end-to-end — no cv2, no Qt event loop, no Ultralytics. The
conftest autouse fixture (``_aislar_logs_suite``) isolates ``POLYX_LOG_DIR``
per test, matching the rest of the suite.

Two latent bugs are pinned here:

- ``IndexError`` when the last ``results.csv`` data row is shorter than the
  header (a run truncated mid-epoch): ``col()`` now bounds-checks the index
  against the last row and returns ``None`` (KPI shows the em-dash).
- XSS via the run name: the ``<title>`` sink rendered ``run_dir.name`` raw;
  it is now HTML-escaped like the ``<h1>`` and meta sinks.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pytest

from polyx.gui.trainer.pages.summary import build_training_report

# A 1x1 transparent PNG — small valid PNG bytes for figure fixtures.
# Written with Path.write_bytes, never cv2.imwrite (fails silently on
# Windows with non-ASCII paths).
PNG_MINI = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# The Ultralytics results.csv column names — the runner's contract surface.
ULTRALYTICS_HEADER = [
    "epoch",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
]

# Default realistic fixture: header + one data row per epoch (3 epochs).
CSV_3_EPOCAS = [
    ULTRALYTICS_HEADER,
    [1, 0.51, 0.42, 0.61, 0.31, 0.100, 0.050, 0.030, 0.110, 0.060, 0.040, 0.01, 0.02, 0.03],
    [2, 0.53, 0.44, 0.63, 0.33, 0.090, 0.045, 0.028, 0.100, 0.055, 0.038, 0.01, 0.02, 0.03],
    [3, 0.55, 0.46, 0.65, 0.35, 0.080, 0.040, 0.026, 0.090, 0.050, 0.036, 0.01, 0.02, 0.03],
]


# ── Helpers ──────────────────────────────────────────────────────────

def _run_dir(root: Path, name: str, *, csv_rows=None, figures=None) -> Path:
    """Creates ``runs_train/<name>/`` with a realistic results.csv and figures.

    ``csv_rows=None`` writes the default 3-epoch Ultralytics-format CSV;
    ``csv_rows=False`` writes no results.csv at all (to probe absence).
    ``figures`` is an iterable of filenames written as small valid PNGs.
    """
    run_dir = root / "runs_train" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if csv_rows is None:
        csv_rows = CSV_3_EPOCAS
    if csv_rows is not False:
        lineas = [",".join(str(c) for c in fila) for fila in csv_rows]
        (run_dir / "results.csv").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    for fname in (figures or []):
        (run_dir / fname).write_bytes(PNG_MINI)
    return run_dir


def _generar(run_dir: Path, destino: Path, **kwargs) -> str:
    """Runs build_training_report and returns the generated HTML text."""
    out = build_training_report(run_dir, destino, **kwargs)
    return out.read_text(encoding="utf-8")


# ── 1. Self-contained HTML ───────────────────────────────────────────

class TestHtmlAutocontenido:
    def test_reporte_genera_html_autocontenido(self, tmp_path):
        """Output exists, is non-empty, starts with <!doctype html>, has <style>."""
        run_dir = _run_dir(tmp_path, "basico")

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert html.startswith("<!doctype html>")
        assert "<style>" in html
        assert html.strip()

    def test_reporte_autocontenido_sin_referencias_externas(self, tmp_path):
        """No http://, https:// or src="file: — every asset is inline base64/CSS.

        Run with the default ``include_refs=True``: even the bibliography is
        plain text with no URLs, so the document is fully self-contained.
        """
        run_dir = _run_dir(tmp_path, "autocontenido", figures=["results.png"])

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert "http://" not in html
        assert "https://" not in html
        assert 'src="file:' not in html


# ── 2. Metrics extraction and KPI rendering ──────────────────────────

class TestMetricas:
    def test_metricas_extraidas_de_results_csv(self, tmp_path):
        """KPIs show the LAST row's mAP50, mAP50-95, precision and recall."""
        run_dir = _run_dir(tmp_path, "metricas")  # default 3-epoch CSV

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert "mAP@50" in html and "mAP@50-95" in html and "P / R" in html
        assert "0.650" in html          # mAP50  last row (0.65 → :.3f)
        assert "0.350" in html          # mAP95  last row (0.35 → :.3f)
        assert "0.55/0.46" in html      # P / R  last row (0.55/0.46 → :.2f)

    def test_kpi_con_metricas_ausentes(self, tmp_path):
        """CSV with only the epoch column: metric KPIs show the em-dash."""
        run_dir = _run_dir(tmp_path, "sin_metricas", csv_rows=[["epoch"], [1], [2], [3]])

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert "Épocas" in html
        assert "<div class='v'>3</div>" in html   # epoch KPI still reads data
        assert html.count("—") == 3               # mAP50, mAP50-95, P/R → em-dash


# ── 3. Figures ────────────────────────────────────────────────────────

class TestFiguras:
    def test_figuras_embebidas_como_base64(self, tmp_path):
        """Existing PNGs are embedded as data:image/png;base64, blocks."""
        run_dir = _run_dir(
            tmp_path, "figuras",
            figures=["results.png", "confusion_matrix.png"],
        )

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert html.count("data:image/png;base64,") == 2

    def test_figures_inexistentes_no_rompen(self, tmp_path):
        """No figure files: graceful message, no base64 image blocks."""
        run_dir = _run_dir(tmp_path, "sin_figuras")

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert "No se encontraron figuras en el run." in html
        assert "data:image/png;base64," not in html

    def test_figura_ilegible_se_omite_sin_reventar(self, tmp_path):
        """An unreadable figure (results.png is a directory) is skipped, not fatal.

        ``_b64_image`` catches the read error and the figure is omitted; the
        report still completes with the remaining content.
        """
        run_dir = _run_dir(tmp_path, "figura_rota")
        (run_dir / "results.png").mkdir()   # directorio donde PNG: read_bytes revienta

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert "data:image/png;base64," not in html
        assert "No se encontraron figuras en el run." in html


# ── 4. Hostile input ─────────────────────────────────────────────────

class TestEntradaHostil:
    def test_nombre_de_run_hostil_escapado(self, tmp_path):
        """XSS payload in the run dir name never appears raw in the HTML."""
        payload = "<script>alert(1)</script>"
        run_dir = _run_dir(tmp_path, payload)

        texto = _generar(run_dir, tmp_path / "reporte.html")

        assert payload not in texto
        assert html.escape(payload) in texto   # rendered escaped (h1/title/meta)

    def test_csv_con_ultima_fila_corta_no_revienta(self, tmp_path):
        """Truncated last row: no IndexError; missing columns show em-dash.

        The last data row has a single cell while the header has 14. Without
        the ``col()`` bounds guard this raises IndexError inside the KPI
        rendering and the report worker emits a failed signal.
        """
        filas = [
            ULTRALYTICS_HEADER,
            [1, 0.50, 0.40, 0.60, 0.30, 0.100, 0.050, 0.030, 0.110, 0.060, 0.040, 0.01, 0.02, 0.03],
            [2],   # corrida corta a mitad: la ultima fila viene truncada
        ]
        run_dir = _run_dir(tmp_path, "truncada", csv_rows=filas)

        html = _generar(run_dir, tmp_path / "reporte.html")   # no IndexError

        assert "Épocas" in html
        assert "<div class='v'>2</div>" in html   # epoch still read from the short row
        assert html.count("—") == 3               # mAP50, mAP50-95, P/R → em-dash


# ── 5. Absence and failure paths ─────────────────────────────────────

class TestAusenciaYFalla:
    def test_run_dir_ausente_lanza_runtime_error(self, tmp_path):
        """Non-existent run directory bubbles a RuntimeError."""
        with pytest.raises(RuntimeError, match="No existe"):
            build_training_report(tmp_path / "runs_train" / "no_existe", tmp_path / "reporte.html")

    def test_sin_results_csv_muestra_mensaje_graceful(self, tmp_path):
        """Run dir without results.csv: graceful message, no KPI block."""
        run_dir = _run_dir(tmp_path, "sin_csv", csv_rows=False)

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert "Sin results.csv legible." in html
        assert "mAP@50" not in html      # KPI block omitted entirely

    def test_results_csv_ilegible_propaga_el_error(self, tmp_path):
        """Unreadable results.csv (a directory) bubbles OSError, not silent garbage."""
        run_dir = _run_dir(tmp_path, "csv_roto", csv_rows=False)
        (run_dir / "results.csv").mkdir()

        with pytest.raises(OSError):
            build_training_report(run_dir, tmp_path / "reporte.html")


# ── 6. References flag ───────────────────────────────────────────────

class TestReferencias:
    def test_include_refs_false_omite_referencias(self, tmp_path):
        """include_refs=False drops the heading and the bibliographic entries."""
        run_dir = _run_dir(tmp_path, "refs")

        con_refs = _generar(run_dir, tmp_path / "con_refs.html")
        sin_refs = _generar(run_dir, tmp_path / "sin_refs.html", include_refs=False)

        assert "Referencias" in con_refs
        assert "Jocher" in con_refs
        assert "Referencias" not in sin_refs
        assert "Jocher" not in sin_refs


# ── 7. Runner-to-report contract (S02 → S03) ─────────────────────────

class TestContratoRunner:
    def test_contrato_runner_a_reporte(self, tmp_path):
        """A run dir shaped exactly as TrainerRunner leaves it is consumed.

        Ultralytics (invoked by the runner) writes results.csv with the real
        column names plus the figure PNGs at the run root and weights/best.pt
        under weights/. The report must extract the real metric values from
        the last row and embed every figure.
        """
        filas_runner = [
            ULTRALYTICS_HEADER,
            [3, 0.790, 0.740, 0.870, 0.520, 0.061, 0.021, 0.014, 0.070, 0.025, 0.017, 0.005, 0.005, 0.005],
            [4, 0.800, 0.750, 0.880, 0.530, 0.058, 0.020, 0.013, 0.067, 0.024, 0.016, 0.005, 0.005, 0.005],
            [5, 0.812, 0.764, 0.891, 0.543, 0.055, 0.019, 0.012, 0.064, 0.023, 0.015, 0.005, 0.005, 0.005],
        ]
        run_dir = _run_dir(
            tmp_path, "runner",
            csv_rows=filas_runner,
            figures=["results.png", "confusion_matrix.png", "F1_curve.png"],
        )
        (run_dir / "weights").mkdir()
        (run_dir / "weights" / "best.pt").write_bytes(b"pesos falsos")

        html = _generar(run_dir, tmp_path / "reporte.html")

        assert (run_dir / "weights" / "best.pt").exists()
        assert "0.891" in html           # mAP50 last row
        assert "0.543" in html           # mAP50-95 last row
        assert "0.81/0.76" in html       # P / R last row
        assert html.count("data:image/png;base64,") == 3
