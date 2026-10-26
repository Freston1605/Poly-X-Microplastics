"""Conversion of a self-contained HTML report to PDF.

Uses the Chromium engine of QtWebEngine (`QWebEnginePage.printToPdf`),
which does respect the report's modern CSS (grid, flexbox, base64
images). The Poly-X HTML is self-contained (all images are embedded in
base64), so the resulting PDF is too: it can be sent to someone else
without the figures breaking.

It runs synchronously through a local event loop, so it must be called
from the GUI thread with a QApplication already created.
"""

from __future__ import annotations

from pathlib import Path


def is_available() -> bool:
    """Is QtWebEngine available to export to PDF?"""
    import importlib.util

    return importlib.util.find_spec("PySide6.QtWebEngineWidgets") is not None


def html_to_pdf(html_path: Path, pdf_path: Path, timeout_ms: int = 60000) -> bool:
    """Renders `html_path` to `pdf_path`. Returns True on success.

    Requires a running QApplication (GUI thread). The local event loop
    blocks until the printing finishes or `timeout_ms` is exhausted.
    """
    from PySide6.QtCore import QEventLoop, QMarginsF, QTimer, QUrl
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtWebEngineWidgets import QWebEngineView

    html_path = Path(html_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    view = QWebEngineView()  # not shown; offscreen render
    loop = QEventLoop()
    state = {"ok": False}

    def _on_pdf_finished(file_path: str, ok: bool):
        state["ok"] = bool(ok)
        loop.quit()

    def _print():
        layout = QPageLayout(
            QPageSize(QPageSize.A4),
            QPageLayout.Portrait,
            QMarginsF(8, 8, 8, 8),  # margins in mm
        )
        view.page().printToPdf(str(pdf_path), layout)

    def _on_load_finished(ok: bool):
        if not ok:
            state["ok"] = False
            loop.quit()
            return
        # A little breathing room so the layout finishes before printing
        QTimer.singleShot(350, _print)

    view.page().pdfPrintingFinished.connect(_on_pdf_finished)
    view.loadFinished.connect(_on_load_finished)
    view.load(QUrl.fromLocalFile(str(html_path.resolve())))

    # Anti-hang guard
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()

    view.deleteLater()
    return state["ok"] and pdf_path.exists()
