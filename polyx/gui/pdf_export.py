"""Conversión de un reporte HTML autocontenido a PDF.

Usa el motor Chromium de QtWebEngine (`QWebEnginePage.printToPdf`), que sí
respeta el CSS moderno del reporte (grid, flexbox, imágenes base64). El HTML
de Poly-X es autocontenido (todas las imágenes van embebidas en base64), así
que el PDF resultante también lo es: se puede enviar a otra persona sin que
las figuras se rompan.

Se ejecuta de forma síncrona mediante un bucle de eventos local, por lo que
debe llamarse desde el hilo de la GUI con un QApplication ya creado.
"""
from __future__ import annotations
from pathlib import Path


def is_available() -> bool:
    """¿Está disponible QtWebEngine para exportar a PDF?"""
    import importlib.util
    return importlib.util.find_spec("PySide6.QtWebEngineWidgets") is not None


def html_to_pdf(html_path: Path, pdf_path: Path, timeout_ms: int = 60000) -> bool:
    """Renderiza `html_path` a `pdf_path`. Devuelve True si tuvo éxito.

    Requiere QApplication en ejecución (hilo de la GUI). El bucle de eventos
    local bloquea hasta que termina la impresión o se agota `timeout_ms`.
    """
    from PySide6.QtCore import QUrl, QEventLoop, QTimer, QMarginsF
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtWebEngineWidgets import QWebEngineView

    html_path = Path(html_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    view = QWebEngineView()          # no se muestra; render offscreen
    loop = QEventLoop()
    state = {"ok": False}

    def _on_pdf_finished(file_path: str, ok: bool):
        state["ok"] = bool(ok)
        loop.quit()

    def _print():
        layout = QPageLayout(
            QPageSize(QPageSize.A4),
            QPageLayout.Portrait,
            QMarginsF(8, 8, 8, 8),   # márgenes en mm
        )
        view.page().printToPdf(str(pdf_path), layout)

    def _on_load_finished(ok: bool):
        if not ok:
            state["ok"] = False
            loop.quit()
            return
        # Pequeño respiro para que termine el layout antes de imprimir
        QTimer.singleShot(350, _print)

    view.page().pdfPrintingFinished.connect(_on_pdf_finished)
    view.loadFinished.connect(_on_load_finished)
    view.load(QUrl.fromLocalFile(str(html_path.resolve())))

    # Guarda anti-cuelgue
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()

    view.deleteLater()
    return state["ok"] and pdf_path.exists()
