"""Permite `python -m polyx.gui.detector`."""
import sys
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtWidgets import QApplication

from polyx.gui import marca
from .window import DetectorWindow


def main():
    # Requisito de QtWebEngine (exportación a PDF del reporte): debe fijarse
    # antes de construir el QApplication.
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(sys.argv)
    # Antes del primer show(): el boton de la barra de tareas se crea
    # con la identidad que haya en ese momento.
    marca.identificar(app, "detector")
    w = DetectorWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
