"""Punto de entrada: python -m polyx.gui.visor"""
import sys
from PySide6.QtWidgets import QApplication
from polyx.gui import marca
from .window import VisorWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Antes del primer show(): el boton de la barra de tareas se crea
    # con la identidad que haya en ese momento.
    marca.identificar(app, "visor")
    w = VisorWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
