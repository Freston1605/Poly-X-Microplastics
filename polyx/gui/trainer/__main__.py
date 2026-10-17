"""Permite `python -m polyx.gui.trainer`."""
import sys
from PySide6.QtWidgets import QApplication

from polyx.gui import marca
from .window import TrainerWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Antes del primer show(): el boton de la barra de tareas se crea
    # con la identidad que haya en ese momento.
    marca.identificar(app, "entrenador")
    w = TrainerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
