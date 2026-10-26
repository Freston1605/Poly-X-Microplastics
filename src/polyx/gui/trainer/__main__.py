"""Enables `python -m polyx.gui.trainer`."""
import sys
from PySide6.QtWidgets import QApplication

from polyx.gui import marca
from .window import TrainerWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Before the first show(): the taskbar button is created
    # with whatever identity exists at that moment.
    marca.identify(app, "entrenador")
    w = TrainerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
