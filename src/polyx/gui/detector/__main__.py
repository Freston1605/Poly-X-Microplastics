"""Enables `python -m polyx.gui.detector`."""
import sys
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtWidgets import QApplication

from polyx.gui import marca
from .window import DetectorWindow


def main():
    # QtWebEngine requirement (PDF export of the report): it must be set
    # before building the QApplication.
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(sys.argv)
    # Before the first show(): the taskbar button is created
    # with whatever identity exists at that moment.
    marca.identify(app, "detector")
    w = DetectorWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
