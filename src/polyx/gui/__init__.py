"""Poly-X GUI layer — every module that touches PySide6.

Importing ``polyx.gui`` itself never imports Qt: submodules bind PySide6
individually, and :mod:`polyx.gui.qt` converts a missing install into a
readable message (``pip install polyx-microplastics[gui]``).

Headless consumers should stay in ``polyx.core``, which does not import Qt.
"""
from .qt import qt_available, require_qt

__all__ = [
    "qt_available",
    "require_qt",
]
