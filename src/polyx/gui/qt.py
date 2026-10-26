"""Guarded Qt import for the Poly-X GUI layer.

PySide6 lives behind the ``[gui]`` packaging extra so headless consumers
(data pipelines, CI, library imports of ``polyx.core``) do not pay for Qt.
This single helper is where the import errors are converted into a
readable message; the Qt-touching modules keep importing PySide6 directly.
"""
from __future__ import annotations


from importlib.util import find_spec
import sys
from types import ModuleType, SimpleNamespace
from typing import Tuple

_INSTALL_HINT = (
    "PySide6 is required to run the Poly-X GUI. Install it with:\n"
    "    pip install polyx-microplastics[gui]\n"
    "(or `pip install PySide6==6.7.2` in a development checkout)."
)


def qt_available() -> bool:
    """Is PySide6 importable in this environment?"""
    try:
        return find_spec("PySide6") is not None
    except (ImportError, ValueError, OSError, AttributeError, KeyError):
        return False


def _raise_missing() -> None:
    raise ImportError(_INSTALL_HINT) from None


def _bind_gui_modules() -> Tuple[ModuleType, ModuleType]:
    """Import QtCore/QtGui, raising a readable ImportError when absent."""
    try:
        import PySide6.QtCore as qt_core
        import PySide6.QtGui as qt_gui
    except (ImportError, OSError) as exc:
        raise ImportError(f"{_INSTALL_HINT}\n{exc}") from exc
    return qt_core, qt_gui


def require_qt() -> None:
    """Fail fast with a readable message when PySide6 is not installed."""
    if not qt_available():
        _raise_missing()


__all__ = ["qt_available", "require_qt", "_bind_gui_modules", "_INSTALL_HINT"]
