"""Enables ``python -m polyx.gui``: runs the launcher.

This is the modern entry point of the GUI package. The historic entry
point ``python -m polyx.launcher`` keeps working through the Qt-free
shim ``polyx/launcher.py``.
"""
from polyx.gui.qt import require_qt


def main() -> None:
    require_qt()
    # Imported lazily so the readable gate above fires before Qt modules load.
    from polyx.gui.launcher import main as _launcher_main
    _launcher_main()


if __name__ == "__main__":
    main()