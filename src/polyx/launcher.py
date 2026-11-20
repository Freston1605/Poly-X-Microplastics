"""Compatibility: historic entry point ``python -m polyx.launcher``.

This module exists only so that notes, shortcuts and muscle memory that
invoke ``python -m polyx.launcher`` keep working while ``python -m
polyx.gui`` is the current entry point.

All the Qt work lives inside :mod:`polyx.gui.launcher`; this shim does
not import Qt directly (the import below brings in the real module,
which is the one that talks to the graphical interface).
"""

from polyx.gui.launcher import LauncherWindow, main  # noqa: F401

if __name__ == "__main__":
    main()
