"""Compatibility: historic entry point ``python -m polyx.launcher``.

This module exists only so that the historic entry point and the
external installation scripts that invoke ``python -m polyx.launcher``
(Poly-X.vbs, iniciar_polyx.bat, Lanzar_macOS.command,
crear_icono_macOS.command, lanzar_polyx.py and the PyInstaller bundle)
keep working without editing them.

All the Qt work lives inside :mod:`polyx.gui.launcher`; this shim does
not import Qt directly (the import below brings in the real module,
which is the one that talks to the graphical interface).
"""

from polyx.gui.launcher import LauncherWindow, main  # noqa: F401

if __name__ == "__main__":
    main()
