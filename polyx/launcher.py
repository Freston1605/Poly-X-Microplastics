"""Compatibilidad: punto de entrada historico ``python -m polyx.launcher``.

Este modulo existe solo para que el entry point historico y los scripts de
instalacion externos que invocan ``python -m polyx.launcher`` (Poly-X.vbs,
iniciar_polyx.bat, Lanzar_macOS.command, crear_icono_macOS.command,
lanzar_polyx.py y el bundle de PyInstaller) sigan funcionando sin editarlos.

Todo el trabajo Qt vive dentro de :mod:`polyx.gui.launcher`; este shim no
importa Qt directamente (el import de abajo trae el modulo real, que es quien
habla con la interfaz grafica).
"""
from polyx.gui.launcher import LauncherWindow, main  # noqa: F401

if __name__ == "__main__":
    main()
