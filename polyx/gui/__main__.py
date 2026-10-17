"""Permite ``python -m polyx.gui``: ejecuta el launcher.

Es el punto de entrada moderno del paquete GUI. El punto de entrada historico
``python -m polyx.launcher`` sigue funcionando a traves del shim Qt-free
``polyx/launcher.py``.
"""
from polyx.gui.launcher import main

if __name__ == "__main__":
    main()
