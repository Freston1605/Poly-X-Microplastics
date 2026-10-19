"""Punto de entrada de Poly-X como archivo suelto.

Existe para PyInstaller: al empaquetar hace falta apuntar a un *archivo*, y
``python -m polyx.launcher`` no le sirve porque eso es un modulo. Fuera de
ese caso no aporta nada; el modo normal de arrancar sigue siendo el modulo.

    python -m polyx.launcher      # uso normal
    python lanzar_polyx.py        # equivalente, y lo que empaqueta el .app
"""
from polyx.gui.launcher import main

if __name__ == "__main__":
    main()
