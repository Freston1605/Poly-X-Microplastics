"""Shim Qt-free del entry point historico.

Arranca el launcher sin consola. Equivalente a `python -m polyx.launcher`.
"""
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.gui.launcher import main  # noqa: E402

if __name__ == "__main__":
    main()
