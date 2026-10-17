"""Identifica con que codigo, que versiones y que pesos se produjo un resultado.

Una carpeta de run guarda los parametros de la corrida, pero eso no basta para
rehacerla: dos ``best.pt`` distintos se llaman igual, ultralytics cambia el
calculo del AP entre versiones y el propio Poly-X cambia entre commits. Sin
dejarlo escrito, dentro de un ano no hay forma de demostrar que pesos
produjeron una tabla del paper.

Todo lo de aqui es informativo y no debe tumbar una corrida: si git no esta
instalado o el .pt esta en un disco que se desconecto, se devuelve el hueco y
se sigue.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from .paths import ROOT


def sha256_archivo(ruta: str | Path, _bloque: int = 1 << 20) -> str:
    """Huella del archivo, leido por bloques.

    Por bloques y no de una vez porque un .pt de YOLO medium pesa 52 MB y hay
    hasta tres cargados a la vez. Cadena vacia si no se puede leer.
    """
    try:
        h = hashlib.sha256()
        with open(ruta, "rb") as f:
            for trozo in iter(lambda: f.read(_bloque), b""):
                h.update(trozo)
        return h.hexdigest()
    except OSError:
        return ""


def _version_de(modulo: str) -> str:
    """Version de un paquete ya importado, sin forzar su importacion.

    Se mira ``sys.modules`` a proposito: preguntar por torch aqui lo cargaria
    entero (segundos y VRAM) en una corrida que quiza ni lo use.
    """
    mod = sys.modules.get(modulo)
    return str(getattr(mod, "__version__", "")) if mod is not None else ""


def versiones() -> Dict[str, str]:
    """Version de Poly-X y de las librerias que deciden el resultado numerico."""
    from .. import __version__ as version_polyx

    return {
        "polyx": version_polyx,
        "python": platform.python_version(),
        "torch": _version_de("torch"),
        "ultralytics": _version_de("ultralytics"),
        "opencv": _version_de("cv2"),
        "numpy": _version_de("numpy"),
        "so": f"{platform.system()} {platform.release()}",
    }


def commit_git(raiz: Optional[Path] = None) -> Dict[str, str]:
    """Commit actual del repositorio y si el arbol tenia cambios sin confirmar.

    ``sucio`` importa tanto como el hash: un run hecho sobre codigo modificado y
    sin confirmar no es reproducible aunque el commit quede anotado.
    """
    raiz = Path(raiz) if raiz is not None else ROOT
    salida = {"commit": "", "sucio": ""}

    def _git(*args: str) -> Optional[str]:
        try:
            r = subprocess.run(
                ["git", "-C", str(raiz), *args],
                capture_output=True, text=True, timeout=5,
                # Sin consola negra al correr bajo pythonw.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return salida
    salida["commit"] = commit
    estado = _git("status", "--porcelain")
    salida["sucio"] = "si" if estado else "no"
    return salida


def procedencia(raiz: Optional[Path] = None) -> Dict[str, object]:
    """Bloque completo para volcar en el resumen de un run."""
    datos: Dict[str, object] = {"versiones": versiones()}
    datos.update(commit_git(raiz))
    return datos
