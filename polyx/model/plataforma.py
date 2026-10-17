"""Lo que cambia entre Windows, macOS y Linux, en un solo sitio.

Poly-X nacio en Windows y daba eso por sentado en once puntos del codigo:
``os.startfile`` para abrir una carpeta o un informe, ``cmd /c start`` para
lanzar el actualizador, y ``device="0"`` (CUDA) como dispositivo por defecto.
Ninguna de las tres cosas existe en un Mac.

Aqui se concentran esas diferencias para que el resto del programa no tenga
que saber en que sistema corre. La alternativa -- un ``if sys.platform`` en
cada sitio -- es la forma segura de que dentro de un mes uno de ellos se
quede sin actualizar y falle solo en una plataforma.

SOBRE APPLE SILICON
-------------------
Los Mac con chip M1 en adelante traen **MPS** (Metal Performance Shaders),
que es la via de PyTorch para usar la GPU integrada. No es CUDA y no se
llama igual, pero para inferencia YOLO da una aceleracion real frente a CPU.
Un Mac Intel no tiene MPS y se queda en CPU.

Por eso el dispositivo por defecto no puede ser una constante: se decide
mirando que hay disponible, en este orden -- CUDA, MPS, CPU.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── Identificacion del sistema ──
# Se calcula una vez: sys.platform no cambia a mitad de ejecucion, y
# consultarlo en cada llamada solo hace el codigo mas ruidoso.
ES_WINDOWS = sys.platform.startswith("win")
ES_MAC = sys.platform == "darwin"
ES_LINUX = sys.platform.startswith("linux")


def nombre_sistema() -> str:
    """Nombre legible del sistema, para el informe y los diagnosticos."""
    if ES_MAC:
        # platform.mac_ver() da "14.5"; platform.release() daria el numero de
        # kernel de Darwin (23.5.0), que no le dice nada a nadie.
        version = platform.mac_ver()[0] or platform.release()
        return f"macOS {version} ({arquitectura_mac() or platform.machine()})"
    return f"{platform.system()} {platform.release()}"


def arquitectura_mac() -> str:
    """'Apple Silicon', 'Intel' o '' si no es un Mac.

    Importa porque decide si hay aceleracion por GPU disponible: MPS existe
    en Apple Silicon y no en Intel.
    """
    if not ES_MAC:
        return ""
    maquina = platform.machine().lower()
    if maquina in ("arm64", "aarch64"):
        return "Apple Silicon"
    return "Intel"


def abrir_en_el_sistema(ruta: str | Path) -> bool:
    """Abre un archivo o carpeta con la aplicacion que le corresponda.

    Equivalente multiplataforma de ``os.startfile``, que solo existe en
    Windows. Se usa para abrir la carpeta de un run, un informe HTML en el
    navegador o una imagen anotada en el visor del sistema.

    **Nunca lanza excepcion.** Devuelve si lo consiguio. Que no se abra una
    carpeta es una molestia; que tumbe el analisis recien terminado, no.

    En macOS y Linux se lanza el proceso y no se espera: ``open`` retorna
    enseguida, pero ``xdg-open`` en algunos escritorios se queda vivo
    mientras la aplicacion este abierta, y esperarlo congelaria la interfaz.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        return False
    try:
        if ES_WINDOWS:
            os.startfile(str(ruta))            # type: ignore[attr-defined]
            return True
        comando = ["open"] if ES_MAC else ["xdg-open"]
        subprocess.Popen(comando + [str(ruta)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, AttributeError, subprocess.SubprocessError):
        return False


# ── Dispositivo de computo ──

DISPOSITIVO_CUDA = "0"
DISPOSITIVO_MPS = "mps"
DISPOSITIVO_CPU = "cpu"


def dispositivo_disponible() -> str:
    """El mejor dispositivo que este equipo puede usar: '0', 'mps' o 'cpu'.

    Se prueba en orden de rendimiento y NO se cachea a nivel de modulo: una
    GPU puede quedar ocupada por otro proceso entre el arranque del programa
    y el momento de inferir, y aqui interesa el estado de ahora.

    ``torch.backends.mps.is_built()`` ademas de ``is_available()``: un torch
    instalado desde una rueda generica puede venir sin MPS compilado aunque
    el Mac lo soporte, y entonces ``is_available()`` da False sin explicar
    por que.
    """
    try:
        import torch
    except Exception:
        return DISPOSITIVO_CPU
    try:
        if torch.cuda.is_available():
            return DISPOSITIVO_CUDA
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return DISPOSITIVO_MPS
    except Exception:
        pass
    return DISPOSITIVO_CPU


def etiqueta_dispositivos() -> str:
    """Texto para el rotulo del selector de dispositivo en la interfaz.

    En Windows/Linux con NVIDIA se escribe el indice de la tarjeta; en un Mac
    ese '0' no significa nada y ofrecerlo solo lleva a error.
    """
    if ES_MAC:
        return "Dispositivo (mps/cpu):" if arquitectura_mac() == "Apple Silicon" \
            else "Dispositivo (cpu):"
    return "Dispositivo (0/cpu):"


def descripcion_dispositivo(device: str) -> str:
    """Frase corta que explica que es ese dispositivo, para el informe."""
    d = (device or "").strip().lower()
    if d in ("mps",):
        return "GPU integrada del Mac (Metal Performance Shaders)"
    if d in ("cpu", "-1"):
        return "procesador (CPU)"
    if d.isdigit():
        return f"GPU NVIDIA #{d} (CUDA)"
    return device or "—"


# ── Actualizador ──
def fijar_identidad_app(app_id: str) -> bool:
    """Declara ante el sistema que este proceso es Poly-X y no Python.

    Windows agrupa los botones de la barra de tareas por **AppUserModelID**, no
    por el ejecutable. Un proceso lanzado como ``pythonw.exe -m polyx.gui.detector``
    que no declara el suyo hereda el de la instalacion de Python, y el shell le
    pone el icono de Python aunque la ventana tenga el suyo propio. Tambien
    hace que anclar la ventana a la barra ancle Python.

    Hay que llamarla **antes de que se cree la primera ventana**: el boton de
    la barra se construye con el identificador que hubiera en ese momento, y
    cambiarlo despues no surte efecto hasta reabrir.

    En macOS y Linux no existe el concepto y no hay nada que hacer: alli el
    icono de la ventana basta. Devuelve si llego a fijarse.
    """
    if not ES_WINDOWS:
        return False
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except (AttributeError, OSError):
        # Windows Server sin shell32 completo, o una version antigua. No es
        # motivo para no arrancar: solo se pierde el icono de la barra.
        return False


# El script que aplica la actualizacion es distinto en cada sistema porque
# tiene que hablar con el gestor de paquetes y el shell de cada uno. El
# nombre se resuelve aqui para que updates.py y launcher.py no lo repitan.

def nombre_actualizador() -> str:
    if ES_WINDOWS:
        return "actualizar.bat"
    if ES_MAC:
        return "actualizar_macOS.command"
    return "actualizar_linux.sh"


def lanzar_actualizador(ruta: str | Path) -> bool:
    """Arranca el actualizador en una ventana propia y devuelve el control.

    En una ventana aparte a proposito: el actualizador sobrescribe los
    archivos del programa, incluido el que lo lanzo, asi que Poly-X tiene que
    poder cerrarse mientras aquel sigue.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        return False
    try:
        if ES_WINDOWS:
            # 'start' con titulo vacio: sin ese "" el primer argumento
            # entrecomillado se toma como titulo de la ventana y el script
            # no llega a ejecutarse.
            subprocess.Popen(["cmd", "/c", "start", "", str(ruta)],
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        if ES_MAC:
            # Terminal.app para que el usuario vea el progreso de la descarga.
            # Ejecutarlo en segundo plano dejaria la aplicacion aparentemente
            # colgada durante varios minutos sin decir nada.
            subprocess.Popen(["open", "-a", "Terminal", str(ruta)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        subprocess.Popen(["/bin/bash", str(ruta)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def python_del_entorno(raiz: Optional[Path] = None) -> Optional[Path]:
    """Ruta al python del .venv, que cuelga de una carpeta distinta segun el SO.

    Windows lo pone en ``.venv/Scripts/python.exe`` y el resto del mundo en
    ``.venv/bin/python``. Devuelve None si no hay entorno creado todavia.
    """
    from .paths import ROOT
    base = Path(raiz) if raiz else ROOT
    candidatos = [base / ".venv" / "Scripts" / "python.exe",
                  base / ".venv" / "bin" / "python"]
    for c in candidatos:
        if c.is_file():
            return c
    return None
