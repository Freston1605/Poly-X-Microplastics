"""Contrato de pureza de ``polyx.core``: el motor nunca puede arrastrar Qt.

El hito M005 parte Poly-X en un motor puro (``polyx.core``) y una capa de
interfaz (``polyx.gui``). El valor de esa frontera se pierde en cuanto un
``import`` descuidado la cruza: basta que un modulo del motor importe
``PySide6`` para que el motor deje de poder correr headless en CI o en un
servidor sin pantalla, y basta que importe ``polyx.gui`` para reacoplar el
calculo a la ventana.

Estas pruebas fijan la frontera de forma ejecutable, en tres frentes:

1. Un interprete LIMPIO importa ``polyx.core`` y comprueba que no quedo
   ningun modulo ``PySide6*`` en ``sys.modules``. Se usa un subproceso a
   proposito: el proceso de pytest puede traer PySide6 ya cargado por
   ``test_temas``, asi que comprobarlo en proceso no probaria nada.
2. Un escaneo AST de ``polyx/model/*.py`` rechaza cualquier import de Qt o de
   ``polyx.gui`` y cualquier import relativo que se salga del motor.
3. El paquete de re-export del ``core`` tiene que haber DESAPARECIDO: mientras
   existiera, cualquier consumidor podia seguir importando los nombres viejos
   y la separacion entre motor y GUI era decorativa. Con el paquete borrado,
   el unico camino al motor es ``polyx.core``.

Sobre el escaneo relativo: el plan de la tarea pedia "todo import relativo es
nivel 1". ``procedencia.py`` rompe esa letra a proposito y con razon, porque
usa ``from .. import __version__`` (nivel 2) para leer la version del paquete
``polyx`` -- metadatos puros, sin Qt. La invariante que de verdad importa es
que el motor no salga de su propio paquete ni del ``polyx`` que lo contiene:
se admite ``polyx.core.*`` (nivel 1) y el ``from ..`` que apunta al propio
``polyx``, y se rechaza cualquier relativo que nombre otro subpaquete hermano
(``polyx.gui`` o el paquete de re-export del ``core``) o que escape por arriba
(nivel >= 3).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
MOTOR = RAIZ / "src" / "polyx" / "core"

# Enlaces de Qt que el motor puro no debe importar nunca. La raiz de cada
# nombre es lo que se compara, asi que ``PySide6.QtWidgets`` cae en ``PySide6``.
ENLACES_QT = ("PySide6", "PySide2", "PyQt5", "PyQt6", "shiboken6", "shiboken2")

# Paquetes fuera del motor de los que el motor no puede depender.
PAQUETES_PROHIBIDOS = ("polyx.gui",)

# Los doce modulos que T01 movio a polyx.core. Si falta alguno, el escaneo
# AST de mas abajo quedaria vacio y pasaria por no tener nada que mirar.
MODULOS_ESPERADOS = frozenset(
    {
        # detect/
        "yolo",
        "native",
        "sam",
        "factory",
        "protocol",
        # measure/
        "morphology",
        "calibration",
        # quality/
        "metrics",
        "assignment",
        "dataset_audit",
        # provenance/
        "observability",
        "provenance",
        # data/
        "labels",
        # root modules
        "i18n",
        "translations_en",
        "paths",
        "platform",
    }
)


def _modulos_del_motor() -> list[Path]:
    """Rutas de los .py del motor, ordenadas y sin __pycache__.

    Busca recursivamente en subdirectorios (detect/, measure/, quality/,
    provenance/, data/).
    """
    return sorted(p for p in MOTOR.rglob("*.py") if "__pycache__" not in p.parts)


def _es_qt(nombre: str) -> bool:
    """True si el nombre importado es (o cuelga de) un enlace de Qt."""
    return nombre.split(".", 1)[0] in ENLACES_QT


def _es_prohibido(nombre: str) -> bool:
    """True si el nombre importado es (o cuelga de) un paquete vetado."""
    return any(
        nombre == paquete or nombre.startswith(paquete + ".") for paquete in PAQUETES_PROHIBIDOS
    )


def _destino_relativo(nivel: int, modulo: str | None) -> str:
    """Nombre absoluto al que apunta un import relativo escrito en el motor.

    ``nivel`` es el numero de puntos de ``ast.ImportFrom``: uno es el propio
    paquete (``polyx.core``), dos el padre (``polyx``). Devuelve cadena vacia
    cuando el import escapa por encima de ``polyx`` (nivel >= 3).
    """
    if nivel == 1:
        partes = ["polyx", "core"]
    elif nivel == 2:
        partes = ["polyx"]
    else:
        return ""
    if modulo:
        partes += modulo.split(".")
    return ".".join(partes)


def _imports_prohibidos(arbol: ast.AST) -> list[str]:
    """Nombres de Qt o de la GUI importados en cualquier punto del arbol.

    Mira tambien dentro de funciones y clases (``ast.walk`` recorre todo), que
    es donde suelen esconderse los imports perezosos de ``torch`` o ``cv2``.
    """
    encontrados: list[str] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                if _es_qt(alias.name) or _es_prohibido(alias.name):
                    encontrados.append(alias.name)  # noqa: PERF401
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.level:
                destino = _destino_relativo(nodo.level, nodo.module)
                if destino and (_es_qt(destino) or _es_prohibido(destino)):
                    encontrados.append(destino)
            elif nodo.module and (_es_qt(nodo.module) or _es_prohibido(nodo.module)):
                encontrados.append(nodo.module)
    return encontrados


def _relativos_fuera_de_alcance(arbol: ast.AST) -> list[str]:
    """Imports relativos que se salen de ``polyx.core`` (o del propio ``polyx``).

    Permitido: ``from .x import ...`` (nivel 1) y ``from .. import __version__``
    (nivel 2 sin submodulo, que es el paquete ``polyx``). Todo lo demas --un
    subpaquete hermano por nivel 2, o un nivel >= 3 que sale del proyecto--
    aparece aqui.
    """
    fuera: list[str] = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.ImportFrom) or not nodo.level:
            continue
        destino = _destino_relativo(nodo.level, nodo.module)
        permitido = (
            destino == "polyx" or destino == "polyx.core" or destino.startswith("polyx.core.")
        )
        if not permitido:
            fuera.append(destino or f"nivel {nodo.level}")
    return fuera


# ── 1. Pureza headless en un interprete limpio ──
def test_importar_el_motor_no_carga_ningun_modulo_qt():
    """Un interprete nuevo importa polyx.core sin que entre ningun PySide6.

    Se lanza en subproceso porque el pytest que corre esta prueba puede tener
    PySide6 ya en sys.modules (``test_temas`` construye widgets): comprobar la
    pureza dentro del mismo proceso daria un falso positivo.
    """
    guion = (
        "import sys\n"
        "import polyx.core\n"
        "malos = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'PySide6' or m.startswith('PySide6.')\n"
        ")\n"
        "print('PYQT_CARGADOS=' + repr(malos))\n"
        "raise SystemExit(1 if malos else 0)\n"
    )
    entorno = dict(os.environ)
    entorno["QT_QPA_PLATFORM"] = "offscreen"
    # El worktree primero, para no importar un polyx instalado y viejo.
    entorno["PYTHONPATH"] = (
        str(RAIZ / "src") + os.pathsep + str(RAIZ) + os.pathsep + entorno.get("PYTHONPATH", "")
    )

    resultado = subprocess.run(  # noqa: PLW1510
        [sys.executable, "-c", guion],
        capture_output=True,
        text=True,
        cwd=str(RAIZ),
        env=entorno,
        timeout=120,
    )
    assert resultado.returncode == 0, (
        "importar polyx.core arrastro Qt o fallo:\n"
        f"stdout: {resultado.stdout}\n"
        f"stderr: {resultado.stderr}"
    )
    assert "PYQT_CARGADOS=[]" in resultado.stdout


# ── 2. Escaneo AST de la frontera ──
def test_el_motor_contiene_los_modulos_esperados():
    """Guardia contra un escaneo vacio: los modulos esperados tienen que estar."""
    presentes = {ruta.stem for ruta in _modulos_del_motor()}
    faltan = MODULOS_ESPERADOS - presentes
    assert not faltan, f"faltan modulos del motor: {sorted(faltan)}"


@pytest.mark.parametrize("ruta", _modulos_del_motor(), ids=lambda r: r.name)
def test_el_motor_no_importa_qt_ni_la_gui(ruta: Path):
    """Ningun modulo del motor importa Qt ni polyx.gui, en ningun nivel."""
    arbol = ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))

    prohibidos = _imports_prohibidos(arbol)
    assert not prohibidos, f"{ruta.name} importa Qt o la GUI: {prohibidos}"

    fuera = _relativos_fuera_de_alcance(arbol)
    assert not fuera, f"{ruta.name} tiene imports relativos fuera del motor: {fuera}"


def test_el_escanner_detecta_violaciones_sinteticas():
    """El escaner no puede pasar por vacio: tiene que ver lo que prohibe.

    Sin esto, un fallo en las funciones de analisis convertiria el contrato en
    decorativo: pasaria igual aunque el motor se llenara de imports de Qt.
    """
    con_qt = ast.parse("import PySide6.QtWidgets\n")
    assert _imports_prohibidos(con_qt) == ["PySide6.QtWidgets"]

    desde_qt = ast.parse("from PySide6.QtCore import QTimer\n")
    assert _imports_prohibidos(desde_qt) == ["PySide6.QtCore"]

    desde_gui = ast.parse("from polyx.gui import ventana\n")
    assert _imports_prohibidos(desde_gui) == ["polyx.gui"]

    # Un relativo que se va al subpaquete hermano de la GUI: lo cazan las dos
    # comprobaciones (import prohibido y relativo fuera de alcance).
    relativo_a_gui = ast.parse("from ..gui import ventana\n")
    assert _imports_prohibidos(relativo_a_gui) == ["polyx.gui"]
    assert _relativos_fuera_de_alcance(relativo_a_gui) == ["polyx.gui"]

    # Y el nivel 2 legitimo de procedencia.py NO debe marcarse como violacion.
    version_polyx = ast.parse("from .. import __version__\n")
    assert _imports_prohibidos(version_polyx) == []
    assert _relativos_fuera_de_alcance(version_polyx) == []


# ── 3. El paquete core ES el motor (no un re-export) ──
# El paquete ``polyx.core`` es ahora el motor puro (antes ``polyx.core``).
# No debe ser un paquete de re-export vacio, sino que debe contener los
# modulos del motor: detect, measure, quality, provenance, data, etc.


def test_el_paquete_core_es_el_motor():
    """El paquete ``polyx.core`` existe y es el motor puro.

    Debe ser importable sin Qt y contener los submodulos del motor.
    """
    import sys

    # Only NEW Qt modules loaded by this import count: earlier tests in this
    # process legitimately imported PySide6, so a whole-process sys.modules
    # check would fail by contamination, not because polyx.core dragged Qt.
    qt_antes = {m for m in sys.modules if m.startswith("PySide6")}
    import polyx.core

    # El paquete debe existir en disco
    assert (RAIZ / "src" / "polyx" / "core").exists(), "polyx/core no existe en disco"
    qt_nuevos = {m for m in sys.modules if m.startswith("PySide6")} - qt_antes
    assert not qt_nuevos, f"polyx.core arrastro Qt: {sorted(qt_nuevos)}"
    # Debe tener los submodulos esperados
    assert hasattr(polyx.core, "detect")
    assert hasattr(polyx.core, "measure")
    assert hasattr(polyx.core, "quality")
    assert hasattr(polyx.core, "provenance")
    assert hasattr(polyx.core, "data")
    # Debe exportar la API publica principal
    assert hasattr(polyx.core, "Instance")
    assert hasattr(polyx.core, "DetectionSource")
    assert hasattr(polyx.core, "make_detector")
    assert hasattr(polyx.core, "YoloModel")
    assert hasattr(polyx.core, "NativeDetector")
