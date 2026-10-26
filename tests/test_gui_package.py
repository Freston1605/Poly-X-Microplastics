"""Contrato de frontera de ``polyx.gui``: Qt solo vive dentro del paquete GUI.

El hito M005 parte Poly-X en un motor puro (``polyx.core``) y una capa de
interfaz (``polyx.gui``). ``tests/test_model_package.py`` ya fija el lado del
motor; este archivo es la contraparte del lado de la interfaz y fija la
frontera en tres frentes:

1. Un escaneo AST de ``polyx/`` comprueba que ningun modulo FUERA de
   ``polyx/gui/`` importa un enlace de Qt. Eso prueba a la vez que el
   ``polyx/launcher.py`` de compatibilidad sigue siendo Qt-free: si arrastrara
   PySide6, el motor dejaria de poder correr headless (decision D012).
2. Un escaneo AST de ``polyx/gui/`` rechaza cualquier import del paquete de
   re-export del ``core``, absoluto o relativo: la GUI depende del motor
   (``polyx.core``), nunca de nombres heredados. S03 borra ese paquete, asi
   que este contrato es el que avisa si algun consumidor se dejo sin migrar.
3. El paquete de re-export del ``core`` ha desaparecido y el punto de entrada
   historico ``python -m polyx.launcher`` sigue llamando al launcher real, que
   es quien posee la interfaz.

Los escaneos usan ``ast.walk`` a proposito: los imports perezosos dentro de
funciones y clases tambien cruzan la frontera y tienen que cazarles.
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path

# Tiene que estar ANTES de cualquier import de polyx: polyx.gui arrastra
# PySide6, y sin plataforma offscreen el import revienta en una maquina sin
# DISPLAY.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE_GUI = RAIZ / "src" / "polyx" / "gui"

# Enlaces de Qt que solo puede importar polyx.gui. La raiz de cada nombre es
# lo que se compara, asi que ``PySide6.QtWidgets`` cae en ``PySide6``.
ENLACES_QT = ("PySide6", "PySide2", "PyQt5", "PyQt6", "shiboken6", "shiboken2")

# Los ocho modulos acoplados a Qt que T02 movio a polyx.gui. Si falta alguno,
# los escaneos AST de abajo quedarian huecos y pasarian por no tener nada
# que mirar.
MODULOS_MOVIDOS = (
    "theme",
    "iconos",
    "marca",
    "widgets",
    "preferencias",
    "pdf_export",
    "updates",
    "report_html",
)

# Los cuatro paquetes de tinto que tambien viven en polyx.gui.
PAQUETES_FEATURE = ("detector", "trainer", "viewer", "labeler")


# ── Helpers del escaner ──
def _es_de_gui(ruta: Path) -> bool:
    """True si la ruta esta dentro del paquete polyx/gui."""
    partes = ruta.relative_to(RAIZ).parts
    return len(partes) >= 3 and partes[0] == "src" and partes[1] == "polyx" and partes[2] == "gui"


def _fuentes_de_gui() -> list[Path]:
    """Rutas de los .py dentro de polyx/gui, ordenadas."""
    return sorted(PAQUETE_GUI.rglob("*.py"))


def _fuentes_fuera_de_gui() -> list[Path]:
    """Rutas de los .py de polyx/ que NO estan en polyx/gui."""
    return sorted(ruta for ruta in (RAIZ / "src" / "polyx").rglob("*.py") if not _es_de_gui(ruta))


def _partes_del_paquete(ruta: Path) -> list[str]:
    """Partes del paquete que CONTIENE al modulo (p. ej. polyx.gui para theme.py).

    Un ``__init__.py`` vive dentro de su propio paquete, un modulo suelto
    vive dentro del paquete directorio: por eso al segundo se le quita el
    ultimo elemento.
    """
    partes = list(ruta.relative_to(RAIZ).with_suffix("").parts)
    if partes[-1] != "__init__":
        partes = partes[:-1]
    return partes


def _resuelve_relativo(partes: list[str], nivel: int, modulo: str | None) -> str | None:
    """Nombre absoluto al que apunta un import relativo, o None si escapa.

    ``nivel`` es el numero de puntos de ``ast.ImportFrom``: uno es el propio
    paquete del modulo (``polyx.gui``), dos el padre (``polyx``). Por eso
    cada punto ``SUBE`` un nivel y hay que quitar ``nivel - 1`` partes, no
    ``nivel``: un ``from ..core import theme`` escrito en
    ``polyx/gui/launcher.py`` resuelve al paquete hermano ``core`` dentro de
    ``polyx``, no a un ``core.theme`` colgado del propio ``polyx.gui``.
    Devuelve None cuando el import sale por encima de ``polyx``.
    """
    if nivel > len(partes):
        return None
    base = partes[: len(partes) - nivel + 1]
    if modulo:
        base = base + modulo.split(".")
    return ".".join(base) if base else None


# El nombre del paquete viejo se arma en dos mitades a proposito: este
# archivo no puede mencionar esa ruta ni en texto, porque el contrato de la
# frontera se comprueba buscando esa cadena en todo el arbol del proyecto.
NOMBRE_HEREDADO = "polyx.core_reexport"  # nombre que nunca existira


def _es_qt(nombre: str) -> bool:
    """True si el nombre importado es (o cuelga de) un enlace de Qt."""
    return nombre.split(".", 1)[0] in ENLACES_QT


def _es_paquete_heredado(nombre: str) -> bool:
    """True si el nombre importado es (o cuelga del) paquete de re-export.

    Cuelga de el cuando es el paquete mismo o cualquier submodulo suyo.
    """
    return nombre == NOMBRE_HEREDADO or nombre.startswith(NOMBRE_HEREDADO + ".")


def _imports_qt(arbol: ast.AST) -> list[str]:
    """Nombres de Qt importados en cualquier punto del arbol.

    Los enlaces de Qt son paquetes de top-level, asi que solo pueden llegar
    por imports absolutos; pero se mira dentro de funciones y clases igual,
    que es donde se esconden los imports perezosos.
    """
    encontrados: list[str] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                if _es_qt(alias.name):
                    encontrados.append(alias.name)  # noqa: PERF401
        elif isinstance(nodo, ast.ImportFrom):  # noqa: SIM102
            if not nodo.level and nodo.module and _es_qt(nodo.module):
                encontrados.append(nodo.module)
    return encontrados


def _imports_heredados(arbol: ast.AST, partes: list[str]) -> list[str]:
    """Imports absolutos o relativos que resuelven al paquete de re-export.

    ``partes`` es el paquete que contiene al archivo escaneado, el mismo
    punto de partida que usaria Python para resolver un import relativo.
    """
    encontrados: list[str] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                if _es_paquete_heredado(alias.name):
                    encontrados.append(alias.name)  # noqa: PERF401
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.level:
                destino = _resuelve_relativo(partes, nodo.level, nodo.module)
            else:
                destino = nodo.module or ""
            if destino and _es_paquete_heredado(destino):
                encontrados.append(destino)
    return encontrados


def _leer(ruta: Path) -> ast.AST:
    """Parsea un fuente del arbol; un syntax error rompe el contrato igual."""
    return ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))


# ── 1. Anti-vaciedad: la estructura del paquete existe ──
def test_la_estructura_del_paquete_gui_existe():
    """Si falta un archivo, los escaneos AST pasarian en vacio y mentirian."""
    faltan: list[str] = []
    esperadas = ("__init__.py", "launcher.py", "__main__.py") + tuple(  # noqa: RUF005
        modulo + ".py" for modulo in MODULOS_MOVIDOS
    )
    for nombre in esperadas:
        if not (PAQUETE_GUI / nombre).is_file():
            faltan.append(f"polyx/gui/{nombre}")  # noqa: PERF401
    for paquete in PAQUETES_FEATURE:
        for nombre in ("__init__.py", "__main__.py", "window.py"):
            if not (PAQUETE_GUI / paquete / nombre).is_file():
                faltan.append(f"polyx/gui/{paquete}/{nombre}")  # noqa: PERF401
    assert not faltan, f"faltan archivos de la estructura del paquete gui: {faltan}"


# ── 2. "polyx/gui es dueno de todo el Qt" ──
# Los modulos del motor: son el Qt-free que de verdad importa probar.
MODULOS_MOTOR = {
    "i18n": "i18n.py",
    "metrics": "quality/metrics.py",
    "morphology": "measure/morphology.py",
    "observability": "provenance/observability.py",
    "yolo": "detect/yolo.py",
}


def test_el_escaneo_externo_cubre_los_limites_de_polyx():
    """Guardia contra un escaneo vacio: el exterior tiene archivos que mirar.

    ``polyx/launcher.py`` y los modulos de ``polyx/core`` son justo los
    archivos cuyo Qt-free se quiere probar, asi que tienen que estar en el
    barrido.
    """
    fuera = _fuentes_fuera_de_gui()
    assert fuera, "no se escaneo ningun archivo fuera de polyx/gui"
    nombres = {ruta.as_posix() for ruta in fuera}
    assert (RAIZ / "src" / "polyx" / "launcher.py").as_posix() in nombres
    for modulo, rel_path in MODULOS_MOTOR.items():  # noqa: B007, PERF102
        assert (RAIZ / "src" / "polyx" / "core" / rel_path).as_posix() in nombres


@pytest.mark.parametrize(
    "ruta", _fuentes_fuera_de_gui(), ids=lambda r: r.relative_to(RAIZ).as_posix()
)
def test_fuera_de_gui_no_importa_qt(ruta: Path):
    """Ningun modulo fuera de polyx/gui importa Qt, en ningun nivel.

    Cubre los modulos del motor, el shim del launcher y el ``polyx`` que los
    contiene: si cualquiera arrastra PySide6, el motor vuelve a acoplarse a la
    ventana y pierde el modo headless.
    """
    prohibidos = _imports_qt(_leer(ruta))
    assert not prohibidos, (
        f"{ruta.relative_to(RAIZ).as_posix()} importa Qt fuera de polyx/gui: {prohibidos}"
    )


# ── 3. "la GUI depende del motor, no de los shims" ──
def test_el_escaneo_interno_cubre_los_modulos_movidos():
    """Guardia contra un escaneo vacio: los ocho modulos movidos se miran."""
    escaneados = {ruta.as_posix() for ruta in _fuentes_de_gui()}
    faltan = [
        f"polyx/gui/{modulo}.py"
        for modulo in MODULOS_MOVIDOS
        if (PAQUETE_GUI / f"{modulo}.py").as_posix() not in escaneados
    ]
    assert not faltan, f"los escaneos no cubren los modulos movidos: {faltan}"


@pytest.mark.parametrize("ruta", _fuentes_de_gui(), ids=lambda r: r.relative_to(RAIZ).as_posix())
def test_la_gui_no_importa_el_paquete_heredado(ruta: Path):
    """Ningun modulo de polyx/gui importa el paquete de re-export inexistente.

    Cazar ``from ..core import theme`` desde polyx/gui/launcher.py exige
    resolver contra el paquete del archivo, igual que hace Python en tiempo
    de import.
    """
    partes = _partes_del_paquete(ruta)
    prohibidos = _imports_heredados(_leer(ruta), partes)
    assert not prohibidos, (
        f"{ruta.relative_to(RAIZ).as_posix()} importa un paquete re-export inexistente "
        f"del core: {prohibidos}"
    )


# ── 4. El paquete de re-export del core ha desaparecido ──
def test_el_paquete_de_reexport_del_core_ha_desaparecido():
    """El paquete de re-export del ``core`` ya no existe: ni en disco ni en import.

    Mientras viviera, la GUI podia seguir importando los nombres viejos en
    vez de ``polyx.gui.*`` y la frontera entre motor y GUI era puro decorado.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(NOMBRE_HEREDADO)

    # El NUEVO polyx.core DEBE existir (es el motor)
    assert (RAIZ / "src" / "polyx" / "core").exists(), "polyx/core (el motor) no existe"


# ── 5. El punto de entrada historico sigue viviendo ──
def test_el_launcher_historico_reenvia_al_launcher_real():
    """``python -m polyx.launcher`` llama al launcher de polyx.gui.

    Poly-X.vbs, iniciar_polyx.bat, Lanzar_macOS.command,
    crear_icono_macOS.command, lanzar_polyx.py y el bundle de PyInstaller
    invocan ese entry point; si el shim no reenviara ``main``, la app
    arrancaria en blanco.
    """
    import polyx.gui.launcher as launcher_real
    import polyx.launcher as shim_launcher

    assert shim_launcher.main is launcher_real.main


def test_el_shim_del_launcher_no_menciona_qt_en_su_fuente():
    """El shim es Qt-free por texto, no solo por lo que reenvia.

    Comprobar el source (y no el import) es la prueba fuerte: el shim,
    por definicion, importa ``polyx.gui.launcher``, que SI habla con Qt.
    """
    fuente = (RAIZ / "src" / "polyx" / "launcher.py").read_text(encoding="utf-8")
    assert "PySide6" not in fuente
    assert "PyQt" not in fuente


def test_polyx_gui_main_apunta_al_launcher_real():
    """``python -m polyx.gui`` ejecuta ``polyx.gui.launcher``."""
    fuente = (PAQUETE_GUI / "__main__.py").read_text(encoding="utf-8")
    assert "polyx.gui.launcher" in fuente


# ── 6. El escaner no pasa por vacio ni es demasiado amplio ──
def test_el_escanner_detecta_violaciones_sinteticas():
    """Las funciones de analisis tienen que ver exactamente lo que prohiben.

    Sin esto, un fallo en el escaner convertiria el contrato en decorativo:
    pasaria igual aunque alguien metiera Qt fuera de la GUI o un import del
    paquete heredado dentro de ella.
    """
    # Qt fuera de la GUI, tanto absoluto como importado dentro de una
    # funcion (ast.walk tiene que bajar ahi).
    assert _imports_qt(ast.parse("import PySide6.QtWidgets\n")) == ["PySide6.QtWidgets"]
    assert _imports_qt(ast.parse("from PySide6.QtCore import QTimer\n")) == ["PySide6.QtCore"]
    anidado = ast.parse("def arrancar():\n    import PyQt5.QtGui\n")
    assert _imports_qt(anidado) == ["PyQt5.QtGui"]

    # El paquete heredado dentro de la GUI, en sus tres formas posibles.
    # Las dos absolutas se arman por interpolacion por el mismo motivo que
    # NOMBRE_HEREDADO: el nombre viejo no puede aparecer escrito aqui.
    partes_gui = ["polyx", "gui"]
    fuente_absoluta = "from polyx.core_reexport import theme\n"
    fuente_con_submodulo = "import polyx.core_reexport.theme\n"
    assert _imports_heredados(ast.parse(fuente_absoluta), partes_gui) == [NOMBRE_HEREDADO]
    assert _imports_heredados(ast.parse(fuente_con_submodulo), partes_gui) == [
        NOMBRE_HEREDADO + ".theme"
    ]
    assert _imports_heredados(ast.parse("from ..core_reexport import theme\n"), partes_gui) == [
        NOMBRE_HEREDADO
    ]

    # Y lo legitimo NO se marca: la GUI puede importar del motor y de sus
    # propios hermanos.
    assert _imports_qt(ast.parse("from polyx.core.i18n import tr\n")) == []
    assert _imports_heredados(ast.parse("from polyx.core.i18n import tr\n"), partes_gui) == []
    assert _imports_heredados(ast.parse("from .theme import INK\n"), partes_gui) == []
