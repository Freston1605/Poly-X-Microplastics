"""Paleta, temas y QSS comunes para Poly-X.

Como funciona el sistema de temas
---------------------------------
Cada tema es un diccionario de *tokens* (INK, BG, ACCENT...). Al importar este
modulo se resuelve cual esta activo y sus valores se publican como atributos de
modulo, de modo que el resto del programa sigue escribiendo ``T.INK`` sin saber
que existen los temas.

Eso funciona porque **todo el codigo importa el modulo, no los nombres**
(``from . import theme as T`` y luego ``T.INK``): el valor se lee en el
momento de construir cada widget, no al importar. Si alguien escribiera
``from .theme import INK`` se quedaria con el color del tema que hubiera al
arrancar y no cambiaria nunca; por eso esa forma no se usa en ningun archivo.

El tema se resuelve una vez al arrancar, igual que el idioma: ``POLYX_TEMA`` si
esta puesta, si no lo guardado por el usuario, y a falta de todo, claro.

Anadir un tema es agregar una entrada a ``PALETAS`` con los mismos tokens. No
hay que tocar ninguna ventana.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from PySide6.QtGui import QColor

_ARCHIVO = Path.home() / ".polyx_tema.json"


# ════════════════════════════════════════════════════════════════════
#  Paletas
# ════════════════════════════════════════════════════════════════════
# Tokens y su significado, para que un tema nuevo no tenga que adivinarse:
#
#   INK / INK2 / INK3 / MUTED   texto, de mas a menos contraste
#   RULE / RULE_SOFT            bordes: visible y apenas insinuado
#   BG                          superficie elevada: tarjetas, barras, campos
#   BG_SOFT                     fondo hundido: el lienzo de la ventana
#   ACCENT / ACCENT_D           color primario y su estado hover
#   ON_ACCENT                   texto que va ENCIMA del acento; no siempre blanco
#   OK / WARN / ERR / VIO       estados semanticos, como FONDO de boton relleno
#   *_TX                        el mismo estado, pero como TEXTO sobre la tarjeta
#   FIELD_1 / FIELD_2 / RING    campo del microscopio (el panel del launcher)
#   TIP_BG / TIP_FG             tooltip; en oscuro no puede ser INK sobre blanco
#
# Por que un color de estado necesita dos versiones: el tono que da 5:1 con
# texto blanco encima es demasiado apagado para leerse *como* texto sobre un
# fondo oscuro. Un solo valor no puede cumplir las dos cosas, y elegir uno deja
# la mitad de la interfaz por debajo de 4.5:1. En los temas claros ambas
# coinciden; solo divergen donde hace falta.
#
# Las clases de polimero (PET/PP/LDPE) NO son tokens de tema: son datos del
# dominio y deben verse igual en los cuatro, o dos capturas del mismo analisis
# no serian comparables.

PALETAS: dict[str, dict[str, str]] = {

    # ── Claro: el de siempre, alineado con el manual HTML ──────────
    "claro": {
        "INK": "#1f2328", "INK2": "#424a53", "INK3": "#656d76", "MUTED": "#8c959f",
        "RULE": "#d0d7de", "RULE_SOFT": "#eaeef2",
        "BG": "#ffffff", "BG_SOFT": "#f6f8fa",
        "ACCENT": "#0969da", "ACCENT_D": "#0550ae", "ON_ACCENT": "#ffffff",
        "OK": "#1f6b5e", "WARN": "#9a6700", "ERR": "#cf222e", "VIO": "#6639ba",
        "ACCENT_D_TX": "#0550ae", "ACCENT_TX": "#0969da", "OK_TX": "#1f6b5e", "WARN_TX": "#9a6700",
        "ERR_TX": "#cf222e", "VIO_TX": "#6639ba",
        "FIELD_1": "#0d1117", "FIELD_2": "#1c2128", "RING": "#22272e",
        "TIP_BG": "#1f2328", "TIP_FG": "#ffffff",
    },

    # ── Oscuro: para trabajar de noche y junto a fotos de UV ───────
    # El acento es #1f6feb y no el #58a6ff habitual de GitHub porque encima
    # lleva texto blanco: con el azul claro el contraste cae a 2.6:1.
    "oscuro": {
        "INK": "#e6edf3", "INK2": "#c9d1d9", "INK3": "#8b949e", "MUTED": "#6e7681",
        "RULE": "#30363d", "RULE_SOFT": "#21262d",
        "BG": "#161b22", "BG_SOFT": "#0d1117",
        "ACCENT": "#1f6feb", "ACCENT_D": "#1a5fd0", "ON_ACCENT": "#ffffff",
        "OK": "#2ea043", "WARN": "#bb8009", "ERR": "#da3633", "VIO": "#8957e5",
        "ACCENT_D_TX": "#79c0ff", "ACCENT_TX": "#58a6ff", "OK_TX": "#3fb950", "WARN_TX": "#d29922",
        "ERR_TX": "#f85149", "VIO_TX": "#a371f7",
        "FIELD_1": "#05080d", "FIELD_2": "#11161e", "RING": "#30363d",
        "TIP_BG": "#e6edf3", "TIP_FG": "#0d1117",
    },

    # ── Azul nocturno: azul profundo con acento mas frio ───────────
    "azul": {
        "INK": "#e8f0fb", "INK2": "#bed0e6", "INK3": "#8ba3c0", "MUTED": "#6b83a1",
        "RULE": "#26405c", "RULE_SOFT": "#182b41",
        "BG": "#102039", "BG_SOFT": "#08111f",
        "ACCENT": "#2563eb", "ACCENT_D": "#1d4fd8", "ON_ACCENT": "#ffffff",
        "OK": "#0d9488", "WARN": "#d97706", "ERR": "#e11d48", "VIO": "#7c3aed",
        "ACCENT_D_TX": "#93c5fd", "ACCENT_TX": "#60a5fa", "OK_TX": "#2dd4bf", "WARN_TX": "#fbbf24",
        "ERR_TX": "#fb7185", "VIO_TX": "#a78bfa",
        "FIELD_1": "#040a14", "FIELD_2": "#0c1a2e", "RING": "#26405c",
        "TIP_BG": "#e8f0fb", "TIP_FG": "#08111f",
    },

    # ── Alto contraste: cumple AAA, para proyector y para vista cansada ──
    # No es un tema decorativo: en una sala con el proyector encendido el tema
    # claro normal pierde los bordes de 1 px y los grises de texto secundario.
    "contraste": {
        "INK": "#000000", "INK2": "#16191d", "INK3": "#2b3138", "MUTED": "#464e57",
        "RULE": "#24292f", "RULE_SOFT": "#8c959f",
        "BG": "#ffffff", "BG_SOFT": "#ffffff",
        "ACCENT": "#0a47a1", "ACCENT_D": "#062f6d", "ON_ACCENT": "#ffffff",
        "OK": "#0b5138", "WARN": "#5c3d00", "ERR": "#9b0d1b", "VIO": "#42217d",
        "ACCENT_D_TX": "#062f6d", "ACCENT_TX": "#0a47a1", "OK_TX": "#0b5138", "WARN_TX": "#5c3d00",
        "ERR_TX": "#9b0d1b", "VIO_TX": "#42217d",
        "FIELD_1": "#000000", "FIELD_2": "#101820", "RING": "#000000",
        "TIP_BG": "#000000", "TIP_FG": "#ffffff",
    },
}

# ════════════════════════════════════════════════════════════════════
#  Paleta fija para documentos generados
# ════════════════════════════════════════════════════════════════════
class _PaletaFija:
    """Acceso por atributo a una paleta que no cambia nunca."""

    def __init__(self, tokens: dict[str, str]) -> None:
        self.__dict__.update(tokens)


# Los informes HTML NO siguen el tema de la interfaz, y es deliberado: un
# informe se imprime, se adjunta a un correo y se archiva. Que saliera oscuro
# porque quien lo genero prefiere trabajar de noche gastaria tinta, se leeria
# mal en papel y haria que dos informes del mismo analisis no se parecieran.
# El documento es un artefacto que sale del programa, no una vista de el.
DOC = _PaletaFija(PALETAS["claro"])


# Nombre visible de cada tema. Se traduce con tr() en el punto de uso, no aqui:
# este modulo no puede importar i18n sin crear un ciclo (i18n no depende de
# theme, pero widgets.py depende de los dos y el orden dejaria de estar claro).
NOMBRES_TEMA: dict[str, str] = {
    "claro": "Claro",
    "oscuro": "Oscuro",
    "azul": "Azul nocturno",
    "contraste": "Alto contraste",
}

# Un tema oscuro necesita decisiones distintas en algunos sitios (sombras,
# opacidad de los halos). En vez de comprobar el nombre por ahi, se declara.
OSCUROS = {"oscuro", "azul"}

TEMA_POR_DEFECTO = "claro"


# ════════════════════════════════════════════════════════════════════
#  Colores de dominio: iguales en todos los temas
# ════════════════════════════════════════════════════════════════════
CLASS_COLOR_HEX = {
    "PET":  "#e3342f",   # se dibuja rojo
    "PP":   "#ff8c00",   # se dibuja naranjo
    "LDPE": "#ffd700",   # se dibuja amarillo
    "PE":   "#ffd700",   # alias
}


def class_qcolor(name: str) -> QColor:
    return QColor(CLASS_COLOR_HEX.get(name, "#888888"))


APP_FONT_FAMILY = "Segoe UI"


# ════════════════════════════════════════════════════════════════════
#  Estado del tema
# ════════════════════════════════════════════════════════════════════
_tema = TEMA_POR_DEFECTO
_animaciones = True
_suscriptores: list[Callable[[str], None]] = []


def tema() -> str:
    """Nombre del tema activo."""
    return _tema


def animaciones() -> bool:
    """Si la interfaz puede moverse.

    Vive junto al tema porque es la misma decision —como se ve el programa— y
    porque asi no hace falta un tercer archivo de preferencias. Apagarlo no es
    solo cuestion de gusto: el movimiento continuo molesta a quien tiene
    sensibilidad vestibular, y en un equipo sin GPU ahorra dibujar 30 cuadros
    por segundo que nadie mira.
    """
    return _animaciones


def set_animaciones(activas: bool) -> None:
    global _animaciones
    _animaciones = bool(activas)
    _guardar()
    for f in list(_suscriptores):
        try:
            f(_tema)
        except Exception:
            pass


def _guardar() -> None:
    try:
        _ARCHIVO.write_text(
            json.dumps({"tema": _tema, "animaciones": _animaciones}),
            encoding="utf-8")
    except Exception:
        pass


def es_oscuro() -> bool:
    return _tema in OSCUROS


def _publicar(nombre: str) -> None:
    """Vuelca los tokens de la paleta como atributos de este modulo."""
    for token, valor in PALETAS[nombre].items():
        globals()[token] = valor


def _construir_qss() -> str:
    """Arma la hoja global con los tokens ya publicados.

    Es una funcion y no una constante porque al cambiar de tema hay que
    rehacerla: una f-string a nivel de modulo se evaluaria una sola vez, con la
    paleta que hubiera al importar.
    """
    return f"""
* {{
    font-family: "{APP_FONT_FAMILY}", Helvetica, Arial, sans-serif;
    color: {INK};
}}
QWidget {{
    background: {BG};
}}

/* Botones primarios */
QPushButton#primary {{
    background: {ACCENT};
    color: {ON_ACCENT};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 10pt;
}}
QPushButton#primary:hover {{
    background: {ACCENT_D};
}}
QPushButton#primary:disabled {{
    background: {RULE};
    color: {MUTED};
}}

/* Botones de peligro (Detener, eliminar) */
QPushButton#danger {{
    background: {ERR};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton#danger:hover {{
    background: {_mezclar(ERR, INK, 0.22)};
}}
QPushButton#danger:disabled {{
    background: {RULE_SOFT};
    color: {MUTED};
}}

/* Botones secundarios (default) */
QPushButton {{
    background: {BG};
    color: {INK2};
    border: 1px solid {RULE};
    border-radius: 6px;
    padding: 7px 14px;
}}
QPushButton:hover {{
    background: {BG_SOFT};
    border-color: {MUTED};
}}
QPushButton:pressed {{
    background: {RULE_SOFT};
}}
QPushButton:disabled {{
    color: {MUTED};
    background: {BG_SOFT};
}}

/* Foco visible. Sin esto no se puede recorrer la interfaz con el teclado:
   el widget activo no se distingue de los demas. */
QPushButton:focus, QComboBox:focus, QLineEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QCheckBox:focus, QRadioButton:focus, QTabBar::tab:focus {{
    outline: none;
    border: 2px solid {ACCENT};
}}
QListWidget::item:focus, QTableWidget::item:focus {{
    outline: 1px solid {ACCENT};
}}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG};
    color: {INK};
    border: 1px solid {RULE};
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: {ON_ACCENT};
}}
QComboBox QAbstractItemView {{
    background: {BG};
    color: {INK};
    border: 1px solid {RULE};
    selection-background-color: {ACCENT};
    selection-color: {ON_ACCENT};
}}

/* Tarjetas (QGroupBox) */
QGroupBox {{
    background: {BG};
    border: 1px solid {RULE};
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 16px 16px 16px;
    font-weight: 600;
    color: {INK};
    font-size: 11pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
    background: {BG};
}}

/* Tablas */
QTableWidget, QTableView, QListWidget, QTreeWidget {{
    background: {BG};
    alternate-background-color: {BG_SOFT};
    gridline-color: {RULE_SOFT};
    border: 1px solid {RULE};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: {ON_ACCENT};
}}
QHeaderView::section {{
    background: {BG_SOFT};
    color: {INK2};
    padding: 8px 10px;
    border: none;
    border-right: 1px solid {RULE_SOFT};
    border-bottom: 1px solid {RULE};
    font-weight: 600;
}}

/* Casillas y pestanas */
QCheckBox, QRadioButton {{
    color: {INK2};
    spacing: 7px;
}}
QTabWidget::pane {{
    border: 1px solid {RULE};
    border-radius: 6px;
    background: {BG};
}}
QTabBar::tab {{
    background: {BG_SOFT};
    color: {INK2};
    border: 1px solid {RULE};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 14px;
}}
QTabBar::tab:selected {{
    background: {BG};
    color: {INK};
    font-weight: 600;
}}

/* Barras de progreso */
QProgressBar {{
    background: {BG_SOFT};
    border: 1px solid {RULE};
    border-radius: 6px;
    text-align: center;
    color: {INK2};
    height: 18px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}

/* Scrollbars discretas */
QScrollBar:vertical {{
    background: {BG_SOFT};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {RULE};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {BG_SOFT};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {RULE};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {MUTED};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    height: 0px;
    width: 0px;
}}

/* Menus y dialogos */
QMenu {{
    background: {BG};
    border: 1px solid {RULE};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 22px 6px 14px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {ACCENT};
    color: {ON_ACCENT};
}}
QDialog {{
    background: {BG_SOFT};
}}

/* Tooltips */
QToolTip {{
    background: {TIP_BG};
    color: {TIP_FG};
    border: 1px solid {RULE};
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 9pt;
}}

/* Labels especiales */
QLabel[role="kicker"] {{
    color: {ACCENT_D};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}}
QLabel[role="h1"] {{
    color: {INK};
    font-size: 28pt;
    font-weight: 600;
}}
QLabel[role="h2"] {{
    color: {INK};
    font-size: 18pt;
    font-weight: 600;
}}
QLabel[role="h3"] {{
    color: {INK2};
    font-size: 13pt;
    font-weight: 600;
}}
QLabel[role="muted"] {{
    color: {INK3};
    font-size: 9.5pt;
}}
QLabel[role="caption"] {{
    color: {INK3};
    font-size: 9pt;
}}
"""


# ════════════════════════════════════════════════════════════════════
#  Utilidades de color
# ════════════════════════════════════════════════════════════════════
def _mezclar(a: str, b: str, t: float) -> str:
    """Mezcla dos colores hex. t=0 devuelve a; t=1 devuelve b."""
    ca, cb = QColor(a), QColor(b)
    return QColor(
        round(ca.red()   + (cb.red()   - ca.red())   * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue()  + (cb.blue()  - ca.blue())  * t),
    ).name()


def mezclar(a: str, b: str, t: float) -> str:
    """Version publica de la mezcla, para las vistas que la necesiten."""
    return _mezclar(a, b, t)


def con_alfa(color: str, alfa: int) -> QColor:
    c = QColor(color)
    c.setAlpha(max(0, min(255, alfa)))
    return c


def sobre_fondo(color: str, t: float = 0.08) -> str:
    """Color tenue del mismo tono, para fondos de aviso o de chip.

    Se mezcla contra BG y no contra blanco: en un tema oscuro mezclar hacia
    blanco produce un pastel que no pertenece a la paleta.
    """
    return _mezclar(BG, color, t)


# ════════════════════════════════════════════════════════════════════
#  Carga y cambio de tema
# ════════════════════════════════════════════════════════════════════
def cargar() -> str:
    """Resuelve el tema una vez, en orden de precedencia."""
    global _tema, _animaciones
    guardado: dict = {}
    try:
        if _ARCHIVO.exists():
            guardado = json.loads(_ARCHIVO.read_text(encoding="utf-8")) or {}
    except Exception:
        # Un JSON corrupto no puede impedir que el programa arranque: se cae a
        # los valores por defecto y se sigue.
        guardado = {}

    env = os.environ.get("POLYX_TEMA", "").strip().lower()
    if env in PALETAS:
        _tema = env
    elif guardado.get("tema") in PALETAS:
        _tema = guardado["tema"]

    if os.environ.get("POLYX_SIN_ANIMACION", "").strip():
        _animaciones = False
    elif isinstance(guardado.get("animaciones"), bool):
        _animaciones = guardado["animaciones"]

    _aplicar(_tema)
    return _tema


def _aplicar(nombre: str) -> None:
    global GLOBAL_QSS
    _publicar(nombre)
    GLOBAL_QSS = _construir_qss()


def set_tema(nombre: str) -> None:
    """Cambia el tema, lo guarda y avisa a quien se haya suscrito.

    Los widgets ya construidos llevan su color incrustado en el stylesheet, asi
    que no cambian solos: cada ventana decide si se rehace (el launcher lo hace)
    o si espera a la proxima apertura (los modulos, que son procesos aparte).
    """
    global _tema
    if nombre not in PALETAS:
        return
    _tema = nombre
    _aplicar(nombre)
    _guardar()
    for f in list(_suscriptores):
        try:
            f(nombre)
        except Exception:
            pass


def al_cambiar(funcion: Callable[[str], None]) -> None:
    _suscriptores.append(funcion)


# Se resuelve al importar, antes de que nadie lea un token.
cargar()
