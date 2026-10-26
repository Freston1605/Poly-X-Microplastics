"""Palette, themes and common QSS for Poly-X.

How the theme system works
--------------------------
Each theme is a dictionary of *tokens* (INK, BG, ACCENT...). When this
module is imported, the active one is resolved and its values are
published as module attributes, so the rest of the program keeps writing
``T.INK`` without knowing themes exist.

That works because **all the code imports the module, not the names**
(``from . import theme as T`` and then ``T.INK``): the value is read at
the moment each widget is built, not at import. If someone wrote
``from .theme import INK`` they would keep the color of whichever theme
happened to be active at startup and it would never change; that is why
that form is used in no file.

The theme is resolved once at startup, same as the language:
``POLYX_TEMA`` if set, then what the user saved, and failing all of
that, the default (`claro`).

Adding a theme means adding an entry to ``PALETTES`` with the same
tokens. No window has to be touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from PySide6.QtGui import QColor

_FILE = Path.home() / ".polyx_tema.json"


# ════════════════════════════════════════════════════════════════════
#  Palettes
# ════════════════════════════════════════════════════════════════════
# Tokens and their meaning, so a new theme does not have to be guessed:
#
#   INK / INK2 / INK3 / MUTED   text, from most to least contrast
#   RULE / RULE_SOFT            borders: clear and barely hinted at
#   BG                          raised surface: cards, bars, fields
#   BG_SOFT                     sunken background: the window canvas
#   ACCENT / ACCENT_D           primary colour and its hover state
#   ON_ACCENT                   text that goes ON TOP of the accent; not always white
#   OK / WARN / ERR / VIO       semantic states, as the BACKGROUND of a filled button
#   *_TX                        the same state, but as TEXT on the card
#   FIELD_1 / FIELD_2 / RING    microscope field (the launcher panel)
#   TIP_BG / TIP_FG             tooltip; in dark mode it cannot be INK on white
#
# Why a state colour needs two versions: the tone that gives 5:1 with
# white text on top is too dull to be read *as* text on a
# dark background. A single value cannot satisfy both, and choosing one leaves
# half the interface below 4.5:1. In light themes both
# coincide; they only diverge where necessary.
#
# The polymer classes (PET/PP/LDPE) are NOT theme tokens: they are domain
# data and must look the same in all four, or two captures of the same analysis
# would not be on the same scale.

PALETTES: dict[str, dict[str, str]] = {
    # ── Light: the usual one, aligned with the HTML manual ──
    "claro": {
        "INK": "#1f2328",
        "INK2": "#424a53",
        "INK3": "#656d76",
        "MUTED": "#8c959f",
        "RULE": "#d0d7de",
        "RULE_SOFT": "#eaeef2",
        "BG": "#ffffff",
        "BG_SOFT": "#f6f8fa",
        "ACCENT": "#0969da",
        "ACCENT_D": "#0550ae",
        "ON_ACCENT": "#ffffff",
        "OK": "#1f6b5e",
        "WARN": "#9a6700",
        "ERR": "#cf222e",
        "VIO": "#6639ba",
        "ACCENT_D_TX": "#0550ae",
        "ACCENT_TX": "#0969da",
        "OK_TX": "#1f6b5e",
        "WARN_TX": "#9a6700",
        "ERR_TX": "#cf222e",
        "VIO_TX": "#6639ba",
        "FIELD_1": "#0d1117",
        "FIELD_2": "#1c2128",
        "RING": "#22272e",
        "TIP_BG": "#1f2328",
        "TIP_FG": "#ffffff",
    },
    # ── Dark: for working at night and next to UV photos ──
    # The accent is #1f6feb and not GitHub's usual #58a6ff because on top
    # it carries white text: with the light blue the contrast drops to 2.6:1.
    "oscuro": {
        "INK": "#e6edf3",
        "INK2": "#c9d1d9",
        "INK3": "#8b949e",
        "MUTED": "#6e7681",
        "RULE": "#30363d",
        "RULE_SOFT": "#21262d",
        "BG": "#161b22",
        "BG_SOFT": "#0d1117",
        "ACCENT": "#1f6feb",
        "ACCENT_D": "#1a5fd0",
        "ON_ACCENT": "#ffffff",
        "OK": "#2ea043",
        "WARN": "#bb8009",
        "ERR": "#da3633",
        "VIO": "#8957e5",
        "ACCENT_D_TX": "#79c0ff",
        "ACCENT_TX": "#58a6ff",
        "OK_TX": "#3fb950",
        "WARN_TX": "#d29922",
        "ERR_TX": "#f85149",
        "VIO_TX": "#a371f7",
        "FIELD_1": "#05080d",
        "FIELD_2": "#11161e",
        "RING": "#30363d",
        "TIP_BG": "#e6edf3",
        "TIP_FG": "#0d1117",
    },
    # ── Night blue: deep blue with a cooler accent ──
    "azul": {
        "INK": "#e8f0fb",
        "INK2": "#bed0e6",
        "INK3": "#8ba3c0",
        "MUTED": "#6b83a1",
        "RULE": "#26405c",
        "RULE_SOFT": "#182b41",
        "BG": "#102039",
        "BG_SOFT": "#08111f",
        "ACCENT": "#2563eb",
        "ACCENT_D": "#1d4fd8",
        "ON_ACCENT": "#ffffff",
        "OK": "#0d9488",
        "WARN": "#d97706",
        "ERR": "#e11d48",
        "VIO": "#7c3aed",
        "ACCENT_D_TX": "#93c5fd",
        "ACCENT_TX": "#60a5fa",
        "OK_TX": "#2dd4bf",
        "WARN_TX": "#fbbf24",
        "ERR_TX": "#fb7185",
        "VIO_TX": "#a78bfa",
        "FIELD_1": "#040a14",
        "FIELD_2": "#0c1a2e",
        "RING": "#26405c",
        "TIP_BG": "#e8f0fb",
        "TIP_FG": "#08111f",
    },
    # ── High contrast: meets AAA, for projectors and tired eyes ──
    # It is not a decorative theme: in a room with the projector on, the normal
    # light theme loses the 1 px borders and the secondary text greys.
    "contraste": {
        "INK": "#000000",
        "INK2": "#16191d",
        "INK3": "#2b3138",
        "MUTED": "#464e57",
        "RULE": "#24292f",
        "RULE_SOFT": "#8c959f",
        "BG": "#ffffff",
        "BG_SOFT": "#ffffff",
        "ACCENT": "#0a47a1",
        "ACCENT_D": "#062f6d",
        "ON_ACCENT": "#ffffff",
        "OK": "#0b5138",
        "WARN": "#5c3d00",
        "ERR": "#9b0d1b",
        "VIO": "#42217d",
        "ACCENT_D_TX": "#062f6d",
        "ACCENT_TX": "#0a47a1",
        "OK_TX": "#0b5138",
        "WARN_TX": "#5c3d00",
        "ERR_TX": "#9b0d1b",
        "VIO_TX": "#42217d",
        "FIELD_1": "#000000",
        "FIELD_2": "#101820",
        "RING": "#000000",
        "TIP_BG": "#000000",
        "TIP_FG": "#ffffff",
    },
}


# ════════════════════════════════════════════════════════════════════
#  Fixed palette for generated documents
# ════════════════════════════════════════════════════════════════════
class _PaletteFixed:
    """Attribute access to a palette that never changes."""

    def __init__(self, tokens: dict[str, str]) -> None:
        self.__dict__.update(tokens)


# HTML reports do NOT follow the interface theme, and that is deliberate: a
# report is printed, attached to an email and archived. If it came out dark
# because whoever generated it prefers working at night, it would waste ink, read
# badly on paper and make two reports of the same analysis not look alike.
# The document is an artefact that leaves the program, not a view of it.
DOC = _PaletteFixed(PALETTES["claro"])


# Display name of each theme. It is translated with tr() at the point of use, not here:
# this module cannot import i18n without creating a cycle (i18n does not depend on
# theme, but widgets.py depends on both and the order would stop being clear).
NAMES_THEME: dict[str, str] = {
    "claro": "Claro",
    "oscuro": "Oscuro",
    "azul": "Azul nocturno",
    "contraste": "Alto contraste",
}

# A dark theme needs different decisions in some places (shadows,
# halo opacity). Instead of checking the name all over the place, it is declared.
DARK = {"oscuro", "azul"}

DEFAULT_THEME = "claro"


# ════════════════════════════════════════════════════════════════════
#  Domain colours: the same in every theme
# ════════════════════════════════════════════════════════════════════
CLASS_COLOR_HEX = {
    "PET": "#e3342f",  # drawn red
    "PP": "#ff8c00",  # drawn orange
    "LDPE": "#ffd700",  # drawn yellow
    "PE": "#ffd700",  # alias
}


def class_qcolor(name: str) -> QColor:
    return QColor(CLASS_COLOR_HEX.get(name, "#888888"))


APP_FONT_FAMILY = "Segoe UI"


# ════════════════════════════════════════════════════════════════════
#  Theme state
# ════════════════════════════════════════════════════════════════════
_theme = DEFAULT_THEME
_animations = True
_subscribers: list[Callable[[str], None]] = []


def theme() -> str:
    """Name of the active theme."""
    return _theme


def animations() -> bool:
    """Whether the interface may move.

    Lives next to the theme because it is the same decision — how the
    program looks — and because that way no third preferences file is
    needed. Turning it off is not only a matter of taste: continuous
    motion bothers people with vestibular sensitivity, and on a machine
    without a GPU it saves drawing 30 frames per second nobody watches.
    """
    return _animations


def set_animations(active: bool) -> None:
    global _animations  # noqa: PLW0603
    _animations = bool(active)
    _save()
    for f in _subscribers:
        try:  # noqa: SIM105
            f(_theme)
        except Exception:  # noqa: BLE001, S110
            pass


def _save() -> None:
    try:  # noqa: SIM105
        _FILE.write_text(json.dumps({"tema": _theme, "animaciones": _animations}), encoding="utf-8")
    except Exception:  # noqa: BLE001, S110
        pass


def is_dark() -> bool:
    return _theme in DARK


def _post(name_2: str) -> None:
    """Publishes the palette tokens as attributes of this module."""
    for token, value in PALETTES[name_2].items():
        globals()[token] = value


def _build_qss() -> str:
    """Assembles the global sheet from the already published tokens.

    A function and not a constant because on a theme change it has to be
    rebuilt: a module-level f-string would be evaluated once, with the
    palette active at import.
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
    background: {_mingle(ERR, INK, 0.22)};
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
#  Colour utilities
# ════════════════════════════════════════════════════════════════════
def _mingle(a: str, b: str, t: float) -> str:
    """Mixes two hex colors. t=0 returns a; t=1 returns b."""
    ca, cb = QColor(a), QColor(b)
    return QColor(
        round(ca.red() + (cb.red() - ca.red()) * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue() + (cb.blue() - ca.blue()) * t),
    ).name()


def mingle(a: str, b: str, t: float) -> str:
    """Public version of the mix, for the views that need it."""
    return _mingle(a, b, t)


def with_alpha(color: str, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(max(0, min(255, alpha)))
    return c


def on_background(color: str, t: float = 0.08) -> str:
    """Dimmed color of the same hue, for warning or chip backgrounds.

    It is mixed against BG and not against white: in a dark theme mixing
    toward white produces a pastel that does not belong to the palette.
    """
    return _mingle(BG, color, t)


# ════════════════════════════════════════════════════════════════════
#  Theme loading and switching
# ════════════════════════════════════════════════════════════════════
def carry() -> str:
    """Resolves the theme once, in precedence order."""
    global _theme, _animations  # noqa: PLW0603
    saved: dict = {}
    try:
        if _FILE.exists():
            saved = json.loads(_FILE.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        # A corrupt JSON cannot prevent the program from starting: it falls back to
        # the default values and continues.
        saved = {}

    env = os.environ.get("POLYX_TEMA", "").strip().lower()
    if env in PALETTES:
        _theme = env
    elif saved.get("tema") in PALETTES:
        _theme = saved["tema"]

    if os.environ.get("POLYX_SIN_ANIMACION", "").strip():
        _animations = False
    elif isinstance(saved.get("animaciones"), bool):
        _animations = saved["animaciones"]

    _apply(_theme)
    return _theme


def _apply(name_2: str) -> None:
    global GLOBAL_QSS  # noqa: PLW0603
    _post(name_2)
    GLOBAL_QSS = _build_qss()


def set_theme(name_2: str) -> None:
    """Changes the theme, saves it and notifies whoever subscribed.

    Widgets already built carry their color embedded in the stylesheet,
    so they do not change by themselves: each window decides whether to
    rebuild (the launcher does) or to wait for the next opening (the
    modules, which are separate processes).
    """
    global _theme  # noqa: PLW0603
    if name_2 not in PALETTES:
        return
    _theme = name_2
    _apply(name_2)
    _save()
    for f in _subscribers:
        try:  # noqa: SIM105
            f(name_2)
        except Exception:  # noqa: BLE001, S110
            pass


def when_changing(function: Callable[[str], None]) -> None:
    _subscribers.append(function)


# It is resolved on import, before anyone reads a token.
carry()
