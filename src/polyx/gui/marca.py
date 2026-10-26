"""Identity of every window: its icon, its title and its place in the bar.

Why it was needed
-----------------
Poly-X starts as ``pythonw.exe -m polyx.<modulo>``. Windows does not
group taskbar windows by the executable but by its **AppUserModelID**,
and a Python process that does not declare its own inherits the one from
the Python installation. Hence the bar showing the Python logo: the
window was Poly-X's, but for the shell the program was Python.

Two things fix it, and both are needed:

1. **Declaring an own AppUserModelID** before creating the first window
   (``plataforma.fijar_identidad_app``). Without it, Windows keeps using
   the Python icon no matter what window icon is set.
2. **Setting the window icon** (``QApplication.setWindowIcon``). Without
   it, Windows has an own identity but no drawing to associate with it.

Each module declares a distinct identifier, so the Detector and the
Trainer appear in the bar as two separate programs, with their icon and
their name. That is right here: they are independent processes, opened
and closed separately, and often used two at a time.

The icon colors DO NOT follow the theme
---------------------------------------
An icon is identity, not view. If the Detector's changed from blue to
another hue because someone prefers working at night, it would stop
being recognizable in the taskbar — which is exactly what it is for.
They go fixed, same as the report palette (``theme.DOC``).
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap

from polyx.core import paths as _paths
from polyx.core import platform
from polyx.core.i18n import tr

from . import iconos

#: AppUserModelID prefix. The format Windows requires is
#: Company.Product.SubProduct.Version; what matters is that it is stable
#: across launches, because taskbar pinning and jump lists hang off it.
_APP_ID = "CristofherFerrada.PolyX"


# ════════════════════════════════════════════════════════════════════
#  Identity of each module
# ════════════════════════════════════════════════════════════════════
# (icon from `polyx/gui/iconos.py`, badge colour, window title)
#
# The colours are the same ones with which the launcher paints each module's card,
# but written here as literals: the icon has to look the same
# with any theme.
MODULES: dict[str, dict] = {
    "launcher": {
        "icono": None,  # uses assets/polyx.ico, the usual brand
        "color": "#0969da",
        "titulo": "Poly-X · Suite de microplásticos",
        "id": _APP_ID,
    },
    "detector": {
        "icono": "detector",
        "color": "#0969da",
        "titulo": "Poly-X · Detector",
        "id": f"{_APP_ID}.Detector",
    },
    "entrenador": {
        "icono": "entrenador",
        "color": "#1a7f5f",
        "titulo": "Poly-X · Entrenador",
        "id": f"{_APP_ID}.Entrenador",
    },
    "etiquetador": {
        "icono": "etiquetador",
        "color": "#b07400",
        "titulo": "Poly-X · Etiquetador",
        "id": f"{_APP_ID}.Etiquetador",
    },
    "visor": {
        "icono": "visor",
        "color": "#6639ba",
        "titulo": "Poly-X · Visor",
        "id": f"{_APP_ID}.Visor",
    },
}

#: Sizes that Windows requests for an icon, from the one used by the bar to 'Details'.
#: All of them are generated: if one is missing, the shell rescales the nearest one and the
#: result looks dirty precisely in the taskbar, which is where people look most.
_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


# ════════════════════════════════════════════════════════════════════
#  Badge drawing
# ════════════════════════════════════════════════════════════════════
def _badge(tam: int, color: str, glyph: str | None) -> QPixmap:
    """Rounded square in the module's color, with its glyph in white.

    Reproduces the ``LogoBadge`` of the top bar — the blue square with
    the 'P' — so the taskbar icon and the one inside the program read as
    the same brand.
    """
    px = QPixmap(tam, tam)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)

    # The margin and radius are proportional to the size: at 16 px a fixed 2 px
    # margin eats one eighth of the icon.
    margin = max(0.5, tam * 0.045)
    radius = tam * 0.22
    body_2 = QRectF(margin, margin, tam - 2 * margin, tam - 2 * margin)

    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    path = QPainterPath()
    path.addRoundedRect(body_2, radius, radius)
    p.drawPath(path)

    if glyph is None:
        # Launcher: the brand's 'P'.
        p.setPen(QColor("#ffffff"))
        f = QFont("Segoe UI", 1)
        f.setPixelSize(int(tam * 0.60))
        f.setWeight(QFont.Bold)
        p.setFont(f)
        p.drawText(body_2, Qt.AlignCenter, "P")
    else:
        # The glyph grows proportionally as the icon shrinks. At 16 px a drawing
        # at 58 % leaves 9 px per side: with the stroke and antialiasing on top, the
        # detail turns to mush and the five modules are distinguished only by colour.
        # By giving it more surface and a slightly thicker stroke, the silhouette
        # survives the taskbar size, which is where people look most.
        if tam <= 24:
            fraction, streak = 0.76, 2.9
        elif tam <= 48:
            fraction, streak = 0.64, 2.5
        else:
            fraction, streak = 0.58, 2.2
        side = max(8, int(tam * fraction))
        drawing = iconos.pixmap(glyph, side, "#ffffff", thickness=streak)
        p.drawPixmap(int((tam - side) / 2), int((tam - side) / 2), drawing)

    p.end()
    return px


def icon_module(module: str) -> QIcon:
    """Multi-resolution icon of the module, ready for window and bar."""
    cfg = MODULES.get(module, MODULES["launcher"])

    # The launcher uses the usual .ico, which is the Desktop shortcut's:
    # that way the pinned icon and the window icon are the same drawing.
    if cfg["icono"] is None:
        ico = _paths.ASSETS_DIR / "polyx.ico"
        if ico.exists():
            return QIcon(str(ico))

    ic = QIcon()
    for tam in _SIZES:
        ic.addPixmap(_badge(tam, cfg["color"], cfg["icono"]))
    return ic


# ════════════════════════════════════════════════════════════════════
#  Application
# ════════════════════════════════════════════════════════════════════
def identify(app, module: str) -> None:
    """Gives the process its identity before the system. Before the first window.

    The order matters: the AppUserModelID must be set before Windows
    creates the taskbar button, that is, before the first ``show()``.
    Setting it later has no effect until reopened.
    """
    cfg = MODULES.get(module, MODULES["launcher"])
    platform.set_identity_app(cfg["id"])
    app.setApplicationName("Poly-X")
    app.setApplicationDisplayName(tr(cfg["titulo"]))
    app.setOrganizationName("Cristofher Ferrada")
    app.setWindowIcon(icon_module(module))


def titular(window, module: str) -> None:
    """Sets title and icon on a concrete window.

    Any window already inherits the application icon, but it is repeated
    here on purpose: if the module opens from another process — as the
    launcher does — the window can be built before ``identificar`` has
    run, and then it is left without an icon.
    """
    cfg = MODULES.get(module, MODULES["launcher"])
    window.setWindowTitle(tr(cfg["titulo"]))
    window.setWindowIcon(icon_module(module))
