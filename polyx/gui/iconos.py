"""Hand-drawn vector icons, no files or dependencies.

Why not emoji
-------------
The interface used 🔬 🎯 🏷 📐 as module icons. An emoji is drawn by the
system font, not by the program: it changes shape between Windows 10 and
11, between Windows and macOS, and arrives in a fixed color, so it cannot
follow the theme or the accent color of its card. It also mixes styles —
some flat, others with a gradient — which shows when they sit together in
a grid of four.

These are drawn with ``QPainterPath`` over a 24x24 grid, with the stroke
proportional to the requested size. They look the same on both systems,
take whatever color they are given and stay sharp at any resolution.

Usage:

    from polyx.gui import iconos
    lbl.setPixmap(iconos.pixmap("detector", 26, T.ACCENT))
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

#: All paths are defined on this grid and scaled on the fly.
GRID = 24.0


# ════════════════════════════════════════════════════════════════════
#  Paths
# ════════════════════════════════════════════════════════════════════
def _detector(p: QPainter) -> None:
    """Crosshair target: to look up close and mark what is there."""
    p.drawEllipse(QPointF(11, 11), 7.0, 7.0)
    p.drawEllipse(QPointF(11, 11), 2.4, 2.4)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        p.drawLine(QPointF(11 + dx * 8.6, 11 + dy * 8.6),
                   QPointF(11 + dx * 10.4, 11 + dy * 10.4))
    p.drawLine(QPointF(16.2, 16.2), QPointF(20.6, 20.6))


def _trainer(p: QPainter) -> None:
    """Curve rising toward a bullseye: the training converges."""
    path = QPainterPath()
    path.moveTo(3.0, 19.5)
    path.cubicTo(7.5, 19.0, 9.5, 13.0, 12.5, 9.0)
    p.drawPath(path)
    p.drawEllipse(QPointF(17.0, 7.0), 4.6, 4.6)
    p.drawEllipse(QPointF(17.0, 7.0), 1.5, 1.5)
    p.drawLine(QPointF(3.0, 21.4), QPointF(21.0, 21.4))


def _labeler(p: QPainter) -> None:
    """Label with its eyelet, and an annotation box behind it."""
    path = QPainterPath()
    path.moveTo(12.6, 2.8)
    path.lineTo(21.2, 11.4)
    path.lineTo(12.4, 20.2)
    path.lineTo(3.8, 11.6)
    path.lineTo(3.8, 3.0)
    path.closeSubpath()
    p.drawPath(path)
    p.drawEllipse(QPointF(8.0, 7.2), 1.9, 1.9)


def _viewfinder(p: QPainter) -> None:
    """Diagonal ruler with its marks: measuring over the image."""
    p.save()
    p.translate(12, 12)
    p.rotate(-38)
    p.drawRoundedRect(QRectF(-10.5, -3.8, 21.0, 7.6), 1.4, 1.4)
    for x in (-6.0, -2.0, 2.0, 6.0):
        length = 3.4 if x in (-6.0, 2.0) else 2.2
        p.drawLine(QPointF(x, -3.8), QPointF(x, -3.8 + length))
    p.restore()


def _settings(p: QPainter) -> None:
    """Sliders.

    A six-tooth gear drawn with loose spokes reads as a sun, not as a
    wheel: for it to look like a gear the teeth would need a body, and at
    18 px they blur. Sliders carry none of that ambiguity and describe
    better what lives behind the button: not an engine to configure, but
    a handful of preferences.
    """
    for y, x_command in ((6.5, 15.5), (12.0, 9.0), (17.5, 14.0)):
        p.drawLine(QPointF(3.4, y), QPointF(20.6, y))
        p.drawEllipse(QPointF(x_command, y), 2.5, 2.5)


def _manual(p: QPainter) -> None:
    """Open book."""
    p.drawLine(QPointF(12, 6.4), QPointF(12, 19.2))
    path = QPainterPath()
    path.moveTo(12, 6.4)
    path.cubicTo(9.0, 4.2, 6.0, 4.0, 3.2, 4.8)
    path.lineTo(3.2, 17.6)
    path.cubicTo(6.0, 16.8, 9.0, 17.0, 12, 19.2)
    p.drawPath(path)
    path2 = QPainterPath()
    path2.moveTo(12, 6.4)
    path2.cubicTo(15.0, 4.2, 18.0, 4.0, 20.8, 4.8)
    path2.lineTo(20.8, 17.6)
    path2.cubicTo(18.0, 16.8, 15.0, 17.0, 12, 19.2)
    p.drawPath(path2)


def _text(p: QPainter) -> None:
    """Sheet with lines: the `LEEME`."""
    p.drawRoundedRect(QRectF(5.0, 3.2, 14.0, 17.6), 1.8, 1.8)
    for y in (8.0, 11.4, 14.8):
        p.drawLine(QPointF(8.2, y), QPointF(15.8, y))


# ── Detector and Trainer sidebars ──
def _model(p: QPainter) -> None:
    """Stacked layers: the weights of a model."""
    tapa = QPainterPath()
    tapa.moveTo(12, 3.0)
    tapa.lineTo(20.8, 7.4)
    tapa.lineTo(12, 11.8)
    tapa.lineTo(3.2, 7.4)
    tapa.closeSubpath()
    p.drawPath(tapa)
    for dy in (4.6, 9.2):
        half = QPainterPath()
        half.moveTo(3.2, 7.4 + dy)
        half.lineTo(12, 11.8 + dy)
        half.lineTo(20.8, 7.4 + dy)
        p.drawPath(half)


def _images(p: QPainter) -> None:
    """Photograph: frame, sun and horizon."""
    p.drawRoundedRect(QRectF(3.2, 4.8, 17.6, 14.4), 2.0, 2.0)
    p.drawEllipse(QPointF(8.4, 9.4), 1.7, 1.7)
    monte = QPainterPath()
    monte.moveTo(4.6, 17.4)
    monte.lineTo(9.6, 12.2)
    monte.lineTo(13.0, 15.4)
    monte.lineTo(16.0, 12.6)
    monte.lineTo(19.6, 17.4)
    p.drawPath(monte)


def _edit(p: QPainter) -> None:
    """Pencil: annotating by hand."""
    body_2 = QPainterPath()
    body_2.moveTo(4.0, 20.0)
    body_2.lineTo(5.5, 15.3)
    body_2.lineTo(16.2, 4.6)
    body_2.lineTo(19.4, 7.8)
    body_2.lineTo(8.7, 18.5)
    body_2.closeSubpath()
    p.drawPath(body_2)
    p.drawLine(QPointF(14.4, 6.4), QPointF(17.6, 9.6))


def _run(p: QPainter) -> None:
    """Play triangle."""
    play = QPainterPath()
    play.moveTo(8.0, 5.4)
    play.lineTo(19.0, 12.0)
    play.lineTo(8.0, 18.6)
    play.closeSubpath()
    p.drawPath(play)


def _results(p: QPainter) -> None:
    """Bars: the analysis figures."""
    p.drawLine(QPointF(3.4, 20.2), QPointF(20.6, 20.2))
    for x, height_2 in ((5.6, 7.6), (10.3, 12.6), (15.0, 5.2)):
        p.drawRoundedRect(QRectF(x, 20.2 - height_2, 3.4, height_2), 1.0, 1.0)


def _errors(p: QPainter) -> None:
    """Warning triangle."""
    tri = QPainterPath()
    tri.moveTo(12.0, 3.6)
    tri.lineTo(21.2, 19.8)
    tri.lineTo(2.8, 19.8)
    tri.closeSubpath()
    p.drawPath(tri)
    p.drawLine(QPointF(12.0, 10.0), QPointF(12.0, 14.4))
    p.drawEllipse(QPointF(12.0, 17.0), 0.55, 0.55)


def _compare(p: QPainter) -> None:
    """Split panel: the same thing seen from two sides."""
    p.drawRoundedRect(QRectF(3.2, 5.0, 17.6, 14.0), 2.0, 2.0)
    p.drawLine(QPointF(12.0, 5.0), QPointF(12.0, 19.0))
    p.drawEllipse(QPointF(7.6, 12.0), 2.0, 2.0)
    p.drawRoundedRect(QRectF(14.4, 10.0, 4.0, 4.0), 0.8, 0.8)


def _report(p: QPainter) -> None:
    """Sheet with a curve: the report."""
    p.drawRoundedRect(QRectF(4.8, 3.2, 14.4, 17.6), 1.8, 1.8)
    p.drawLine(QPointF(8.0, 7.6), QPointF(16.0, 7.6))
    curve = QPainterPath()
    curve.moveTo(8.0, 16.4)
    curve.lineTo(10.8, 13.2)
    curve.lineTo(13.2, 15.0)
    curve.lineTo(16.0, 10.8)
    p.drawPath(curve)


def _dataset(p: QPainter) -> None:
    """Folder: the dataset."""
    folder = QPainterPath()
    folder.moveTo(3.2, 19.0)
    folder.lineTo(3.2, 5.8)
    folder.lineTo(9.4, 5.8)
    folder.lineTo(11.4, 8.4)
    folder.lineTo(20.8, 8.4)
    folder.lineTo(20.8, 19.0)
    folder.closeSubpath()
    p.drawPath(folder)


def _augmentation(p: QPainter) -> None:
    """Two shifted frames: the same image as variants."""
    p.drawRoundedRect(QRectF(8.0, 3.4, 12.6, 12.6), 2.0, 2.0)
    p.drawRoundedRect(QRectF(3.4, 8.0, 12.6, 12.6), 2.0, 2.0)


def _assess(p: QPainter) -> None:
    """Clipboard with a checkmark: validating against a reference."""
    p.drawRoundedRect(QRectF(4.8, 4.6, 14.4, 16.0), 2.0, 2.0)
    p.drawRoundedRect(QRectF(9.0, 2.6, 6.0, 4.0), 1.0, 1.0)
    seen = QPainterPath()
    seen.moveTo(8.6, 13.4)
    seen.lineTo(11.0, 15.8)
    seen.lineTo(15.6, 10.4)
    p.drawPath(seen)


def _export(p: QPainter) -> None:
    """Out of the box: converting to another format."""
    box = QPainterPath()
    box.moveTo(4.4, 14.2)
    box.lineTo(4.4, 19.8)
    box.lineTo(19.6, 19.8)
    box.lineTo(19.6, 14.2)
    p.drawPath(box)
    p.drawLine(QPointF(12.0, 15.4), QPointF(12.0, 4.4))
    arrow = QPainterPath()
    arrow.moveTo(7.9, 8.6)
    arrow.lineTo(12.0, 4.4)
    arrow.lineTo(16.1, 8.6)
    p.drawPath(arrow)


# ── Card headers ──
def _see(p: QPainter) -> None:
    """Eye: inspecting."""
    eye = QPainterPath()
    eye.moveTo(2.6, 12.0)
    eye.cubicTo(6.5, 5.8, 17.5, 5.8, 21.4, 12.0)
    eye.cubicTo(17.5, 18.2, 6.5, 18.2, 2.6, 12.0)
    p.drawPath(eye)
    p.drawEllipse(QPointF(12, 12), 3.1, 3.1)


def _ray(p: QPainter) -> None:
    """Bolt: the quick, the automatic."""
    r = QPainterPath()
    r.moveTo(13.6, 2.6)
    r.lineTo(5.4, 13.4)
    r.lineTo(11.2, 13.4)
    r.lineTo(10.4, 21.4)
    r.lineTo(18.6, 10.6)
    r.lineTo(12.8, 10.6)
    r.closeSubpath()
    p.drawPath(r)


def _diana(p: QPainter) -> None:
    """Bullseye: the goal."""
    p.drawEllipse(QPointF(12, 12), 8.6, 8.6)
    p.drawEllipse(QPointF(12, 12), 4.8, 4.8)
    p.drawEllipse(QPointF(12, 12), 1.2, 1.2)


def _list(p: QPainter) -> None:
    """Clipboard with lines."""
    p.drawRoundedRect(QRectF(4.8, 4.6, 14.4, 16.0), 2.0, 2.0)
    p.drawRoundedRect(QRectF(9.0, 2.6, 6.0, 4.0), 1.0, 1.0)
    for y in (11.0, 14.2, 17.4):
        p.drawLine(QPointF(8.2, y), QPointF(15.8, y))


def _save(p: QPainter) -> None:
    """Floppy disk: saving to disk."""
    body_2 = QPainterPath()
    body_2.moveTo(4.4, 4.4)
    body_2.lineTo(16.6, 4.4)
    body_2.lineTo(19.6, 7.4)
    body_2.lineTo(19.6, 19.6)
    body_2.lineTo(4.4, 19.6)
    body_2.closeSubpath()
    p.drawPath(body_2)
    p.drawRect(QRectF(8.2, 4.4, 7.6, 5.0))
    p.drawRect(QRectF(7.4, 13.4, 9.2, 6.2))


def _record(p: QPainter) -> None:
    """Sheet with many lines: the run log."""
    p.drawRoundedRect(QRectF(4.6, 3.4, 14.8, 17.2), 1.8, 1.8)
    for y in (7.4, 10.2, 13.0, 15.8):
        p.drawLine(QPointF(7.8, y), QPointF(16.2, y))


def _package(p: QPainter) -> None:
    """Box in perspective: a packaged artifact."""
    box = QPainterPath()
    box.moveTo(12, 2.8)
    box.lineTo(20.6, 7.2)
    box.lineTo(20.6, 16.8)
    box.lineTo(12, 21.2)
    box.lineTo(3.4, 16.8)
    box.lineTo(3.4, 7.2)
    box.closeSubpath()
    p.drawPath(box)
    p.drawLine(QPointF(3.4, 7.2), QPointF(12, 11.8))
    p.drawLine(QPointF(20.6, 7.2), QPointF(12, 11.8))
    p.drawLine(QPointF(12, 11.8), QPointF(12, 21.2))


def _wait(p: QPainter) -> None:
    """Clock: how long it will take."""
    p.drawEllipse(QPointF(12, 12), 8.6, 8.6)
    p.drawLine(QPointF(12, 6.8), QPointF(12, 12))
    p.drawLine(QPointF(12, 12), QPointF(15.8, 14.4))


def _lookfor(p: QPainter) -> None:
    """Magnifying glass."""
    p.drawEllipse(QPointF(10.6, 10.6), 6.4, 6.4)
    p.drawLine(QPointF(15.4, 15.4), QPointF(20.4, 20.4))


def _parts(p: QPainter) -> None:
    """Grid: splitting into parts."""
    p.drawRoundedRect(QRectF(3.4, 3.4, 7.6, 7.6), 1.4, 1.4)
    p.drawRoundedRect(QRectF(13.0, 3.4, 7.6, 7.6), 1.4, 1.4)
    p.drawRoundedRect(QRectF(3.4, 13.0, 7.6, 7.6), 1.4, 1.4)
    p.drawRoundedRect(QRectF(13.0, 13.0, 7.6, 7.6), 1.4, 1.4)


def _rehearsal(p: QPainter) -> None:
    """Flask: the test."""
    flask = QPainterPath()
    flask.moveTo(9.4, 3.0)
    flask.lineTo(9.4, 9.4)
    flask.lineTo(4.2, 18.6)
    flask.cubicTo(3.2, 20.4, 4.4, 21.2, 6.0, 21.2)
    flask.lineTo(18.0, 21.2)
    flask.cubicTo(19.6, 21.2, 20.8, 20.4, 19.8, 18.6)
    flask.lineTo(14.6, 9.4)
    flask.lineTo(14.6, 3.0)
    p.drawPath(flask)
    p.drawLine(QPointF(8.2, 3.0), QPointF(15.8, 3.0))


def _reload(p: QPainter) -> None:
    """Circular arrow: re-reading."""
    arco = QPainterPath()
    arco.arcMoveTo(QRectF(4.0, 4.0, 16.0, 16.0), 65)
    arco.arcTo(QRectF(4.0, 4.0, 16.0, 16.0), 65, 285)
    p.drawPath(arco)
    tip = QPainterPath()
    tip.moveTo(11.6, 2.6)
    tip.lineTo(16.4, 6.0)
    tip.lineTo(11.4, 9.0)
    p.drawPath(tip)


def _idea(p: QPainter) -> None:
    """Light bulb: the recommendation."""
    bulb = QPainterPath()
    bulb.moveTo(8.4, 14.6)
    bulb.cubicTo(5.2, 11.0, 7.0, 4.2, 12.0, 4.2)
    bulb.cubicTo(17.0, 4.2, 18.8, 11.0, 15.6, 14.6)
    p.drawPath(bulb)
    p.drawLine(QPointF(9.2, 17.4), QPointF(14.8, 17.4))
    p.drawLine(QPointF(10.2, 20.2), QPointF(13.8, 20.2))


def _curve(p: QPainter) -> None:
    """Curve with dots: the evolution."""
    p.drawLine(QPointF(3.4, 20.2), QPointF(20.6, 20.2))
    line = QPainterPath()
    line.moveTo(4.8, 16.6)
    line.lineTo(9.4, 11.4)
    line.lineTo(13.6, 14.0)
    line.lineTo(19.4, 5.6)
    p.drawPath(line)
    for x, y in ((9.4, 11.4), (13.6, 14.0), (19.4, 5.6)):
        p.drawEllipse(QPointF(x, y), 1.15, 1.15)


def _team(p: QPainter) -> None:
    """Monitor: the hardware."""
    p.drawRoundedRect(QRectF(3.0, 4.6, 18.0, 12.4), 1.8, 1.8)
    p.drawLine(QPointF(9.0, 20.6), QPointF(15.0, 20.6))
    p.drawLine(QPointF(12.0, 17.0), QPointF(12.0, 20.6))


# ── Actions ──
def _stop(p: QPainter) -> None:
    """Square: stop."""
    p.drawRoundedRect(QRectF(6.4, 6.4, 11.2, 11.2), 1.6, 1.6)


def _circle(p: QPainter) -> None:
    """Circumference with three marks: calibrating from three rim points."""
    p.drawEllipse(QPointF(12, 12), 8.2, 8.2)
    for x, y in ((12.0, 3.8), (19.1, 16.1), (4.9, 16.1)):
        p.drawEllipse(QPointF(x, y), 1.5, 1.5)


def _line(p: QPainter) -> None:
    """Segment with end caps: calibrating from a known length."""
    p.drawLine(QPointF(4.4, 12.0), QPointF(19.6, 12.0))
    p.drawLine(QPointF(4.4, 8.2), QPointF(4.4, 15.8))
    p.drawLine(QPointF(19.6, 8.2), QPointF(19.6, 15.8))


def _bin(p: QPainter) -> None:
    """Trash can: delete."""
    p.drawLine(QPointF(3.6, 6.4), QPointF(20.4, 6.4))
    p.drawRoundedRect(QRectF(9.0, 3.0, 6.0, 3.4), 1.0, 1.0)
    cube = QPainterPath()
    cube.moveTo(5.8, 6.4)
    cube.lineTo(7.0, 20.6)
    cube.lineTo(17.0, 20.6)
    cube.lineTo(18.2, 6.4)
    p.drawPath(cube)
    p.drawLine(QPointF(10.2, 10.2), QPointF(10.6, 17.0))
    p.drawLine(QPointF(13.8, 10.2), QPointF(13.4, 17.0))


def _undo(p: QPainter) -> None:
    """Arrow that returns to the left."""
    arco = QPainterPath()
    arco.moveTo(4.4, 10.4)
    arco.cubicTo(9.0, 4.6, 18.4, 5.6, 19.6, 12.4)
    arco.cubicTo(20.4, 16.8, 17.2, 19.8, 13.0, 20.0)
    p.drawPath(arco)
    tip = QPainterPath()
    tip.moveTo(9.4, 10.0)
    tip.lineTo(4.0, 10.6)
    tip.lineTo(5.6, 5.4)
    p.drawPath(tip)


def _redo(p: QPainter) -> None:
    """The same arrow, mirrored."""
    p.save()
    p.translate(24, 0)
    p.scale(-1, 1)
    _undo(p)
    p.restore()


DRAWINGS = {
    "detector": _detector,
    "entrenador": _trainer,
    "etiquetador": _labeler,
    "visor": _viewfinder,
    "ajustes": _settings,
    "manual": _manual,
    "texto": _text,
    # Sidebars
    "modelo": _model,
    "imagenes": _images,
    "editar": _edit,
    "ejecutar": _run,
    "resultados": _results,
    "errores": _errors,
    "comparar": _compare,
    "reporte": _report,
    "dataset": _dataset,
    "augmentacion": _augmentation,
    "evaluar": _assess,
    "exportar": _export,
    # Card headers
    "ver": _see,
    "rayo": _ray,
    "diana": _diana,
    "lista": _list,
    "guardar": _save,
    "registro": _record,
    "paquete": _package,
    "espera": _wait,
    "buscar": _lookfor,
    "piezas": _parts,
    "ensayo": _rehearsal,
    "recargar": _reload,
    "idea": _idea,
    "curva": _curve,
    "equipo": _team,
    # Actions
    "detener": _stop,
    "circulo": _circle,
    "linea": _line,
    "papelera": _bin,
    "deshacer": _undo,
    "rehacer": _redo,
}


# ════════════════════════════════════════════════════════════════════
#  Render
# ════════════════════════════════════════════════════════════════════
def pixmap(name_2: str, tam: int, color: str, thickness: float = 1.9) -> QPixmap:
    """Draws the icon ``nombre`` at ``tam`` px in the requested color.

    ``grosor`` is in units of the 24 grid, not in pixels, so the stroke
    keeps its relative weight when scaled.
    """
    drawing = DRAWINGS.get(name_2)
    px = QPixmap(tam, tam)
    px.fill(Qt.transparent)
    if drawing is None:
        return px

    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(tam / GRID, tam / GRID)
    pencil = QPen(QColor(color), thickness)
    pencil.setCapStyle(Qt.RoundCap)
    pencil.setJoinStyle(Qt.RoundJoin)
    p.setPen(pencil)
    p.setBrush(Qt.NoBrush)
    drawing(p)
    p.end()
    return px


def icon(name_2: str, tam: int, color: str, thickness: float = 1.9) -> QIcon:
    return QIcon(pixmap(name_2, tam, color, thickness))


def switchable_icon(name_2: str, tam: int, color_off: str, color_on: str,
                     thickness: float = 1.85) -> QIcon:
    """Icon that changes color when the button is toggled.

    Qt keeps a per-state map inside the ``QIcon`` itself: for a
    ``checkable`` button it uses ``QIcon.On`` when checked and
    ``QIcon.Off`` when not. That way the active page's icon tints to the
    accent together with its text, without rebuilding the button or
    listening to ``toggled``.
    """
    ic = QIcon()
    ic.addPixmap(pixmap(name_2, tam, color_off, thickness), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(name_2, tam, color_on, thickness), QIcon.Normal, QIcon.On)
    # Without the Active pair, Qt reuses the Off one when the mouse hovers over
    # a button that is already checked and the icon flashes to the dimmed colour.
    ic.addPixmap(pixmap(name_2, tam, color_on, thickness), QIcon.Active, QIcon.On)
    ic.addPixmap(pixmap(name_2, tam, color_off, thickness), QIcon.Active, QIcon.Off)
    return ic
