"""Module 1 — Launcher (main menu). Reproduces Figure 1 of the manual.

Hero with kicker, 'Poly-X analytics' title, subtitle, 4 KPI chips,
and 4 large module cards (Detector, Trainer, Labeler, Viewer).
"""
from __future__ import annotations
import logging
import math
import random
import sys
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QLinearGradient, QRadialGradient,
    QPainterPath, QFont, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QFrame, QScrollArea, QMessageBox
)

from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.gui import marca
from polyx.model.i18n import tr, language
from polyx.gui.preferencias import PreferencesDialog
from polyx.gui.widgets import LogoBadge, HLine
from polyx.gui.updates import UPDATER, SearchUpdates, can_update
from polyx.model.plataforma import open_in_system, launch_updater
from polyx import __version__

# ──────────────────────────────────────────────────────────────────────
# Instrumented startup: the launcher leaves a JSONL trace in `app.jsonl`.
# The observability imports are lazy, inside the functions,
# so that a failure in them neither blocks nor delays the window launch.
# ──────────────────────────────────────────────────────────────────────


def _register_boot() -> None:
    """Emits ``bootstrap.finish`` with version, python and system onto app.jsonl.

    Never blocks the launch: if HOME is read-only or the log directory
    cannot be created, the launcher opens as usual anyway.
    Observability must never be worse than having none.
    """
    try:
        from polyx.model import observabilidad
        from polyx.model.plataforma import name_system
        log = observabilidad.logger_for("app")
        observabilidad.event(
            log,
            "bootstrap.finish",
            version=__version__,
            python=sys.version.split()[0],
            sistema=name_system(),
        )
    except Exception:
        pass


def _spawn_module(dotted: str) -> bool:
    """Spawns the module in a separate process; True if it started, False if not.

    Returns instead of raising: that way the window decides how to warn
    the user and the observability drag (module.launch /
    module.launch.error in app.jsonl) cannot turn into a crash of the
    interface.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(dotted)
        if spec is None:
            return False
        subprocess.Popen(
            [sys.executable, "-m", dotted],
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except Exception as e:
        try:
            from polyx.model import observabilidad
            observabilidad.event(
                observabilidad.logger_for("app"),
                "module.launch.error",
                level=logging.WARNING,
                modulo=dotted,
                error=str(e),
            )
        except Exception:
            pass
        return False
    try:
        from polyx.model import observabilidad
        observabilidad.event(
            observabilidad.logger_for("app"), "module.launch", modulo=dotted
        )
    except Exception:
        pass
    return True


# ──────────────────────────────────────────────────────────────────────
# Hero panel: a microscope field with a detection in progress
# ──────────────────────────────────────────────────────────────────────
class MicroscopePanel(QWidget):
    """Dark field with fluorescent particles that a sweep keeps detecting.

    It is not just any decoration: it counts what the program does. The
    particles emit the way Nile Red emits under UV — a saturated core
    inside a halo that fades with distance, not a flat color disc — and
    over them a sweep passes, leaving boxes with class and confidence
    behind, which is exactly the Detector's output.

    What moves, and why:

    * **Drift on two axes.** Each particle travels its own Lissajous
      figure, with its own period and phase. Before, all of them went up
      and down on the same sine and the set pulsed as a single block,
      which is exactly what betrays a cheap animation.
    * **Three depth planes.** The ones in the back are smaller, dimmer
      and slower: that is what a preparation with thickness looks like,
      not a tracing.
    * **A sweep that detects.** It crosses the field each cycle and
      marks what it finds, the box appearing when the sweep reaches it.

    The two smallest particles never get a box. Deliberate: they are the
    ones the real detector loses, and drawing them all boxed would
    promise a recall the model does not have.

    With animations off (see ``theme.animaciones``) a single composed
    frame is drawn, with the sweep finished and all the boxes placed.
    """

    #: Duration of a full cycle, in seconds.
    CYCLE = 7.6
    #: Cycle segments, as a fraction: wait, sweep, hold, exit.
    _START_SWEEP = 0.06
    _FIN_SWEEP = 0.52
    _START_OUTPUT = 0.86

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 240)
        self.setMaximumWidth(360)

        rnd = random.Random(11)
        classes = ("PET", "PP", "LDPE")
        planes = (0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2)   # 4 background, 5 middle, 4 foreground

        # Distribution by rings, not by free draw. Thirteen random positions
        # inside a circle leave half the field empty and a clump in the other
        # half: that is what randomness does with small samples.
        # By distributing 1 + 4 + 8 across rings and drawing only the deviation, the
        # field ends up covered and still no grid shows through.
        sites = [(0.0, 0.0)]
        for howmany, radius in ((4, 0.215), (8, 0.355)):
            for k in range(howmany):
                ang = 2 * math.pi * k / howmany + rnd.uniform(-0.32, 0.32)
                rad = radius + rnd.uniform(-0.035, 0.035)
                sites.append((rad * math.cos(ang), rad * math.sin(ang)))
        rnd.shuffle(sites)

        # The classes are assigned in rotation, not by draw. With thirteen independent
        # draws the field came out almost entirely yellow with a single PET, which
        # gives a false impression: the program distinguishes three polymers and this
        # panel is the first thing one sees of it.
        shift = [classes[i % 3] for i in range(len(planes))]
        rnd.shuffle(shift)

        self._particles: list[dict] = []
        for plane, (dx, dy), class_id in zip(planes, sites, shift):
            scale = (0.55, 0.82, 1.0)[plane]
            self._particles.append({
                "x": 0.5 + dx,
                "y": 0.5 + dy,
                "r": rnd.uniform(5.5, 15.0) * scale,
                "clase": class_id,
                "forma": rnd.choices((0, 1, 2), weights=(5, 3, 2))[0],
                "giro": rnd.uniform(0, 360),
                "plano": plane,
                "conf": rnd.uniform(0.71, 0.97),
                "ax": rnd.uniform(0.006, 0.020) * scale,
                "ay": rnd.uniform(0.006, 0.020) * scale,
                "wx": rnd.uniform(0.20, 0.44),
                "wy": rnd.uniform(0.20, 0.44),
                "fx": rnd.uniform(0, 2 * math.pi),
                "fy": rnd.uniform(0, 2 * math.pi),
            })

        # The two smallest are left without a box: they are the ones that get lost.
        for i, part in enumerate(sorted(self._particles, key=lambda q: q["r"])):
            part["detectable"] = i >= 2
        # Only the larger ones are labelled, or the field becomes illegible.
        for i, part in enumerate(sorted(self._particles, key=lambda q: -q["r"])):
            part["etiqueta"] = i < 3 and part["detectable"]

        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        # 33 ms ~ 30 fps: drift already looks continuous and costs half as much as 60,
        # in a panel nobody stares at.
        self._timer.setInterval(33)

    # ── Lifecycle ──────────────────────────────────────────────────
    def showEvent(self, e):
        super().showEvent(e)
        if T.animations():
            self._timer.start()

    def hideEvent(self, e):
        # A covered panel that keeps repainting drains battery without being seen.
        self._timer.stop()
        super().hideEvent(e)

    def _tick(self):
        self._t += self._timer.interval() / 1000.0
        self.update()

    # ── Cycle geometry ─────────────────────────────────────────────
    def _phase(self) -> float:
        """Position within the cycle, from 0 to 1."""
        if not T.animations():
            # Fixed frame: sweep finished and boxes placed.
            return (self._START_OUTPUT + self._FIN_SWEEP) / 2
        return (self._t % self.CYCLE) / self.CYCLE

    def _advance_sweep(self, phase: float):
        """Height of the sweep (0 top, 1 bottom). None when not passing."""
        if phase < self._START_SWEEP:
            return None
        if phase <= self._FIN_SWEEP:
            return (phase - self._START_SWEEP) / (self._FIN_SWEEP - self._START_SWEEP)
        return None

    def _pos(self, part: dict, t: float):
        return (part["x"] + part["ax"] * math.sin(part["wx"] * t + part["fx"]),
                part["y"] + part["ay"] * math.sin(part["wy"] * t + part["fy"]))

    def _alpha_box(self, phase: float, y_rel: float) -> float:
        """Opacity of the box of a particle located at height y_rel."""
        if phase < self._START_SWEEP:
            return 0.0
        if phase >= self._START_OUTPUT:
            return max(0.0, 1.0 - (phase - self._START_OUTPUT) / (1.0 - self._START_OUTPUT))
        advance = self._advance_sweep(phase)
        if advance is None:          # the sweep has already crossed the whole field
            return 1.0
        if advance < y_rel:
            return 0.0
        return min(1.0, (advance - y_rel) * 9.0)

    # ── Drawing ─────────────────────────────────────────────────────
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        t = self._t
        phase = self._phase()
        W, H = self.width(), self.height()
        side = min(W, H) - 18
        cx, cy = W / 2, H / 2
        field = QRectF(cx - side / 2, cy - side / 2, side, side)

        # ── Objective: ring with a highlight at the top, so that it looks like glass ──
        p.setPen(QPen(QColor(T.RING), 5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(field.adjusted(-3, -3, 3, 3))
        enhancement = QLinearGradient(field.topLeft(), field.bottomLeft())
        enhancement.setColorAt(0.0, T.with_alpha("#ffffff", 45 if T.is_dark() else 60))
        enhancement.setColorAt(0.5, T.with_alpha("#ffffff", 0))
        p.setPen(QPen(QBrush(enhancement), 2))
        p.drawEllipse(field.adjusted(-1, -1, 1, 1))

        # ── Field: radial, brighter at the centre. The vignette comes for free ──
        background = QRadialGradient(field.center(), side / 2)
        background.setColorAt(0.0, QColor(T.FIELD_2))
        background.setColorAt(0.72, QColor(T.FIELD_1))
        background.setColorAt(1.0, QColor(T.FIELD_1).darker(140))
        p.setBrush(QBrush(background))
        p.setPen(Qt.NoPen)
        p.drawEllipse(field)

        p.save()
        cutout = QPainterPath()
        cutout.addEllipse(field)
        p.setClipPath(cutout)

        # ── Faint reticle: gives scale without competing with the particles ──
        p.setPen(QPen(T.with_alpha("#ffffff", 12), 1))
        paso = side / 6
        for i in range(1, 6):
            p.drawLine(QPointF(field.left() + i * paso, field.top()),
                       QPointF(field.left() + i * paso, field.bottom()))
            p.drawLine(QPointF(field.left(), field.top() + i * paso),
                       QPointF(field.right(), field.top() + i * paso))

        # ── Particles, from back to front ──
        for part in sorted(self._particles, key=lambda q: q["plano"]):
            self._draw_particle(p, part, field, side, t)

        # ── Sweep ──
        advance = self._advance_sweep(phase)
        if advance is not None:
            self._draw_sweep(p, field, advance)

        # ── Detection boxes ──
        # Labels are placed from largest to smallest and any that falls
        # on top of one already placed is dropped: two overlapping captions cannot be read, and in
        # a field of 13 particles that happens as soon as drift brings them together.
        busy: list[QRectF] = []
        for part in sorted(self._particles, key=lambda q: -q["r"]):
            if not part["detectable"]:
                continue
            _, y_rel = self._pos(part, t)
            alpha = self._alpha_box(phase, y_rel)
            if alpha > 0.01:
                self._draw_box(p, part, field, side, t, alpha, busy)

        p.restore()
        self._draw_scale(p, cx, cy, side)

    def _draw_particle(self, p, part, field, side, t):
        x_rel, y_rel = self._pos(part, t)
        x = field.left() + x_rel * side
        y = field.top() + y_rel * side
        r = part["r"] * (side / 250.0)
        base = QColor(T.CLASS_COLOR_HEX[part["clase"]])

        # Halo: a radial that fades out, not a constant-alpha disc. It is the
        # difference between looking like fluorescence and looking like a sticker.
        intensity = (115, 155, 195)[part["plano"]]
        radius_halo = r * 3.4
        halo = QRadialGradient(QPointF(x, y), radius_halo)
        center_2 = QColor(base); center_2.setAlpha(intensity)
        half = QColor(base); half.setAlpha(int(intensity * 0.30))
        tenue = QColor(base); tenue.setAlpha(int(intensity * 0.10))
        edge = QColor(base); edge.setAlpha(0)
        halo.setColorAt(0.00, center_2)
        halo.setColorAt(0.30, half)
        halo.setColorAt(0.62, tenue)
        halo.setColorAt(1.00, edge)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(QPointF(x, y), radius_halo, radius_halo)

        # Body. The background planes are dimmed: depth of field.
        body_2 = QColor(base)
        if part["plano"] == 0:
            body_2.setAlpha(150)
        elif part["plano"] == 1:
            body_2.setAlpha(205)
        p.setBrush(QBrush(body_2))

        shape = part["forma"]
        if shape == 0:                                    # rounded fragment
            p.drawEllipse(QPointF(x, y), r, r * 0.92)
        elif shape == 1:                                  # fibre
            p.save()
            p.translate(x, y)
            p.rotate(part["giro"] + 9 * math.sin(0.35 * t + part["fx"]))
            p.drawRoundedRect(QRectF(-r * 1.9, -r * 0.24, r * 3.8, r * 0.48),
                              r * 0.24, r * 0.24)
            p.restore()
        else:                                             # irregular fragment
            p.save()
            p.translate(x, y)
            p.rotate(part["giro"])
            path = QPainterPath()
            path.addEllipse(QPointF(0, 0), r * 1.15, r * 0.68)
            p.drawPath(path)
            p.restore()

        # Core: the point where emission saturates. Without it, a large
        # particle reads as a flat patch of colour.
        if part["plano"] > 0 and shape != 1:
            core = QRadialGradient(QPointF(x - r * 0.18, y - r * 0.18), r * 0.85)
            core.setColorAt(0.0, T.with_alpha("#ffffff", 120))
            core.setColorAt(1.0, T.with_alpha("#ffffff", 0))
            p.setBrush(QBrush(core))
            p.drawEllipse(QPointF(x - r * 0.18, y - r * 0.18), r * 0.85, r * 0.85)

    def _draw_sweep(self, p, field, advance):
        """Luminous band crossing the field from top to bottom."""
        y = field.top() + advance * field.height()
        height_2 = field.height() * 0.13
        banda = QLinearGradient(QPointF(0, y - height_2), QPointF(0, y + 2))
        banda.setColorAt(0.0, T.with_alpha(T.ACCENT_TX, 0))
        banda.setColorAt(0.85, T.with_alpha(T.ACCENT_TX, 26))
        banda.setColorAt(1.0, T.with_alpha(T.ACCENT_TX, 52))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(banda))
        p.drawRect(QRectF(field.left(), y - height_2, field.width(), height_2))
        p.setPen(QPen(T.with_alpha(T.ACCENT_TX, 190), 1.4))
        p.drawLine(QPointF(field.left(), y), QPointF(field.right(), y))

    def _draw_box(self, p, part, field, side, t, alpha, busy):
        x_rel, y_rel = self._pos(part, t)
        x = field.left() + x_rel * side
        y = field.top() + y_rel * side
        r = part["r"] * (side / 250.0)
        margin = r * (2.0 if part["forma"] == 1 else 1.55)
        box = QRectF(x - margin, y - margin * 0.85, margin * 2, margin * 1.7)

        color = QColor(T.CLASS_COLOR_HEX[part["clase"]])
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(T.with_alpha(color.name(), int(215 * alpha)), 1.5))
        p.drawRect(box)

        # Marked corners: it reads as a detection box even at a 1.5 px stroke
        p.setPen(QPen(T.with_alpha(color.name(), int(255 * alpha)), 2.4))
        c = min(box.width(), box.height()) * 0.26
        for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            px = box.left() + sx * box.width()
            py = box.top() + sy * box.height()
            p.drawLine(QPointF(px, py), QPointF(px + (c if sx == 0 else -c), py))
            p.drawLine(QPointF(px, py), QPointF(px, py + (c if sy == 0 else -c)))

        if not part["etiqueta"] or alpha < 0.5:
            return
        text = f'{part["clase"]} {part["conf"]:.2f}'
        p.setFont(QFont(T.APP_FONT_FAMILY, 7, QFont.DemiBold))
        fm = p.fontMetrics()
        width_2 = fm.horizontalAdvance(text) + 8
        height_2 = fm.height() + 2
        label_2 = QRectF(box.left(), box.top() - height_2 - 2, width_2, height_2)
        if any(label_2.intersects(other.adjusted(-3, -3, 3, 3)) for other in busy):
            return
        busy.append(label_2)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(T.with_alpha(color.name(), int(230 * alpha))))
        p.drawRoundedRect(label_2, 2, 2)
        p.setPen(QPen(T.with_alpha("#0d1117", int(255 * alpha))))
        p.drawText(label_2, Qt.AlignCenter, text)

    def _draw_scale(self, p, cx, cy, side):
        # Inside the circle, not at the corner of its bounding box: at
        # 0.355 of the radius downward the edge has already come in and a bar
        # placed at the corner ends up floating over the card.
        width_2 = side * 0.19
        y = cy + side * 0.355
        x = cx + side * 0.30 - width_2
        text = "10 μm"
        p.setFont(QFont(T.APP_FONT_FAMILY, 8, QFont.Bold))
        fm = p.fontMetrics()

        # Translucent plate underneath. The particles drift, so any
        # fixed position sooner or later ends up on top of one: instead of looking for
        # a gap that does not exist, the contrast is guaranteed. It is also what
        # any microscopy program does with its scale bar.
        height_plate = fm.height() + 16
        plate = QRectF(x - 9, y - height_plate + 8, width_2 + 18, height_plate)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(T.with_alpha("#000000", 125)))
        p.drawRoundedRect(plate, 4, 4)

        p.setPen(QPen(T.with_alpha("#ffffff", 230), 2))
        p.drawLine(QPointF(x, y), QPointF(x + width_2, y))
        for extreme in (x, x + width_2):
            p.drawLine(QPointF(extreme, y - 4), QPointF(extreme, y + 4))
        p.setPen(QPen(T.with_alpha("#ffffff", 240)))
        p.drawText(QPointF(x + width_2 / 2 - fm.horizontalAdvance(text) / 2, y - 7), text)


# ──────────────────────────────────────────────────────────────────────
# Top KPI chips (`3 / POLÍMEROS / PET·PP·LDPE`)
# ──────────────────────────────────────────────────────────────────────
class StatChip(QFrame):
    def __init__(self, big: str, label: str, sub: str, color: str):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        self.setMinimumHeight(96)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)

        l_big = QLabel(big)
        l_big.setStyleSheet(f"color: {color}; font-size: 26pt; font-weight: 700; border: none;")
        l_label = QLabel(label.upper())
        l_label.setStyleSheet(
            f"color: {T.INK2}; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.5px; border: none;"
        )
        l_sub = QLabel(sub)
        l_sub.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        l_sub.setWordWrap(True)

        lay.addWidget(l_big)
        lay.addWidget(l_label)
        lay.addWidget(l_sub)


# ──────────────────────────────────────────────────────────────────────
# Module card
# ──────────────────────────────────────────────────────────────────────
class ModuleCard(QFrame):
    """Card of a module: icon, title, description and open action.

    The whole card is the click area, not just the button. That is why
    it needs to say it is pressable — accent border and elevation on
    hover — and be reachable with the keyboard: without ``StrongFocus``
    the module grid was out of the Tab walk and existed only for the
    mouse.
    """

    def __init__(self, number: int, icon_name: str, title: str, description: str,
                 accent: str, on_open, shortcut: str = ""):
        super().__init__()
        self._on_open = on_open
        self._accent = accent
        self.setStyleSheet(self._qss(active_2=False))
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(title)
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(8)

        top = QHBoxLayout()
        tag = QLabel(f'{tr("MÓDULO")} 0{number}')
        tag.setStyleSheet(
            f"color: white; background: {accent}; padding: 3px 8px; "
            f"border-radius: 4px; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.2px; border: none;"
        )
        tag.setFixedHeight(20)
        tag.setAlignment(Qt.AlignCenter)
        top.addWidget(tag, 0, Qt.AlignLeft | Qt.AlignVCenter)
        top.addStretch(1)
        self._btn_open = QPushButton(tr("Abrir  →"))
        self._btn_open.setCursor(Qt.PointingHandCursor)
        self._btn_open.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {accent}; "
            f"border: none; font-weight: 600; padding: 4px 0; }}"
            f"QPushButton:hover {{ color: {T.ACCENT_D_TX}; }}"
        )
        self._btn_open.clicked.connect(self._open)
        top.addWidget(self._btn_open)
        lay.addLayout(top)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(iconos.pixmap(icon_name, 30, accent, 1.85))
        icon_lbl.setStyleSheet("border: none;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {T.INK}; font-size: 17pt; font-weight: 600; border: none;"
        )
        title_row.addWidget(icon_lbl)
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)
        lay.addLayout(title_row)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        lay.addWidget(desc)
        lay.addStretch(1)

        self.setCursor(Qt.PointingHandCursor)
        if shortcut:
            self.setToolTip(f"{title}  ·  {shortcut}")

    # ── Visual state ───────────────────────────────────────────
    def _qss(self, active_2: bool) -> str:
        edge = self._accent if active_2 else T.RULE
        background = T.on_background(self._accent, 0.05) if active_2 else T.BG
        return (f"ModuleCard {{ background: {background}; border: 1px solid {edge};"
                f" border-radius: 10px; }}")

    def enterEvent(self, ev):
        # Without this, a whole clickable card gave no signal of being clickable.
        self.setStyleSheet(self._qss(active_2=True))
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.setStyleSheet(self._qss(active_2=False))
        super().leaveEvent(ev)

    def focusInEvent(self, ev):
        self.setStyleSheet(self._qss(active_2=True))
        super().focusInEvent(ev)

    def focusOutEvent(self, ev):
        self.setStyleSheet(self._qss(active_2=False))
        super().focusOutEvent(ev)

    # ── Action ──────────────────────────────────────────────────
    def _open(self):
        """Opens the module and announces that it is opening.

        A module takes two or three seconds to appear, because its
        process imports torch before drawing anything. Without the
        signal, that silence reads as a click that did not land and
        people click again, so two windows end up open.
        """
        self._btn_open.setEnabled(False)
        self._btn_open.setText(tr("Abriendo…"))
        QTimer.singleShot(3200, self._restore_button)
        self._on_open()

    def _restore_button(self):
        self._btn_open.setEnabled(True)
        self._btn_open.setText(tr("Abrir  →"))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._open()
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._open()
            return
        super().keyPressEvent(ev)


# ──────────────────────────────────────────────────────────────────────
# Main Launcher window
# ──────────────────────────────────────────────────────────────────────
class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        marca.titular(self, "launcher")
        self.resize(1180, 820)

        # It is stored separately because the window is rebuilt when the theme changes
        # and the version notice cannot be lost in that rebuild.
        self._sha_version_new: str = ""

        self._search = SearchUpdates(self)
        self._search.there_is_a_new_version.connect(self._notify_version_new)
        # It is triggered with the window already drawn: startup takes priority.
        QTimer.singleShot(1200, self._search.lookfor)

        self._build_ui()

    def _build_ui(self):
        """Builds the whole window with the active palette.

        Rebuilding it is the simple way to apply a theme hot: the colors
        live embedded in each widget's stylesheet, so there is no way to
        repaint them without recreating them. In a window with four
        cards it costs milliseconds and no flicker is seen.
        """
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Topbar ──
        topbar = QFrame()
        topbar.setStyleSheet(f"background: {T.BG}; border-bottom: 1px solid {T.RULE};")
        topbar.setFixedHeight(58)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(24, 0, 24, 0)
        tb.setSpacing(12)
        tb.addWidget(LogoBadge("POLY-X", "Microplastics analytics suite"))
        tb.addStretch(1)

        # Preferences. The language used to live here as a loose dropdown; now
        # it shares a dialog with the theme and motion, which are the same
        # decision —how the program looks and behaves— and not three
        # controls scattered across the bar.
        self.btn_prefs = QPushButton()
        self.btn_prefs.setIcon(iconos.icon("ajustes", 18, T.INK2, 2.0))
        self.btn_prefs.setIconSize(QSize(18, 18))
        self.btn_prefs.setText(tr("Preferencias"))
        self.btn_prefs.setCursor(Qt.PointingHandCursor)
        self.btn_prefs.setToolTip(tr("Tema, idioma y movimiento"))
        self.btn_prefs.clicked.connect(self._open_preferences)
        tb.addWidget(self.btn_prefs)

        version_lbl = QLabel(f"v{__version__}")
        version_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        tb.addWidget(version_lbl)

        # New version notice. It starts hidden and only appears if the background
        # check confirms that GitHub is ahead of what is installed.
        self.btn_update = QPushButton(tr("Actualizar"))
        self.btn_update.setCursor(Qt.PointingHandCursor)
        self.btn_update.setStyleSheet(
            f"QPushButton {{ background: {T.WARN}; color: white; border: none; "
            f"border-radius: 4px; padding: 4px 10px; font-size: 9pt; "
            f"font-weight: 600; }}"
            f"QPushButton:hover {{ background: {T.ERR}; }}")
        self.btn_update.setVisible(False)
        self.btn_update.clicked.connect(self._launch_updater)
        tb.addWidget(self.btn_update)
        # If the check has already answered before this rebuild, the
        # notice is put back instead of being lost.
        if self._sha_version_new:
            self._notify_version_new(self._sha_version_new)
        root.addWidget(topbar)

        # ── Central scroll ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        root.addWidget(scroll, 1)

        page = QWidget()
        scroll.setWidget(page)
        body = QVBoxLayout(page)
        body.setContentsMargins(48, 32, 48, 40)
        body.setSpacing(28)

        # ── Hero ──
        hero = QHBoxLayout()
        hero.setSpacing(36)

        left = QVBoxLayout()
        left.setSpacing(10)
        kicker = QLabel(tr("● PLATAFORMA DE ANÁLISIS"))
        kicker.setStyleSheet(
            f"color: {T.ACCENT_D_TX}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        left.addWidget(kicker)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        t1 = QLabel(tr("Poly-X"))
        t1.setStyleSheet(f"color: {T.INK}; font-size: 38pt; font-weight: 700; border: none;")
        t2 = QLabel(tr("analytics"))
        t2.setStyleSheet(
            f"color: {T.ACCENT_TX}; font-size: 38pt; font-weight: 300; border: none;"
        )
        title_row.addWidget(t1)
        title_row.addWidget(t2)
        title_row.addStretch(1)
        left.addLayout(title_row)

        subtitle = QLabel(tr("Plataforma de detección y clasificación de microplásticos"))
        subtitle.setStyleSheet(
            f"color: {T.INK2}; font-size: 13pt; font-weight: 500; border: none;"
        )
        left.addWidget(subtitle)

        descr = QLabel(
            tr("Detección automatizada de PET, PP y LDPE por fluorescencia Nile Red (254 nm) "
               "e inteligencia artificial. Entrenamiento, etiquetado, detección y reporte en un mismo flujo.")
        )
        descr.setWordWrap(True)
        descr.setStyleSheet(f"color: {T.INK3}; font-size: 10.5pt; border: none;")
        left.addWidget(descr)

        # Authorship credit (just below the descriptor)
        author = QLabel(tr("Diseñado y desarrollado por <b>Cristofher Ferrada</b> · Dr (c) en Química, 2026"))
        author.setTextFormat(Qt.RichText)
        author.setStyleSheet(
            f"color: {T.ACCENT_D_TX}; font-size: 9.5pt; border: none; "
            f"background: {T.BG_SOFT}; padding: 6px 10px; border-radius: 4px;"
        )
        left.addWidget(author)
        left.addSpacing(4)

        # 4 chips
        chips = QHBoxLayout()
        chips.setSpacing(12)
        chips.addWidget(StatChip("3", tr("Polímeros"), "PET · PP · LDPE", T.ACCENT))
        chips.addWidget(StatChip("254", "nm", tr("fluorescencia Nile Red"), T.OK))
        chips.addWidget(StatChip("YOLO", "v8 / v11", "deep learning", T.VIO))
        chips.addWidget(StatChip("μm", tr("medición"), tr("calibración por píxel"), T.WARN))
        left.addLayout(chips)

        hero.addLayout(left, 3)
        hero.addWidget(MicroscopePanel(), 0, Qt.AlignTop)
        body.addLayout(hero)

        # ── `MÓDULOS` section ──
        mod_kicker = QLabel(tr("MÓDULOS"))
        mod_kicker.setStyleSheet(
            f"color: {T.INK2}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        body.addWidget(mod_kicker)

        grid = QGridLayout()
        grid.setSpacing(16)
        grid.addWidget(ModuleCard(
            1, "detector", tr("Detector"),
            tr("Analiza imágenes con un modelo .pt entrenado. Genera salidas anotadas, "
               "CSV con centroides y diámetros, métricas globales e informe HTML de detección."),
            T.ACCENT, self.open_detector, "Alt+1"), 0, 0)
        grid.addWidget(ModuleCard(
            2, "entrenador", tr("Entrenador"),
            tr("Entrena modelos YOLO v8 / v11. Curvas en vivo, recomendaciones automáticas "
               "de calidad y comparación con runs anteriores."),
            T.OK, self.open_trainer, "Alt+2"), 0, 1)
        grid.addWidget(ModuleCard(
            3, "etiquetador", tr("Etiquetador"),
            tr("Anota imágenes en formato YOLO. Soporta pre-anotación con un modelo "
               "existente y atajos de teclado, ahorra ~80 % del tiempo manual."),
            T.WARN, self.open_labeler, "Alt+3"), 1, 0)
        grid.addWidget(ModuleCard(
            4, "visor", tr("Visor"),
            tr("Inspección de una imagen a la vez con calibración interactiva μm/píxel "
               "(línea o círculo) y medición precisa por partícula."),
            T.VIO, self.open_viewer, "Alt+4"), 1, 1)
        body.addLayout(grid)

        # ── Footer ──
        body.addStretch(1)
        body.addWidget(HLine())
        foot = QHBoxLayout()
        st = QLabel(tr("● Listo"))
        st.setStyleSheet(f"color: {T.OK_TX}; font-size: 9.5pt; font-weight: 600;")
        foot.addWidget(st)
        foot.addSpacing(20)
        copyr = QLabel("© Cristofher Ferrada · Poly-X v" + __version__)
        copyr.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        foot.addWidget(copyr)
        foot.addStretch(1)
        readme_btn = QPushButton(tr("LÉAME"))
        readme_btn.setIcon(iconos.icon("texto", 15, T.INK2, 2.0))
        readme_btn.setCursor(Qt.PointingHandCursor)
        readme_btn.clicked.connect(self.open_readme)
        foot.addWidget(readme_btn)
        manual_btn = QPushButton(tr("Manual de usuario"))
        manual_btn.setIcon(iconos.icon("manual", 15, T.INK2, 2.0))
        manual_btn.setCursor(Qt.PointingHandCursor)
        manual_btn.clicked.connect(self.open_manual)
        foot.addWidget(manual_btn)
        body.addLayout(foot)

        # Keyboard shortcuts for the four modules. People who use the program
        # daily always open the same one, and reaching it with Alt+1 saves the trip to the
        # mouse every time.
        for tecla, action in (("Alt+1", self.open_detector),
                              ("Alt+2", self.open_trainer),
                              ("Alt+3", self.open_labeler),
                              ("Alt+4", self.open_viewer),
                              ("Ctrl+,", self._open_preferences)):
            QShortcut(QKeySequence(tecla), self, activated=action)

    # ── Preferences ────────────────────────────────────────────
    def _open_preferences(self):
        language_before = language()
        dlg = PreferencesDialog(self)
        dlg.theme_changed.connect(lambda _: self._build_ui())
        dlg.exec()
        if language() != language_before:
            # The language cannot be applied hot: each widget fixed its
            # text when it was built. The user is warned once, on close, and not on every
            # dialog click.
            QMessageBox.information(
                self, tr("Idioma"),
                tr("El idioma se aplicará al reabrir Poly-X. Los módulos que "
                   "abras desde ahora ya salen en el idioma nuevo."))

    # ── Updates ────────────────────────────────────────────────
    def _notify_version_new(self, sha_short: str):
        """Shows the update button when GitHub is ahead."""
        self._sha_version_new = sha_short
        self.btn_update.setText(tr("Actualizar") + f" · {sha_short}")
        self.btn_update.setToolTip(
            tr("Hay una versión nueva disponible en GitHub. "
               "Pulsa para descargarla e instalarla."))
        self.btn_update.setVisible(True)

    def _launch_updater(self):
        if not can_update():
            QMessageBox.information(
                self, tr("Actualizar"),
                tr("No se encontró {} en la carpeta de instalación. Descarga "
                   "la versión nueva manualmente desde GitHub.").format(
                       UPDATER.name))
            return
        if QMessageBox.question(
                self, tr("Actualizar"),
                tr("Se descargará la versión nueva y se cerrará Poly-X.\n\n"
                   "Tus modelos, resultados y datos no se tocan. ¿Continuar?"),
        ) != QMessageBox.Yes:
            return
        # The updater overwrites program files; Poly-X has to be released
        # first so it does not clash with the .py files that are in use.
        if not launch_updater(UPDATER):
            QMessageBox.warning(
                self, tr("Actualizar"),
                tr("No se pudo lanzar {}. Ejecútalo a mano desde la carpeta "
                   "de instalación.").format(UPDATER.name))
            return
        self.close()

    # ── Launchers ──────────────────────────────────────────────
    def open_detector(self): self._launch_module("polyx.gui.detector")
    def open_trainer(self):  self._launch_module("polyx.gui.trainer")
    def open_labeler(self):  self._launch_module("polyx.gui.etiquetador")
    def open_viewer(self):   self._launch_module("polyx.gui.visor")

    def _launch_module(self, dotted: str, fallback: str = ""):
        """Spawns the module in a separate process so the launcher does not block.

        Delegates to _spawn_module, which does not raise: False means
        the module does not exist or that its process failed to start,
        and here the user has to be warned.
        """
        if not _spawn_module(dotted):
            self._toast(fallback or f"No se pudo abrir el módulo {dotted}")

    def open_manual(self):
        manual = Path(__file__).resolve().parents[2] / "Manual_PolyX.html"
        if manual.exists():
            import webbrowser
            webbrowser.open(manual.as_uri())
        else:
            self._toast("Manual_PolyX.html no encontrado")

    def open_readme(self):
        readme = Path(__file__).resolve().parents[2] / "LEEME.txt"
        if readme.exists():
            open_in_system(str(readme))
        else:
            self._toast("LEEME.txt no encontrado")

    def _toast(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, tr("Poly-X"), msg)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Before the first `show()`: the taskbar button is created
    # with whatever identity exists at that moment.
    marca.identify(app, "launcher")
    # Startup JSONL trace (`app.jsonl`); it never prevents the window from opening.
    _register_boot()
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
