"""Microscope animation panel for the Poly-X launcher."""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from polyx.gui import theme as T


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

        rnd = random.Random(11)  # noqa: S311
        classes = ("PET", "PP", "LDPE")
        planes = (0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2)  # 4 background, 5 middle, 4 foreground

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
        for plane, (dx, dy), class_id in zip(planes, sites, shift):  # noqa: B905
            scale = (0.55, 0.82, 1.0)[plane]
            self._particles.append(
                {
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
                }
            )

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
        return (
            part["x"] + part["ax"] * math.sin(part["wx"] * t + part["fx"]),
            part["y"] + part["ay"] * math.sin(part["wy"] * t + part["fy"]),
        )

    def _alpha_box(self, phase: float, y_rel: float) -> float:
        """Opacity of the box of a particle located at height y_rel."""
        if phase < self._START_SWEEP:
            return 0.0
        if phase >= self._START_OUTPUT:
            return max(0.0, 1.0 - (phase - self._START_OUTPUT) / (1.0 - self._START_OUTPUT))
        advance = self._advance_sweep(phase)
        if advance is None:  # the sweep has already crossed the whole field
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
            p.drawLine(
                QPointF(field.left() + i * paso, field.top()),
                QPointF(field.left() + i * paso, field.bottom()),
            )
            p.drawLine(
                QPointF(field.left(), field.top() + i * paso),
                QPointF(field.right(), field.top() + i * paso),
            )

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
        center_2 = QColor(base)
        center_2.setAlpha(intensity)
        half = QColor(base)
        half.setAlpha(int(intensity * 0.30))
        tenue = QColor(base)
        tenue.setAlpha(int(intensity * 0.10))
        edge = QColor(base)
        edge.setAlpha(0)
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
        if shape == 0:  # rounded fragment
            p.drawEllipse(QPointF(x, y), r, r * 0.92)
        elif shape == 1:  # fibre
            p.save()
            p.translate(x, y)
            p.rotate(part["giro"] + 9 * math.sin(0.35 * t + part["fx"]))
            p.drawRoundedRect(QRectF(-r * 1.9, -r * 0.24, r * 3.8, r * 0.48), r * 0.24, r * 0.24)
            p.restore()
        else:  # irregular fragment
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
        text = f"{part['clase']} {part['conf']:.2f}"
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
