from __future__ import annotations

import cv2
import numpy as np
from .segmentation import segment, box_margin
from .shape import measure, extremes_feret
from .geodesic import path_geodesic


def _draw_feret(vis: np.ndarray, mask: np.ndarray, p, q, color) -> float:
    """Draws the Feret line: solid inside the particle, dashed outside.

    Returns what fraction of the segment falls OUTSIDE the mask, so it can be
    stated with a number instead of left to the eye.

    It is drawn this way because Feret is the separation of two calipers, not
    a path: on a concave particle the segment goes outside, and that is
    correct. As a continuous line it looked like a measurement error.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    length = float(np.hypot(*(q - p)))
    if length < 1:
        return 0.0
    n = int(length) + 1
    ts = np.linspace(0.0, 1.0, n)
    pts = p[None, :] + ts[:, None] * (q - p)[None, :]
    xs = np.clip(pts[:, 0].round().astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(pts[:, 1].round().astype(int), 0, mask.shape[0] - 1)
    inside = mask[ys, xs] > 0
    for k in range(n - 1):
        # Outside the particle, one pixel out of three: the dashed stroke is
        # what distinguishes "here the jaw does not touch" from "here it was measured".
        if not inside[k] and (k % 3):
            continue
        cv2.line(vis, (xs[k], ys[k]), (xs[k + 1], ys[k + 1]), color, 1)
    return float((~inside).mean())


def draw_measurement(bgr: np.ndarray, det, zoom: int = 4):
    """Crop of the particle with the measurement applied to it, drawn on top.

    Returns (imagen, Morfologia) or (None, None). The image carries the mask
    contour and, on top, the Feret line or the geodesic path depending on
    which of the two decided the size, so it is visible EXACTLY what that
    particle was measured on and nobody has to take it on faith.
    """
    try:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        mask = segment(bgr, x1, y1, x2, y2)
        if mask is None:
            return None, None
        m = measure(mask)
        # Same margin that `segmentar()` used to crop -- it must be
        # EXACTLY `margen_de_caja()`, not a separate number. Here there was a 0.35
        # hand-copied that did not match the real 0.50 of `segmentar()`: the
        # mask was computed on one crop and the displayed background was read
        # from another origin, so the contour and the photo below ended up
        # misaligned by a few px (seen at high zoom). It was seen by eye in a
        # report: the mask "looked shifted" relative to the particle.
        margin = box_margin(x2 - x1, y2 - y1)
        ox, oy = max(0, int(x1) - margin), max(0, int(y1) - margin)
        sub = bgr[oy : oy + mask.shape[0], ox : ox + mask.shape[1]].copy()
        if sub.size == 0:
            return None, None

        vis = sub.copy()
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cont, -1, (0, 220, 0), 1)
        if m.ok:
            if m.method.startswith("geodesico"):
                cam = path_geodesic(mask)
                for k in range(1, len(cam)):
                    cv2.line(vis, cam[k - 1], cam[k], (255, 80, 255), 1)
            elif cont:
                ext = extremes_feret(max(cont, key=cv2.contourArea))
                if ext:
                    outside = _draw_feret(vis, mask, ext[0], ext[1], (0, 210, 255))
                    for q in ext:
                        cv2.circle(vis, q, 2, (0, 210, 255), -1)
                    m.feret_outside = outside
        par = np.hstack([sub, vis])
        if zoom > 1:
            par = cv2.resize(par, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        return par, m
    except Exception as exc:  # noqa: BLE001
        try:
            from polyx.core.provenance.observability import logger_for

            logger_for("detector").warning("draw_measurement failed: %s", exc)
        except Exception:  # noqa: BLE001, S110
            pass
        return None, None

__all__ = ['_draw_feret', 'draw_measurement']
