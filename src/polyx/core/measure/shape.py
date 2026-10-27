from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from .morphology_constants import *
from .geodesic import length_geodesic, path_geodesic
from .segmentation import segment, box_margin, _cutout, feret_maximum


class Morphology:
    """Shape and size of a particle, measured on its mask."""

    ok: bool = False
    # px
    area_px: float = 0.0
    perimeter_px: float = 0.0
    length_px: float = 0.0  # the reported size: max(feret, geodesico)
    width_px: float = 0.0
    cuerda_px: float = 0.0  # straight extent (minAreaRect)
    feret_px: float = 0.0  # greatest distance between two edge points
    geodesic_px: float = 0.0  # longest path INSIDE the particle
    # Length of the rectangle with equal area and perimeter. It is NOT "the length of the
    # particle": it is that of an equivalent rectangle, and a jagged edge inflates
    # the perimeter and with it this figure. It is kept because, measured against feret and
    # geodesico, it tells how irregular or filamentous the particle is.
    length_rect_eq_px: float = 0.0
    # um, if calibrated
    area_um2: float | None = None
    length_um: float | None = None
    width_um: float | None = None
    feret_um: float | None = None
    geodesic_um: float | None = None
    equivalent_rect_length_um: float | None = None
    # dimensionless
    aspect: float = 0.0
    circularity: float = 0.0
    curvature: float = 1.0  # length / chord; 1.0 = straight
    filling: float = 0.0  # particle area / its box area
    solidity: float = 1.0  # area / its convex hull area
    morphotype: str = INDETERMINATE
    method: str = ""  # where the length came from
    # What fraction of the Feret line falls OUTSIDE the particle. It is only
    # filled in by `dibujar_medicion()`, which is what draws it. On a concave
    # particle it is nonzero and it is not an error: Feret measures the separation of
    # two jaws, not a path through the inside.
    feret_outside: float = 0.0
    warning: str = ""
    # The measurement comes out, but it asks for a human look. It is flagged instead of discarded:
    # silently discarding loses real particles from the count, and giving the number
    # without warning puts sizes in the table that do not hold up.
    review: bool = False

    @property
    def curve(self) -> bool:
        return self.curvature >= CURVE_FROM




def extremes_feret(contour: np.ndarray):
    """The two farthest-apart points of the contour, for drawing."""
    if contour is None or len(contour) < 2:
        return None
    env = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(env) < 2:
        return None
    d = env[:, None, :] - env[None, :, :]
    D = np.sqrt((d**2).sum(-1))
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return tuple(env[i].astype(int)), tuple(env[j].astype(int))


def _choose_length(feret, geo, thinness, solidity):
    """Pick between the geodesic and the maximum Feret as the reported length.

    The geodesic only wins on a thin, non-convex particle; otherwise the
    Feret (the tighter bound) is reported.
    """
    if thinness >= THINNESS_FOR_GEODESIC and solidity < SOLIDITY_CONVEX and geo >= feret:
        return geo, "geodesico"
    return feret, "Feret maximo"


def measure(mask: np.ndarray, um_por_px: float | None = None) -> Morphology:
    """Size and shape from the mask of a particle."""
    m = Morphology()
    if mask is None or not mask.any():
        m.warning = "sin mascara"
        return m

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        m.warning = "sin contorno"
        return m
    c = max(contours, key=cv2.contourArea)

    # The area is counted on the mask and not with `contourArea`: the polygon
    # formula underestimates on particles of few pixels, which are the majority here.
    area = float(np.count_nonzero(mask))
    # The contour is traversed pixel by pixel, WITHOUT smoothing. Smoothing it
    # with `approxPolyDP` to remove the digitization staircase was tried against
    # synthetic shapes of known size, and it makes things worse: with epsilon 1 px the median error
    # rises from 1.0% to 2.4% and the worst from 3.1% to 7.8%, because it cuts the particle's
    # real corners along with the staircase. Estimating the
    # length as area/thickness with the distance transform, immune to the
    # perimeter, was also tried, and it is even worse: 26.6% median error.
    perim = float(cv2.arcLength(c, True))
    if area < 4 or perim <= 0:
        m.warning = "particula demasiado pequena para medir su forma"
        return m

    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    cuerda = float(max(rw, rh))
    thickness_straight = float(min(rw, rh))  # noqa: F841

    # Rectangle model: valid only if a rectangle with that area and
    # that perimeter exists. On a compact particle the discriminant is negative.
    # Equal-area-and-perimeter rectangle. A descriptor, NOT the size: it only exists
    # if P^2 >= 16A, which is to say the particle is elongated
    # enough -- a circle gives P^2 = 4*pi*A < 16A and has no
    # equivalent rectangle. Measured against feret and geodesico it tells whether the particle is
    # irregular: if it is much larger than them, the edge is jagged.
    disc = perim * perim - 16.0 * area
    length_rect_eq = (perim + float(np.sqrt(disc))) / 4.0 if disc > 0 else 0.0

    # ── The reported length ──
    # Maximum Feret and geodesic diameter, and the larger is taken. For a connected
    # shape the geodesic is ALWAYS greater than or equal to Feret -- on a convex one
    # they coincide --, so when Feret comes out larger it is only chamfer
    # discretization error, and keeping the larger takes the tighter
    # of the two bounds.
    #
    # This is preferred over the rectangle model, measured against shapes of known
    # size, because the model depends on the perimeter and a jagged edge
    # inflates it: on a straight line with saw teeth it gave +22.5%. max(Feret,
    # geodesico) has a somewhat larger median error (3.6% vs 1.2%) but its
    # worst case is 4.7% instead of 22.5%, and here it matters more not to fail ugly.
    feret = feret_maximum(c)
    geo = length_geodesic(mask)

    # The thickness decides which of the two applies. Only on a thin particle does
    # the geodesic path follow its shape; on a thick one it goes around the concavities and
    # returns a number larger than its real extent.
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thickness = 2.0 * float(dt.max())
    thinness = geo / thickness if thickness > 0 else 0.0

    # Solidity: how far it departs from its own convex hull. A bent
    # fiber leaves a lot of empty space inside the hull; a straight bar leaves none.
    envelope = cv2.contourArea(cv2.convexHull(c))
    solidity = area / envelope if envelope > 0 else 1.0

    length, method = _choose_length(feret, geo, thinness, solidity)

    # The width is that same maximum inscribed thickness: the diameter of the largest
    # circle that fits inside the particle.
    #
    # area/length is not used, which would be the mean width of a ribbon, because on a
    # compact particle it does not give its diameter: a disk has A/L = pi*d/4 = 0.785d,
    # so a perfect circle came out with aspect 1.27 instead of 1.0 and the
    # fiber/fragment classification started out biased.
    width = thickness

    # How far it departs from a straight line. With the geodesic this no longer depends on the
    # perimeter: it is how much longer the path inside is than the chord.
    curvature = length / cuerda if length > 0 and cuerda > 0 else 1.0

    m.ok = True
    m.area_px = area
    m.perimeter_px = perim
    m.length_px = length
    m.width_px = max(width, 0.0)
    m.cuerda_px = cuerda
    m.feret_px = feret
    m.geodesic_px = geo
    m.length_rect_eq_px = length_rect_eq
    m.aspect = length / width if width > 0 else 0.0
    m.circularity = float(4.0 * np.pi * area / (perim * perim))
    m.curvature = curvature
    m.filling = area / (rw * rh) if rw * rh > 0 else 0.0
    m.solidity = solidity
    m.method = method
    m.morphotype = FIBER if m.aspect >= ASPECT_FIBER else FRAGMENT

    if um_por_px and um_por_px > 0:
        m.area_um2 = area * um_por_px * um_por_px
        m.length_um = length * um_por_px
        m.width_um = m.width_px * um_por_px
        m.feret_um = feret * um_por_px
        m.geodesic_um = geo * um_por_px
        m.equivalent_rect_length_um = length_rect_eq * um_por_px if length_rect_eq else None
    return m


def measure_detection(
    bgr: np.ndarray,
    det,
    um_por_px: float | None = None,
    margin: int | None = None,
    mask: np.ndarray | None = None,
) -> Morphology:
    """Segments the box of a ``Detection`` and measures it. Never raises.

    A segmentation failure on one particle cannot take down the batch: a
    Morfologia with ``ok`` False is returned and the caller decides whether to
    fall back to the box size.
    """
    try:
        record: dict = {}
        if mask is None:
            mask = getattr(det, "mask", None)
        if mask is None:
            mask = segment(bgr, det.x1, det.y1, det.x2, det.y2, margin, record=record)
            if mask is None:
                m = Morphology()
                m.warning = "no se pudo separar la particula del fondo"
                return m
        else:
            H, W = bgr.shape[:2]
            if mask.shape[:2] == (H, W):
                if margin is None:
                    margin = box_margin(det.x2 - det.x1, det.y2 - det.y1)
                sub, ox, oy = _cutout(bgr, det.x1, det.y1, det.x2, det.y2, margin)
                if sub is None:
                    m = Morphology()
                    m.warning = "no se pudo separar la particula del fondo"
                    return m
                mask = mask[oy : oy + sub.shape[0], ox : ox + sub.shape[1]]
            mask = (mask > 0).astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
            if not mask.any():
                m = Morphology()
                m.warning = "no se pudo separar la particula del fondo"
                return m
        m = measure(mask, um_por_px)
        diagonal = float(np.hypot(det.x2 - det.x1, det.y2 - det.y1))
        # The warning looks at the mask BEFORE the crop, not after.
        #
        # After the crop the measurement is already inherently bounded, so
        # checking it there cannot reveal anything: it would be a warning that
        # never fires. What says whether this particle is trustworthy is how far it
        # WENT OUT -- if the mask went to twice its box, the threshold caught
        # something that was not this particle, and what is reported is the piece that
        # stayed inside the box, not the whole particle.
        previous = float(record.get("largo_sin_recortar_px", 0.0) or 0.0)
        if m.ok and diagonal > 0 and previous > DRAIN_X_DIAGONAL * diagonal:
            m.review = True
            m.warning = (m.warning + " | " if m.warning else "") + (
                f"la mascara se salia {previous / diagonal:.1f}x la diagonal de su "
                f"caja y se recorto a ella: puede ser una vecina pegada, y la "
                f"talla se queda corta si la particula seguia de verdad"
            )
        return m
    except cv2.error as e:
        m = Morphology()
        m.warning = f"error de OpenCV al segmentar: {e}"
        return m


def apply_to_detection(
    det, bgr: np.ndarray, um_por_px: float | None = None, mask: np.ndarray | None = None
) -> bool:
    """Measures the particle and writes the result into the ``Detection``.

    Returns whether it could be measured. When not, the shape fields stay
    None and the caller must fall back to the box size: losing the particle
    from the count is worse than reporting it with an approximate size and
    saying so.

    ``diam_um`` becomes the diameter of the equal-area circle of the **real**
    particle, not of the box. It is the same quantity as always but properly
    computed, so the size filters and the report histogram keep working
    untouched, now on the correct number.
    """
    m = measure_detection(bgr, det, um_por_px, mask=mask)
    if not m.ok:
        return False
    det.aspect = m.aspect
    det.curvature = m.curvature
    det.morphotype = m.morphotype
    det.method_length = m.method
    # The warning is copied to the `Detection` so the report can show it. Without
    # this, a size inflated by two touching particles reached the document
    # as a clean number, indistinguishable from a good measurement.
    det.review = m.review
    det.warning_shape = m.warning or None
    if um_por_px and um_por_px > 0:
        det.area_um2 = m.area_um2
        det.length_um = m.length_um
        det.width_um = m.width_um
        det.feret_um = m.feret_um
        det.geodesic_um = m.geodesic_um
        det.equivalent_rect_length_um = m.equivalent_rect_length_um
        det.diam_um = 2.0 * float(np.sqrt(m.area_um2 / np.pi)) if m.area_um2 else None
    return True

__all__ = ['Morphology', 'feret_maximum', 'extremes_feret', 'measure', 'measure_detection', 'apply_to_detection']
