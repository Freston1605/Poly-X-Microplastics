"""Measures the actual particle, not its box.

A detector box is four numbers and does not describe shape. For an elongated
particle lying diagonally the box is nearly empty: a 2000x20 um fiber at 45
degrees has a 1414x1414 box, whose area is 50 times that of the fiber. And the
worst part is not the magnitude but that **the same particle rotated gives a
different number**, so the measurement is not a measurement.

Here the content is segmented inside the box -- which already says where to
look, so no mask annotation or retraining is needed -- and the magnitudes come
from the mask.

HOW THE LENGTH IS MEASURED
--------------------------
Two bounds are computed and the larger one is reported:

  * MAXIMUM FERET. The greatest distance between two edge points, computed on
    the convex hull. It is the correct measurement of apparent length and of a
    STRAIGHT fiber, and a jagged edge does not affect it. But on a curved fiber
    it gives the chord: on a 180-degree arc it falls 35% short.

  * GEODESIC DIAMETER. The longest path that fits INSIDE the particle, by
    double propagation. Being unable to leave the mask, it follows the curve
    and returns its length. On a compact particle the longest path is the
    straight one and coincides with Feret.

The geodesic is used only if the particle is THIN (length/width >= 4) and NOT
CONVEX (solidity < 0.90). Both conditions come from observed failures:

  * On a thick particle, any concavity makes the path go AROUND it instead of
    through it. A real 44 px clump with one notch measured 73 px.
  * On a convex shape the longest path through the inside is the straight one,
    so geodesic and Feret coincide by definition. If the geodesic comes out
    larger it is a chamfer discretization error, which is not isotropic: a
    straight bar measured 200 px at 0 and 45 degrees on average but 208 at 15,
    30 and 60 -- exactly the orientation dependence the method was meant to
    remove.

With both conditions the median error against known-size shapes drops to 0.6%
and the worst case stays at 4.7%.

WHAT WAS DISCARDED, AND WHY
---------------------------
The equal-area-and-perimeter rectangle, L = (P + sqrt(P^2-16A))/4, was used as
a size and does NOT work for that, although it is kept as a descriptor:

  * It depends on the perimeter, and a jagged edge inflates it. Measured on a
    straight line with saw teeth of known size: +22.5%.
  * It only exists if P^2 >= 16A. A circle gives P^2 = 4*pi*A < 16A and has no
    equivalent rectangle.
  * It is not "the length of the particle" but that of a rectangle with the
    same area and the same perimeter, which is another thing.

It is kept in ``largo_rect_eq`` because, compared with Feret and the geodesic,
it tells how irregular the particle is: when it is much larger than the other
two, the edge is jagged.

The rectangle model gave 1.2% median error but 22.5% in the worst case, and
here it matters more not to fail ugly than to be precise in the good case.

All of this is pinned in tests/test_morfologia.py against known-size shapes,
including the rotated bar that exposed the convexity issue.

PARTICLES IN CONTACT
--------------------
Two touching particles form a single connected component. They are separated
by watershed over the distance transform in ``separar_pegadas()``, up to 27%
diameter overlap.

KNOWN LIMITS
------------
  * Beyond 40% overlap there is no neck left to cut through and both particles
    keep being measured together.
  * On a very coiled fiber the geodesic path cuts through each bend and
    underestimates: on a fiber of tight closed curves it falls 19% short.
    ``largo_rect_eq`` is closer to truth in that specific case (-5%), so when
    the two figures disagree strongly the particle deserves a look.

Only cv2 and numpy are used. Not scipy -- which is not a declared dependency
and only arrives indirectly with ultralytics -- nor skimage nor
opencv-contrib: adding dependencies is exactly what breaks the install on
another machine."""

from __future__ import annotations

from .morphology_constants import *

from dataclasses import dataclass

import cv2
import numpy as np

# Morphotypes. Cutoff at 3 on the REAL aspect ratio (not the box's): it is the cut
# normally used to separate fiber from fragment.


# The geodesic diameter is used only if the particle is at least this thin,
# measured as length/thickness. On a thick particle any concavity makes
# the geodesic path go AROUND it instead of through it, and then it inflates: it was
# seen on a real particle 44 px long and 22 thick to which the
# geodesic gave 73 px by going around a notch.
#
# The threshold is not delicate: on shapes of known size, clumps stay at
# 1.05 and 1.64 thinness and the fattest fiber at 6.7, so any value
# between 3 and 6 separates the same. With it, the median length error drops from 3.1% to
# 0.8% -- a circle goes from 104 to exactly 100 -- without touching any fiber.

# The particle must ALSO be NON-CONVEX for the geodesic to add
# anything. On a convex shape the longest path inside is the straight one, so
# geodesic and Feret coincide BY DEFINITION; if the geodesic comes out larger it is only
# chamfer discretization error, which is not isotropic and deviates more at
# oblique angles. It was seen on a rotated straight bar: at 0 and 45 degrees it measured
# 200 px, but at 15, 30 and 60 it gave 208, a 4% that appeared and disappeared with the
# rotation -- exactly the defect that was to be removed.
#
# Solidity (area / convex hull area) separates them unambiguously
# on shapes of known size: bars and circles stay above 1.00,
# arcs between 0.27 and 0.71.

# Fraction of the maximum distance to the background above which a pixel is
# considered a CORE of a particle. Two touching particles have two
# cores even if they form a single connected component, and the watershed separates them
# at the neck that joins them.
#
# 0.80 comes from sweeping the threshold over shapes of known size. It separates two
# 60 px circles with overlaps of 2, 6, 10 and 16 px -- up to 27% of the
# diameter -- and does not split ANY healthy particle: not a circle, not a bar, not
# a curved fiber, not a four-lobed irregular one. Raising it to 0.90 makes
# the curved fiber burst into 27 pieces, so the margin is wide below and
# narrow above.

# Above this the particle is considered curved and the chord length
# stops working as its size. 1.15 = the contour travels 15% more than the straight line.


@dataclass
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


def box_margin(box_width: float, box_height: float) -> int:
    """How much background is cropped around the box, in pixels.

    Proportional to the smaller side, with a minimum. The 0.50 is not
    arbitrary: with 0.35 a neighbouring particle that barely grazed ours fell
    almost entirely OUTSIDE the crop, and a truncated neighbour loses its
    core, so the watershed never got to separate them. Measured on two 60 px
    circles, the error goes from +16.3% to -2.6% just by widening the margin;
    from 0.50 on the result no longer changes.

    It is exposed as a function so tests use the same number as the code
    instead of repeating it, which is how they drift apart. For the same
    reason ``dibujar_medicion`` uses it: if each computed its own margin, the
    mask and the photo underneath would be drawn from different origins.

    MEASURED ON THE LONGER SIDE, not the shorter. With the shorter one, the
    crop of a fiber was computed from its WIDTH: for a 72x18 box it gave 9 px
    of margin, and then the working crop did not even contain the whole fiber
    -- it was lost before anything was segmented. On a compact particle both
    sides are similar and the number does not change.
    """
    return int(max(3, round(0.50 * max(max(1.0, box_width), max(1.0, box_height)))))  # noqa: PLW3301


def _cutout(bgr: np.ndarray, x1, y1, x2, y2, margin: int):
    """Crops the box with a margin, and also returns the offset.

    The margin is not cosmetic: to separate the particle from the background
    by thresholding, the crop must contain background. A box hugging the
    particle edge is almost all particle, and Otsu would split the particle in
    two instead of separating it from the background.
    """
    H, W = bgr.shape[:2]
    ax = max(0, int(x1) - margin)
    ay = max(0, int(y1) - margin)
    bx = min(W, int(np.ceil(x2)) + margin)
    by = min(H, int(np.ceil(y2)) + margin)
    if bx - ax < 3 or by - ay < 3:
        return None, 0, 0
    return bgr[ay:by, ax:bx], ax, ay


# From this elongation -- length/width of the minimum rectangle that
# contains it -- a mask is treated as elongated, and that changes two decisions:
# it is not split by watershed, and it is allowed to overflow the box further.
#
# 3.0 is not a number picked from nowhere: it is the same cutoff with which
# `ASPECTO_FIBRA` separates fiber from particle, and it also leaves out the case that
# motivated the crop to the box. Two circles of 20 px radius joined by a
# bridge form a set of about 90x40 px -> elongation 2.25, below the
# threshold, so that case is still cropped and separated as before.

# How far the mask is allowed to overflow the detector box, as a
# fraction of one side of the box. They are two numbers because the detector error
# differs in each case.
#
#   COMPACT: fraction of the SHORTER side. The box of a squat particle is
#   good, and the slack only covers the fact that the detector usually fits a hair
#   inside the real edge.
#
#   ELONGATED: fraction of the LONGER side. On a fiber the shorter side is its thickness,
#   so a fraction of it never reaches along the length no matter how much it is
#   enlarged: on a 36 px fiber boxed in 29 px, 10% of the thickness gives
#   2 px and still cuts it. With the longer side, 25% gives 7 px and recovers it.
#
# THE FIX IS THAT THE SECOND CASE EXISTS. Previously an elongated mask was
# NOT cropped at all, and that "at all" was the bug: a leak -- the
# particle plus a bright ridge leaving the box -- is also long and
# thin, so it scored as elongated and disabled the only protection
# that would have stopped it. In the report three of six records came out measuring
# 1.9 to 2.1 times the diagonal of their box, one of them 6.4 mm: more than the
# 5 mm limit that defines a microplastic.
#
# Elongation cannot tell a fiber from a leak -- both are long
# and thin -- so it is not asked to. The BOX is asked to, since it is
# the only thing that knows which particle we are talking about. The 0.25 comes from the fact that the 98
# hand-annotated particles do not exceed 1.41 times the diagonal of their box.
# Minimum brightness difference between the box and the ring around it to
# accept that there is a particle there. Below it None is returned instead of
# measuring: the 90th percentile of the box against the median of the ring, in
# 0-255 levels.


# Threshold above which the mask is considered to have leaked. The length of the
# UNCROPPED mask is measured against the diagonal of the box. 1.5 comes from the
# data: over 98 hand-annotated particles, none exceeds 1.41.


def _elongation(mask: np.ndarray) -> float:
    """How many times longer than thick the mask is. 1.0 when it cannot be computed.

    It is computed as ``area / grosor**2``, with the width taken from the
    largest inscribed circle (distance transform). For a shape of length L and
    width w the area is of the order of L*w, so the ratio gives L/w.

    The minimum-area rectangle is NOT used, which would be the obvious choice,
    because **it fails exactly on curved fibers**: the rectangle wrapping an
    arc is nearly square and gives an elongation close to 1. With that
    measurement, the curved fiber of plate 10.3 -- the one seen truncated in
    the report -- was classified as compact and kept being cropped.

    Depending only on area and width, this version does not change with
    curvature or orientation.
    """
    if mask is None or not mask.any():
        return 1.0
    area = float(np.count_nonzero(mask))
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    thickness = 2.0 * float(dt.max())
    if thickness <= 0:
        return 1.0
    return area / (thickness * thickness)


def separate_glued(mask: np.ndarray, seed) -> np.ndarray:
    """If the mask joins several particles, returns only the seed's one.

    Two particles in contact form a single connected component, and measuring
    them together adds up their sizes: it was observed on real plates and it
    hits mainly the large end of the distribution, which is exactly the end
    reported as "the largest particle".

    They are separated by watershed over the distance transform, the standard
    method for touching objects in microscopy. The idea is that the centre of
    each particle is far from the background and the neck joining them is
    close, so thresholding the distance yields two separated cores even when
    the mask is a single piece; the watershed grows from them and cuts right
    through the neck.

    When there is only one core nothing is touched: the overwhelming majority
    of particles are isolated and there is no point risking a cut where there
    is nothing to cut.
    """
    if mask is None or not mask.any():
        return mask

    # A clearly elongated mask is not split. A fiber of irregular
    # thickness has SEVERAL thick zones, and each one passes the core threshold
    # and becomes a watershed center: then the fiber is cut at
    # its own thinning, as if they were two particles in contact. It was
    # measured on a real fiber from plate 10.3: of 369 px of area 254
    # remained, 31% less, and the length fell with it.
    #
    # The 0.80 threshold was calibrated on TOUCHING CIRCLES, which is the case
    # this function exists for. A fiber is not that case.
    #
    # The risk of not splitting -- two fibers that really touch stay as
    # one -- is already covered by `medir_deteccion()`, which flags for review any
    # particle longer than 1.5 times the diagonal of its box. Between underestimating
    # silently and overestimating with a warning, warning is the right choice.
    if _elongation(mask) >= ELONGATION_ELONGATED:
        return mask

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dt.max() < 2:
        return mask

    _, cores = cv2.threshold(dt, THRESHOLD_CORE * dt.max(), 255, cv2.THRESH_BINARY)
    cores = cores.astype(np.uint8)
    n, marks = cv2.connectedComponents(cores)
    if n <= 2:  # background + one core: isolated particle
        return mask

    # `cv2.watershed` reserves 0 for "unknown" and 1 for the background.
    marks = marks + 1
    marks[mask == 0] = 1
    marks[(mask > 0) & (cores == 0)] = 0
    try:
        cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), marks)
    except cv2.error:
        return mask

    sx, sy = int(seed[0]), int(seed[1])
    sx = int(np.clip(sx, 0, marks.shape[1] - 1))
    sy = int(np.clip(sy, 0, marks.shape[0] - 1))
    label = marks[sy, sx]
    if label <= 1:
        # The center fell on the dividing line or on the background. The largest
        # region touching the center is kept, rather than returning the whole set.
        neighbours = marks[max(0, sy - 2) : sy + 3, max(0, sx - 2) : sx + 3]
        candidates = [v for v in np.unique(neighbours) if v > 1]
        if not candidates:
            return mask
        label = max(candidates, key=lambda v: int((marks == v).sum()))

    output = ((marks == label) & (mask > 0)).astype(np.uint8) * 255
    return output if output.any() else mask


def segment(
    bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    margin: int | None = None,
    record: dict | None = None,
) -> np.ndarray | None:
    """Mask of the particle inside the box. None if it could not be isolated.

    If ``registro`` is passed, it is annotated there with how far the mask
    spilled out before being cropped to the box. It is the only thing that
    tells a leak apart from a big particle, and only here can it be known:
    after the crop the evidence is gone.

    Under Nile Red the particle emits and the filter background stays dark, so
    the channel with the highest contrast is thresholded with Otsu. The
    connected component covering the centre of the box is taken: neighbouring
    particles may fall partly inside the crop, and the one that matters is the
    one the detector pointed at.
    """
    if bgr is None or bgr.size == 0:
        return None
    # A box that does not cut the image is bad data, not an edge case. Without
    # this guard, the margin reached a corner of the photo and a background
    # measurement was returned as if it were a particle.
    height_img, width_img = bgr.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= width_img or y1 >= height_img:
        return None
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    if margin is None:
        margin = box_margin(box_width, box_height)

    sub, ox, oy = _cutout(bgr, x1, y1, x2, y2, margin)
    if sub is None:
        return None

    # The highest-contrast channel, not always gray: PP and LDPE emit in yellow
    # (R and G high, B low) and PET in red, so the per-pixel maximum separates
    # from the background better than the weighted average used by `BGR2GRAY`.
    canal = sub.max(axis=2)
    canal = cv2.GaussianBlur(canal, (3, 3), 0)

    # The threshold comes from the CONTRAST BETWEEN THE RING AND THE BOX, not from Otsu over
    # the whole crop. Otsu splits the histogram into two classes without knowing where
    # the particle is, and when the crop is almost all particle it ends up
    # putting dark background inside the mask: measured over 2863 particles,
    # the median fraction of the mask that was dark background was 5.4%, and in 7.4%
    # of them it exceeded 20%. That inflated the area systematically.
    #
    # Here we use what the detector box already knows: outside it there is background,
    # inside there is particle. The background level is the median of the ring, the
    # particle level the 90th percentile of the box, and the cut is halfway -- the
    # same technique with which `calibracion.py` finds the plate edge.
    bx1 = int(np.clip(int(x1) - ox, 0, canal.shape[1]))
    by1 = int(np.clip(int(y1) - oy, 0, canal.shape[0]))
    bx2 = int(np.clip(int(np.ceil(x2)) - ox, 0, canal.shape[1]))
    by2 = int(np.clip(int(np.ceil(y2)) - oy, 0, canal.shape[0]))
    ring = np.ones(canal.shape, dtype=bool)
    ring[by1:by2, bx1:bx2] = False
    inside_box = canal[by1:by2, bx1:bx2]

    threshold_2 = None
    if ring.sum() >= 20 and inside_box.size >= 9:
        background = float(np.median(canal[ring]))
        peak = float(np.percentile(inside_box, 90))
        if peak - background >= CONTRAST_MINIMUM:
            threshold_2 = background + 0.5 * (peak - background)
        else:
            # WITHOUT CONTRAST A SIZE IS NOT INVENTED. Here it fell back to Otsu, which
            # with no particle to separate splits the histogram of the NOISE and
            # returns a fragmented mask -- with a normally looking size,
            # indistinguishable from good data unless you look at the image.
            #
            # Measured over 100 hand-annotated boxes: 7 fall here, and the 7
            # are from the two plates annotated in the wrong pixel frame,
            # where the box points to a place without a particle. The cutoff is sharp:
            # those 7 have contrast 2-7 and the next one has 34.
            #
            # Returning None makes the report say "the particle could not be separated from
            # the background" and counts it separately, which is what really
            # happened.
            return None

    if threshold_2 is None:
        _, mask = cv2.threshold(canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = (canal > threshold_2).astype(np.uint8) * 255

    # It is NOT filtered by color, even though that would separate two particles of
    # different polymer that touch. It was tried and measured, and it does not work on this material:
    #
    #   core -> edge of the SAME particle   median 0.197 chromaticity
    #   core -> neighbor of ANOTHER class   median 0.281, p25 0.173
    #
    # The two distributions overlap, so there is no threshold: with 0.10 it cuts
    # 93.7% of the particle's own edges and with 0.26 only 54% of the
    # foreign neighbors are separated. The cause is that PP and LDPE are distinguished by BRIGHTNESS and not by
    # hue, and that the faint edge of any particle shifts its color toward the
    # background. The case is solved by geometry in `separar_pegadas()`, which does not
    # need to tell which polymer each one is.

    # Removes salt noise without closing the gap of a U-shaped particle.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    # Box center in crop coordinates.
    cx = int(round((x1 + x2) / 2 - ox))  # noqa: RUF046
    cy = int(round((y1 + y2) / 2 - oy))  # noqa: RUF046
    cx = int(np.clip(cx, 0, labels.shape[1] - 1))
    cy = int(np.clip(cy, 0, labels.shape[0] - 1))
    idx = labels[cy, cx]
    if idx == 0:
        # The center fell on background -- annular particle or a slightly off-center box.
        # The largest component touching the box is used.
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 4:
        return None
    mask = (labels == idx).astype(np.uint8) * 255

    # Separate two touching particles BEFORE cropping to the box, not
    # after. The crop truncates the neighbor, and a truncated neighbor loses its
    # core: on two touching 60 px circles, uncropped both cores were detected
    # with overlaps up to 16 px, and once cropped neither was detected
    # at all. That is, with the reverse order the separation never got to
    # act.
    mask = separate_glued(mask, (cx, cy))

    # Crop to the box. The margin exists so the threshold has background to
    # measure against, not to measure in: without this crop, a particle stuck
    # to another forms a single connected component with it and the whole set is measured. A
    # mask 1.71 times wider than its box was observed on real plates, and
    # from that came 8.9 mm sizes that no 97 px box can contain.
    #
    # A slack is left: the detector usually fits the box somewhat inside the
    # real edge, and cutting flush would amputate the particle tip.
    #
    # THE SLACK DEPENDS ON HOW ELONGATED THE MASK IS, but IT IS ALWAYS
    # CROPPED. The two halves of that sentence each cost one bug:
    #
    #   * With the fixed 10% slack of the shorter side, FIBERS came out cut
    #     by a straight edge. On a fiber the shorter side is its thickness, so
    #     that fraction never reaches along the length. On two real fibers from
    #     plate 10.3 the crop took 31% and 34% of the area.
    #
    #   * The fix at the time was to crop NOTHING if the mask was
    #     elongated, and that opened the opposite bug. Elongation does not tell
    #     a fiber from a leak: a particle with a bright ridge stuck
    #     to it is also long and thin, so it scored as elongated and turned off
    #     the only protection that would have stopped it. In one report three
    #     of six records came out measuring 1.9 to 2.1 times the diagonal of their box,
    #     one of them 6.4 mm -- above the 5 mm limit that
    #     defines a microplastic.
    #
    # Hence the current form: an elongated mask is given MORE slack, not
    # infinite slack, and measured on the LONGER side so it reaches along the
    # length. The box is the only thing that knows which particle we are talking about; the
    # mask shape cannot say it, because a fiber and a leak have
    # the same shape.
    #
    # The protection still stands where it matters: two 20 px circles joined by
    # a bridge give elongation 1.7 -- squat -- and are cropped as always.
    # It is the case that motivated this crop and from which 8.9 mm sizes came.
    if record is not None:
        record["largo_sin_recortar_px"] = feret_maximum(_contour_elderly(mask))
    if _elongation(mask) < ELONGATION_ELONGATED:
        slack = max(2, int(round(SLACK_COMPACT * min(box_width, box_height))))  # noqa: RUF046
    else:
        slack = max(2, int(round(SLACK_ELONGATED * max(box_width, box_height))))  # noqa: RUF046
    hx1 = max(0, int(x1) - ox - slack)
    hy1 = max(0, int(y1) - oy - slack)
    hx2 = min(mask.shape[1], int(np.ceil(x2)) - ox + slack)
    hy2 = min(mask.shape[0], int(np.ceil(y2)) - oy + slack)
    outside = np.ones_like(mask, dtype=bool)
    outside[hy1:hy2, hx1:hx2] = False
    mask[outside] = 0
    return mask if mask.any() else None


def _contour_elderly(mask: np.ndarray) -> np.ndarray | None:
    """The external contour with the largest area, or None if the mask is empty."""
    if mask is None or not mask.any():
        return None
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cont, key=cv2.contourArea) if cont else None


def feret_maximum(contour: np.ndarray) -> float:
    """Maximum Feret diameter: the greatest distance between two edge points.

    It is computed on the convex hull, not on the whole contour: the most
    distant pair always lies on the hull, and the hull usually has tens of
    points against the contour's thousands, so the exhaustive search is cheap
    and exact.

    It is the correct measurement for "maximum apparent length" and for a
    STRAIGHT fiber. For a curved fiber it gives the chord, which
    underestimates its length.
    """
    if contour is None or len(contour) < 2:
        return 0.0
    env = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(env) < 2:
        return 0.0
    d = env[:, None, :] - env[None, :, :]
    return float(np.sqrt((d**2).sum(-1)).max())


def _spread_geodesic(inside: np.ndarray, seed: tuple) -> np.ndarray:
    """Geodesic distance from a seed, staying inside the mask.

    chamfer 3x3 successive-minima sweep over the 8 neighbours, weight 1 on
    straight and sqrt(2)-ish on diagonal. Vectorized: each iteration is eight
    shifted adds, and as many are needed as the longest path has pixels.

    Hot-loop buffer design (S02 rewrite): the loop allocates NOTHING on the
    steady state. All scratch is preallocated once (candidate buffer ``p``,
    update buffer ``e``, cached ``fuera`` mask, three slab temps for the
    shifted adds), every shifted add writes into a preallocated ``out=`` view,
    and the convergence update happens in-place into ``e`` followed by a
    reference swap instead of a full copy. Allocation churn was the dominant
    avoidable cost at ~5.9 ms/particle measured on a 10-plate batch.

    Bit-identical contract, pinned by tests/test_geodesia.py:
      * float32 weights np.float32(0.95509) / np.float32(1.36930), UNCHANGED;
      * the 8 shifted minimums accumulate into ``p`` in the SAME arange order
        as the pre-rewrite body (rectos then diagonals, bottom-up then top-down)
        — reordering is not free: float minimum is exact but the accumulator
        order is part of the frozen behavior;
      * ``tope = int(dentro.sum()) + 2`` iteration budget, unchanged;
      * convergence breaks on ``np.array_equal`` between successive sweeps,
        bit-exactly as before.

    Done this way rather than with scipy.sparse.csgraph because scipy is not
    a declared project dependency: it arrives transitively with ultralytics,
    and relying on it would break the install the day ultralytics stops
    shipping it.
    """
    INF = np.inf
    H, W = inside.shape
    d = np.full(inside.shape, INF, dtype=np.float32)
    d[seed] = 0.0
    # Optimal 3x3 chamfer weights, not 1 and sqrt(2). With the naive weights the
    # staircase path overestimates the Euclidean distance by up to 8% -- measured
    # on a circle of known diameter, it gave 108 instead of 100 --. These
    # two minimize that maximum error and leave it around 1.4%.
    UNO = np.float32(0.95509)
    R2 = np.float32(1.36930)
    # Safety bound: no simple path can have more steps than pixels.
    max_size = int(inside.sum()) + 2
    # Buffers preallocated once: p (candidates), e (update), the
    # cached inverse of `dentro` and three slabs for the shifted addends
    # (row, column and diagonal). None of this is allocated inside the loop.
    p = np.full_like(d, INF)
    e = np.empty_like(d)
    outside = ~inside  # cached: `~dentro` is constant throughout the sweep
    t_row = np.empty((max(H - 1, 0), W), dtype=np.float32)
    t_col = np.empty((H, max(W - 1, 0)), dtype=np.float32)
    t_diag = np.empty((max(H - 1, 0), max(W - 1, 0)), dtype=np.float32)
    for _ in range(max_size):
        p.fill(INF)
        # Straight
        np.add(d[:-1, :], UNO, out=t_row[: H - 1])
        np.minimum(p[1:, :], t_row[: H - 1], out=p[1:, :])
        np.add(d[1:, :], UNO, out=t_row[: H - 1])
        np.minimum(p[:-1, :], t_row[: H - 1], out=p[:-1, :])
        np.add(d[:, :-1], UNO, out=t_col[:, : W - 1])
        np.minimum(p[:, 1:], t_col[:, : W - 1], out=p[:, 1:])
        np.add(d[:, 1:], UNO, out=t_col[:, : W - 1])
        np.minimum(p[:, :-1], t_col[:, : W - 1], out=p[:, :-1])
        # Diagonal
        np.add(d[:-1, :-1], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[1:, 1:], t_diag[: H - 1, : W - 1], out=p[1:, 1:])
        np.add(d[:-1, 1:], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[1:, :-1], t_diag[: H - 1, : W - 1], out=p[1:, :-1])
        np.add(d[1:, :-1], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[:-1, 1:], t_diag[: H - 1, : W - 1], out=p[:-1, 1:])
        np.add(d[1:, 1:], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[:-1, :-1], t_diag[: H - 1, : W - 1], out=p[:-1, :-1])
        p[outside] = INF
        # In-place update with reference swap, without copying the array.
        np.minimum(d, p, out=e)
        # Checking against the previous step is enough: the propagation is monotonically
        # decreasing, so if nothing went down, it has converged.
        if np.array_equal(e, d):
            break
        d, e = e, d
    return d


def length_geodesic(mask: np.ndarray) -> float:
    """Longest path that fits INSIDE the particle, following its shape.

    It is the geodesic diameter. It does the right thing in the two cases that
    matter, without changing the formula:

      * Curved fiber: the path cannot leave the particle, so it follows the
        curve and returns its length, not the chord between the endpoints.
      * Compact particle: the longest path is the straight one, and it
        coincides with the maximum Feret diameter.

    It is obtained with the double-propagation technique, the same one that
    gives the diameter of a tree: from any point the farthest one is found,
    and from that one it propagates again; the maximum of that second
    propagation is the diameter. It does not depend on the perimeter, so a
    jagged edge does not inflate it the way the rectangle model does.
    """
    idx = np.argwhere(mask > 0)
    if len(idx) < 2:
        return 0.0
    # Crop to the minimum rectangle containing the particle. The cost of the
    # propagation goes with the canvas area, and a 30 px particle inside
    # a 4096 plate would cost the same as the whole plate.
    y0, x0 = idx.min(axis=0)
    y1_, x1_ = idx.max(axis=0) + 1
    inside = mask[y0:y1_, x0:x1_] > 0
    idx = np.argwhere(inside)

    d1 = _spread_geodesic(inside, tuple(idx[0]))
    finite = np.isfinite(d1)
    if not finite.any():
        return 0.0
    a = np.unravel_index(np.argmax(np.where(finite, d1, -1)), d1.shape)

    d2 = _spread_geodesic(inside, a)
    finite2 = np.isfinite(d2)
    return float(d2[finite2].max()) if finite2.any() else 0.0


def path_geodesic(mask: np.ndarray):
    """The points of the longest geodesic path, so it can be drawn.

    Same double propagation as ``largo_geodesico``, but returning the route
    instead of only its length. It is reconstructed by always stepping to the
    lowest-distance neighbour from the farthest endpoint.

    It exists so the report can SHOW along which route each fiber was
    measured. A size that cannot be seen measured cannot be verified.
    """
    idx = np.argwhere(mask > 0)
    if len(idx) < 2:
        return []
    y0, x0 = idx.min(axis=0)
    y1, x1 = idx.max(axis=0) + 1
    inside = mask[y0:y1, x0:x1] > 0
    ii = np.argwhere(inside)
    d1 = _spread_geodesic(inside, tuple(ii[0]))
    if not np.isfinite(d1).any():
        return []
    a = np.unravel_index(np.argmax(np.where(np.isfinite(d1), d1, -1)), d1.shape)
    d2 = _spread_geodesic(inside, a)
    if not np.isfinite(d2).any():
        return []
    b = np.unravel_index(np.argmax(np.where(np.isfinite(d2), d2, -1)), d2.shape)

    path, p = [b], b
    for _ in range(int(inside.sum()) + 2):
        if p == a:
            break
        best, minor = None, d2[p]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (p[0] + dy, p[1] + dx)
                if 0 <= q[0] < d2.shape[0] and 0 <= q[1] < d2.shape[1] and d2[q] < minor:
                    minor, best = d2[q], q
        if best is None:
            break
        p = best
        path.append(p)
    return [(int(c[1] + x0), int(c[0] + y0)) for c in path]


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

    if thinness >= THINNESS_FOR_GEODESIC and solidity < SOLIDITY_CONVEX and geo >= feret:
        length, method = geo, "geodesico"
    else:
        length, method = feret, "Feret maximo"

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
