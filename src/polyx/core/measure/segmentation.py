from __future__ import annotations

import cv2
import numpy as np

from . import morphology_constants as C


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
    if _elongation(mask) >= C.ELONGATION_ELONGATED:
        return mask

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dt.max() < 2:
        return mask

    _, cores = cv2.threshold(dt, C.THRESHOLD_CORE * dt.max(), 255, cv2.THRESH_BINARY)
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


def _contrast_threshold(canal, x1, y1, x2, y2, ox, oy):
    """Ring-vs-box contrast threshold.

    Returns (threshold, abort). ``abort`` is True when there is no contrast
    between the box and its surrounding ring, meaning the box does not
    contain a particle: ``segment`` then returns None instead of inventing
    a size. When ``threshold`` is None (but not abort) the caller falls
    back to Otsu.
    """
    bx1 = int(np.clip(int(x1) - ox, 0, canal.shape[1]))
    by1 = int(np.clip(int(y1) - oy, 0, canal.shape[0]))
    bx2 = int(np.clip(int(np.ceil(x2)) - ox, 0, canal.shape[1]))
    by2 = int(np.clip(int(np.ceil(y2)) - oy, 0, canal.shape[0]))
    ring = np.ones(canal.shape, dtype=bool)
    ring[by1:by2, bx1:bx2] = False
    inside_box = canal[by1:by2, bx1:bx2]

    if ring.sum() < 20 or inside_box.size < 9:
        return None, False
    background = float(np.median(canal[ring]))
    peak = float(np.percentile(inside_box, 90))
    if peak - background < C.CONTRAST_MINIMUM:
        # WITHOUT CONTRAST A SIZE IS NOT INVENTED. Returning None makes the
        # report say the particle could not be separated from the background.
        return None, True
    return background + 0.5 * (peak - background), False


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
    threshold_2, abort = _contrast_threshold(canal, x1, y1, x2, y2, ox, oy)
    if abort:
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
    if _elongation(mask) < C.ELONGATION_ELONGATED:
        slack = max(2, int(round(C.SLACK_COMPACT * min(box_width, box_height))))  # noqa: RUF046
    else:
        slack = max(2, int(round(C.SLACK_ELONGATED * max(box_width, box_height))))  # noqa: RUF046
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

__all__ = ['_contour_elderly', '_cutout', '_elongation', 'box_margin', 'feret_maximum', 'segment', 'separate_glued']
