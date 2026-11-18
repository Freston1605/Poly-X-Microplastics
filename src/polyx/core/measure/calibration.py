"""Fixes the µm/px of a photo from a known real length.

The Petri plate is the reference for unattended runs. Its external diameter
is known -- 100 mm by default -- so its radius in pixels determines the scale
of THAT photo. It is needed because the shooting distance changes between
shots: in this material the µm/px range from 31.3 to 33.9, and without the
correction the same particle would be reported with different sizes depending
on which photo it landed on.

Hough alone is not usable as a reference: on some plates the radius was off by
up to 12%, and that error enters the scale in full. Here Hough only provides
the approximate centre; the radius comes from tracing the ring edge in 720
directions and fitting a circle by least squares while discarding outliers.
The fit targets the EXTERNAL edge of the ring, the one that matches the
nominal diameter.

The algorithm used to live in ``paper/detectar_placa.py`` and was moved here
so the application and the paper script measure with the same code. Two
implementations of the same measurement diverge, and then the figure in the
report stops being the figure in the paper.

The Viewer also calibrates interactively, from a line or a circle the user
draws over an object of known size. Those helpers are plain geometry over
coordinate tuples -- no Qt and no numpy requirement -- that return a named
failure for degenerate input instead of inventing a scale.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from polyx.core.i18n import _

# Canonical implementation lives in polyx.core.image_io; re-exported here
# for backwards compatibility with callers that import from this module.
from polyx.core.image_io import read_image

from .warnings import (
    DETOUR_HOUGH,
    IMAGE_UNREADABLE,
    NOMINAL_DIAMETER_INVALID,
    PLATE_MEASURE_FAILED,
    PLATE_NOT_FOUND,
    RING_FIT_FAILED,
    ZERO_ADJUSTED_RADIUS,
    Warning,
    warning_to_dict,
)

# Directions in which the ring edge is searched. 720 = one every half degree:
# enough for the fit to survive reflections and partial occlusions.
N_ANGLES = 720

# Diameter of the study Petri plate, in mm. Measured on the real plates.
# It is noted because standard Petri plates come in 90 and 100 mm, and mixing them up
# would introduce an 11% error in every size without any symptom.
#
# THIS VALUE GOES WITH THE **EXTERNAL** EDGE OF THE RING, which is the edge fitted by
# `puntos_del_borde()` -- it walks OUTWARD from the brightness peak. Changing one
# without the other shifts the whole study scale.
DIAMETER_PLATE_MM = 100.0

# Wall thickness of the ring, measured on the study photos: the inner
# edge falls at 0.960 of the fitted radius and the external one at 1.000, about 2.0 mm of
# wall (1.95 / 2.02 / 2.02 mm on three different plates).
#
# THE SCALE UNCERTAINTY COMES FROM HERE, and it is not symmetric. The nominal
# diameter of a Petri plate is ambiguous at the wall level: it may refer to the external diameter
# or to the usable inner one. Here the EXTERNAL one is taken. If it actually referred to the
# inner one -- 96 mm instead of 100 -- the correct scale would be 4.2% larger
# (100/96) and EVERY size in the study would be underestimated by that figure.
#
# The error can only go in that direction, never the other way, because the external edge is the
# larger of the two possible edges.
#
# It was not resolved in favour of the inner one because: (a) the tape-measure photos give
# 98-100 mm at the external edge, (b) Petri plate catalogues usually name
# the external diameter, and (c) the inner edge is not usable for calibration -- it is a
# gradual transition that is not seen in 3 out of 14 plates, hidden by the
# sediment. It is declared in the report instead of being concealed.
THICKNESS_WALL_MM = 2.0
BIAS_MAXIMUM_WALL_PCT = 4.2

# Where the scale came from, from most to least reliable. It goes into the report: a measurement without
# its provenance is not verifiable.
ORIGIN_PLATE = "plate"  # fitted on the ring of this same photo
ORIGIN_INDEX = "index"  # inherited from the crop via the calibration index
ORIGIN_MANUAL = "manual"  # typed by hand in Parameters
ORIGIN_NONE = "none"  # not calibrated; sizes cannot be reported

# Canonical per-image calibration index. It is an English machine contract
# (ADR-0004): the file is named and asked for explicitly, never searched for,
# and the legacy Spanish files and headers are refused rather than parsed.
CALIBRATION_INDEX_NAME = "calibration_index.csv"
CALIBRATION_INDEX_HEADER = ("image", "um_per_px")
LEGACY_INDEX_NAMES = ("indice.csv", "calibracion_placas.csv")
LEGACY_INDEX_HEADERS = ("recorte", "placa", "um_por_px")


@dataclass
class Calibration:
    """Scale of an image and how it was obtained."""

    um_por_px: float = 0.0
    origin: str = ORIGIN_NONE
    cx: float = 0.0
    cy: float = 0.0
    radius_px: float = 0.0
    n_points: int = 0
    detour_hough_pct: float = 0.0
    diameter_mm: float = DIAMETER_PLATE_MM
    # Parallax correction ring->base. 1.0 = not applied, in which case the
    # sizes are underestimated by an amount that depends on the shooting
    # distance. It is stored so it can be declared in the report.
    factor_parallax: float = 1.0
    # Stable machine code plus structured parameters; never translated prose.
    warning: Warning | None = None

    @property
    def valid(self) -> bool:
        return self.um_por_px > 0

    def description(self) -> str:
        """Short gettext sentence for the interface and the report.

        English is the source language and the message id; the translated
        wording is chosen here at display time, never stored in an artifact.
        """
        if not self.valid:
            return _("not calibrated")
        scale = f"{self.um_por_px:.4f}"
        if self.origin == ORIGIN_PLATE:
            return _(
                "%(scale)s µm/px — plate of %(diameter)s mm, r=%(radius)s px "
                "(%(points)s edge points)"
            ) % {
                "scale": scale,
                "diameter": f"{self.diameter_mm:g}",
                "radius": f"{self.radius_px:.0f}",
                "points": self.n_points,
            }
        if self.origin == ORIGIN_INDEX:
            return _("%(scale)s µm/px — inherited from the crop") % {"scale": scale}
        return _("%(scale)s µm/px — entered by hand") % {"scale": scale}


# ── Interactive line and circle calibration ─────────────────────────────
#
# The Viewer lets the user draw a line or a circle over an object of known
# real size and turns the drawing into µm/px. The geometry is here rather
# than in the GUI state because it is plain arithmetic over coordinate
# tuples: it can be tested without a display, and it sits next to every
# other µm/px concern in the application.
Point = tuple[float, float]

# Named reasons a drawing cannot support a scale. They are stable machine
# tokens: the callers choose the wording shown to the user for each one, and
# a refused calibration never comes back as the number zero.
FAILURE_NON_FINITE = "non_finite_values"
FAILURE_ZERO_LENGTH = "zero_length"
FAILURE_NON_POSITIVE_REAL_SIZE = "non_positive_real_size"
FAILURE_COINCIDENT_POINTS = "coincident_points"
FAILURE_COLLINEAR_POINTS = "collinear_points"

#: Every named refusal this module can produce. A caller that maps failures to
#: user-facing copy iterates this tuple, so a new refusal cannot silently fall
#: through to a generic sentence.
FAILURE_CODES = (
    FAILURE_NON_FINITE,
    FAILURE_ZERO_LENGTH,
    FAILURE_NON_POSITIVE_REAL_SIZE,
    FAILURE_COINCIDENT_POINTS,
    FAILURE_COLLINEAR_POINTS,
)

# Relative tolerance of the collinearity test. A triangle thinner than this
# has no numerically meaningful circumcircle; its radius would be dominated
# by rounding error.
COLLINEAR_TOLERANCE = 1e-12


def _all_finite(*values: float) -> bool:
    """Whether every number is a real one: NaN and ±inf cannot give a scale."""
    return all(math.isfinite(value) for value in values)


def _distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _scale(real_um: float, drawn_px: float) -> tuple[float | None, str | None]:
    """µm per pixel, or the reason the real size cannot give one.

    ``drawn_px`` is the pixel span the real size covers: the length of a line
    or the diameter of a circle. The refusal of a non-positive real size lives
    here, once, because the division it guards is the same one in both.
    """
    if real_um <= 0.0:
        return None, FAILURE_NON_POSITIVE_REAL_SIZE
    return real_um / drawn_px, None


@dataclass(frozen=True)
class LineCalibration:
    """Scale from a drawn line, or why it was refused."""

    length_px: float | None = None
    um_per_px: float | None = None
    failure: str | None = None


def line_calibration(p1: Point, p2: Point, real_um: float) -> LineCalibration:
    """µm/px from a line of known real length, in µm.

    Returns ``None`` plus a named failure instead of a scale when the drawing
    cannot support one; a zero scale is still a scale, and every size computed
    with it would silently come out zero.
    """
    if not _all_finite(*p1, *p2, real_um):
        return LineCalibration(failure=FAILURE_NON_FINITE)
    length_px = _distance(p1, p2)
    if length_px == 0.0:
        return LineCalibration(length_px=0.0, failure=FAILURE_ZERO_LENGTH)
    um_per_px, failure = _scale(real_um, length_px)
    return LineCalibration(length_px=length_px, um_per_px=um_per_px, failure=failure)


@dataclass(frozen=True)
class CircumscribedCircle:
    """Centre and radius of the circle through three drawn points."""

    center: Point
    radius_px: float


def _circumscribed(
    p1: Point, p2: Point, p3: Point
) -> tuple[CircumscribedCircle | None, str | None]:
    """The circle through three FINITE points, or the reason there is none.

    Callers check finiteness first: this only tells apart the degenerate
    triangles, which is the part that names a failure for the user.
    """
    ab = (p2[0] - p1[0], p2[1] - p1[1])
    ac = (p3[0] - p1[0], p3[1] - p1[1])
    ab_len = math.hypot(*ab)
    ac_len = math.hypot(*ac)
    if ab_len == 0.0 or ac_len == 0.0 or _distance(p2, p3) == 0.0:
        return None, FAILURE_COINCIDENT_POINTS

    cross = ab[0] * ac[1] - ab[1] * ac[0]
    if abs(cross) <= COLLINEAR_TOLERANCE * ab_len * ac_len:
        return None, FAILURE_COLLINEAR_POINTS

    ab2 = ab[0] * ab[0] + ab[1] * ab[1]
    ac2 = ac[0] * ac[0] + ac[1] * ac[1]
    ux = (ac[1] * ab2 - ab[1] * ac2) / (2.0 * cross)
    uy = (ab[0] * ac2 - ac[0] * ab2) / (2.0 * cross)
    return CircumscribedCircle(
        center=(p1[0] + ux, p1[1] + uy),
        radius_px=math.hypot(ux, uy),
    ), None


def circumscribed_circle(p1: Point, p2: Point, p3: Point) -> CircumscribedCircle | None:
    """Circle through three points, or ``None`` when there is no single one.

    ``None`` covers a non-finite coordinate, two coincident points and three
    collinear ones: none of the three defines a circle. The Viewer preview
    needs only this geometry and has no real size to give, which is why it is
    not asked for here. A caller that must tell the user WHICH refusal
    happened asks :func:`circle_calibration`, which names it.
    """
    if not _all_finite(*p1, *p2, *p3):
        return None
    circle, _failure = _circumscribed(p1, p2, p3)
    return circle


@dataclass(frozen=True)
class CircleCalibration:
    """Circumscribed circle and its scale, or why they were refused."""

    circle: CircumscribedCircle | None = None
    um_per_px: float | None = None
    failure: str | None = None

    @property
    def center(self) -> Point | None:
        """Centre of the circle, or ``None`` when none was defined."""
        return self.circle.center if self.circle is not None else None

    @property
    def radius_px(self) -> float | None:
        return self.circle.radius_px if self.circle is not None else None


def circle_calibration(
    p1: Point, p2: Point, p3: Point, real_diameter_um: float
) -> CircleCalibration:
    """µm/px from the circle through three points and its real diameter, in µm.

    The radius is the circumradius of the triangle, so the result does not
    depend on where the first point was placed.
    """
    if not _all_finite(*p1, *p2, *p3, real_diameter_um):
        return CircleCalibration(failure=FAILURE_NON_FINITE)

    circle, failure = _circumscribed(p1, p2, p3)
    if circle is None:
        return CircleCalibration(failure=failure)

    um_per_px, failure = _scale(real_diameter_um, 2.0 * circle.radius_px)
    return CircleCalibration(circle=circle, um_per_px=um_per_px, failure=failure)


def hough_approximate(bgr: np.ndarray) -> tuple | None:
    """Approximate centre and radius. Only a seed for the fine fit."""
    height, width = bgr.shape[:2]
    scale = 900 / max(height, width)
    small = cv2.resize(bgr, (int(width * scale), int(height * scale)))
    gris = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    side = min(gris.shape)
    c = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=side,
        param1=120,
        param2=60,
        minRadius=int(side * 0.22),
        maxRadius=int(side * 0.52),
    )
    if c is None:
        return None
    best = max(np.round(c[0]).astype(float), key=lambda v: v[2])
    return best[0] / scale, best[1] / scale, best[2] / scale


def edge_points(gris: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """For each angle, the radius of the external edge of the bright ring."""
    ang = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    radii = np.arange(max(1, int(r * 0.80)), int(r * 1.25))
    xs = np.clip(cx + np.outer(radii, np.cos(ang)), 0, gris.shape[1] - 1).astype(int)
    ys = np.clip(cy + np.outer(radii, np.sin(ang)), 0, gris.shape[0] - 1).astype(int)
    profiles = gris[ys, xs].astype(float)  # (radius, angle)

    base = np.median(
        gris[
            int(max(0, cy - r * 0.5)) : int(cy + r * 0.5),
            int(max(0, cx - r * 0.5)) : int(cx + r * 0.5),
        ]
    )
    points = []
    for k in range(N_ANGLES):
        col = profiles[:, k]
        peak = int(np.argmax(col))
        if col[peak] <= base + 12:  # no ring seen here
            continue
        threshold_2 = base + (col[peak] - base) * 0.5
        j = peak
        while j < len(col) - 1 and col[j] > threshold_2:
            j += 1
        rr = radii[j]
        points.append((cx + rr * np.cos(ang[k]), cy + rr * np.sin(ang[k])))
    return np.array(points)


def adjust_circle(pts: np.ndarray) -> tuple:
    """Algebraic circle fit by least squares (Kasa)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx**2 + cy**2)
    return cx, cy, r


def adjustment_robust(pts: np.ndarray, iterations: int = 3) -> tuple | None:
    """Fits and discards distant points; the ring has reflections and occlusions."""
    for _iteration in range(iterations):
        if len(pts) < 30:
            return None
        cx, cy, r = adjust_circle(pts)
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        pts = pts[d < max(3.0, 2.0 * np.median(d))]
    return adjust_circle(pts) + (len(pts),)  # noqa: RUF005


def correct_parallax(um_por_px: float, height_plate_mm: float, distance_camera_mm: float) -> tuple:
    """Carries the plate RING scale over to the BASE PLANE.

    The ring is closer to the camera than the bottom where the particles rest,
    so it projects larger: 100 mm measured on the ring occupy more pixels than
    100 mm resting on the base. Since
    ``um/px = 100000 / diametro_px``, an inflated diameter gives a small
    um/px, and since ``talla = px * um/px``, **every size comes out
    underestimated**.

    With D the distance from camera to base and h the plate height, the ring
    sits at D-h and the factor between both scales is D/(D-h).

    MEASURED ON THIS MATERIAL THE EFFECT IS NEGLIGIBLE AND NEEDS NO
    CORRECTION. The reasoning holds in general -- shooting close up the error
    would be large, 16% with the plate at 15 mm and the camera at 100 mm --
    but here it was checked like this: the bottom edge of the plate is
    visible, and its radius is 0.863 times that of the ring. That shrinkage
    could be perspective or it could be the shape of the plate (the bottom is
    recessed). They can be told apart because perspective depends on shooting
    distance and shape does not, and there are plates photographed at two
    distances that differ by 1.43x:

        near (n=14):   base/ring = 0.8634 +- 0.0058
        far (n=14):    base/ring = 0.8643 +- 0.0077
        ratio between groups: 1.0010 +- 0.0030  ->  indistinguishable from 1.0

    Since it does not change with distance, the shrinkage is the shape of the
    plate and perspective stays bounded below 1.3% even at the edge of the
    uncertainty. That is why both parameters arrive as 0 and nothing is
    corrected.

    The correction is left implemented because another setup -- a camera
    closer in, or taller plates -- would actually need it, and because it is
    worth writing down how the hypothesis was discarded instead of repeating
    the doubt.

    Returns (um_por_px_corregido, factor). With insufficient data it returns
    the value untouched and factor 1.0: an uncorrected but declared scale is
    preferable to a corrected one computed from invented numbers.
    """
    if height_plate_mm <= 0 or distance_camera_mm <= 0:
        return um_por_px, 1.0
    if distance_camera_mm <= height_plate_mm:
        # The camera would be inside the plate; the value is wrong.
        return um_por_px, 1.0
    factor = distance_camera_mm / (distance_camera_mm - height_plate_mm)
    return um_por_px * factor, factor


def calibrate_from_plate(
    bgr: np.ndarray,
    diameter_mm: float = DIAMETER_PLATE_MM,
    height_plate_mm: float = 0.0,
    distance_camera_mm: float = 0.0,
) -> Calibration:
    """Measures the plate ring and returns the scale for this photo.

    It returns an invalid Calibracion -- with ``aviso`` explaining why -- when
    the plate cannot be found, instead of raising: in a batch of 69 photos the
    run must be able to continue with the rest.
    """
    if bgr is None or bgr.size == 0:
        return Calibration(warning=Warning(IMAGE_UNREADABLE))
    if diameter_mm <= 0:
        return Calibration(warning=Warning(NOMINAL_DIAMETER_INVALID))

    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    seed = hough_approximate(bgr)
    if seed is None:
        return Calibration(diameter_mm=diameter_mm, warning=Warning(PLATE_NOT_FOUND))
    hx, hy, hr = seed

    pts = edge_points(gris, hx, hy, hr)
    fine = adjustment_robust(pts) if len(pts) >= 30 else None
    if fine is None:
        # Hough as a last resort: it serves to avoid losing the photo, but its radius
        # is precisely what is unreliable, so a warning is recorded.
        cx, cy, r, n = hx, hy, hr, 0
        warning: Warning | None = Warning(RING_FIT_FAILED)
    else:
        cx, cy, r, n = fine
        warning = None

    detour = 100.0 * (r - hr) / hr if hr > 0 else 0.0
    if warning is None and abs(detour) > 8:
        warning = Warning(DETOUR_HOUGH, detour_pct=detour)

    if r <= 0:
        return Calibration(diameter_mm=diameter_mm, warning=Warning(ZERO_ADJUSTED_RADIUS))

    um_aro = (diameter_mm * 1000.0) / (2 * r)
    um_base, factor = correct_parallax(um_aro, height_plate_mm, distance_camera_mm)
    # No warning when no correction is applied: it was measured on this material and the effect
    # is undetectable (see `corregir_paralaje()`). Warning on every photo would
    # raise an alarm over a bias that the data bound below 1.3%.
    return Calibration(
        um_por_px=um_base,
        origin=ORIGIN_PLATE,
        cx=cx,
        cy=cy,
        radius_px=r,
        n_points=n,
        detour_hough_pct=detour,
        diameter_mm=diameter_mm,
        factor_parallax=factor,
        warning=warning,
    )


class CalibrationIndexError(ValueError):
    """The file is not the canonical English per-image calibration index."""


def read_calibration_index(path: str | Path) -> dict[str, float]:
    """Read ``image,um_per_px`` from the one file the caller points at.

    The index is the per-image scale source for crops whose dish is not
    visible: a single batch-wide fallback cannot scale photos that were shot
    at different distances. The reader is deliberately strict -- the legacy
    Spanish filenames and headers are refused with an actionable error that
    names the canonical format, and duplicate or invalid rows are never
    silently skipped. It only reads the given path: it never searches the
    filesystem and never rewrites or deletes the user's file.

    Values are keyed by image basename, which is what the run looks up. A
    relative path in the ``image`` column is reduced to its basename too.
    """
    path = Path(path)
    canonical = _("use 'calibration_index.csv' with the headers image,um_per_px")
    if path.name in LEGACY_INDEX_NAMES:
        raise CalibrationIndexError(
            _("'%(name)s' is a legacy calibration index and is not read; %(canonical)s")
            % {"name": path.name, "canonical": canonical}
        )
    if not path.is_file():
        raise CalibrationIndexError(_("calibration index not found: %(path)s") % {"path": path})

    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        try:
            raw_header = next(rows)
        except StopIteration:
            raise CalibrationIndexError(
                _("'%(name)s' is empty; %(canonical)s")
                % {"name": path.name, "canonical": canonical}
            ) from None
        header = [field.lstrip("\ufeff").strip() for field in raw_header]
        if tuple(header) != CALIBRATION_INDEX_HEADER:
            legacy = next((field for field in header if field in LEGACY_INDEX_HEADERS), None)
            if legacy is not None:
                raise CalibrationIndexError(
                    _("legacy header '%(header)s' in '%(name)s' is not read; %(canonical)s")
                    % {"header": legacy, "name": path.name, "canonical": canonical}
                )
            raise CalibrationIndexError(
                _(
                    "'%(name)s' must carry the headers image,um_per_px; got %(header)s; %(canonical)s"
                )
                % {"name": path.name, "header": ",".join(header), "canonical": canonical}
            )

        invalid_value = _("invalid um/px for '%(image)s' in '%(name)s': expected a positive number")
        mapping: dict[str, float] = {}
        for line_number, row in enumerate(rows, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 2:
                raise CalibrationIndexError(
                    _("row %(line)d in '%(name)s' must have two columns: image,um_per_px")
                    % {"line": line_number, "name": path.name}
                )
            image = row[0].strip()
            if not image:
                raise CalibrationIndexError(
                    _("row %(line)d in '%(name)s' has a blank image name; %(canonical)s")
                    % {"line": line_number, "name": path.name, "canonical": canonical}
                )
            try:
                value = float(row[1].strip())
            except ValueError:
                raise CalibrationIndexError(
                    invalid_value % {"image": image, "name": path.name}
                ) from None
            if not math.isfinite(value) or value <= 0:
                raise CalibrationIndexError(invalid_value % {"image": image, "name": path.name})
            basename = Path(image.replace("\\", "/")).name
            if basename in mapping:
                raise CalibrationIndexError(
                    _(
                        "duplicate image '%(image)s' in '%(name)s'; each image needs one um_per_px value"
                    )
                    % {"image": image, "name": path.name}
                )
            mapping[basename] = value
    return mapping


def resolver(
    image_path: str | Path,
    um_por_px_manual: float = 0.0,
    diameter_mm: float = DIAMETER_PLATE_MM,
    measure_plate: bool = False,
    bgr: np.ndarray | None = None,
    height_plate_mm: float = 0.0,
    distance_camera_mm: float = 0.0,
) -> Calibration:
    """Scale of an image, in order of reliability.

    1. measure the plate on the photo itself, if requested and visible.
    2. the value typed in Parameters, which applies a single value to the
       whole batch.

    Order matters: the manual value comes last because it is unique for the
    whole batch, and in this material the true scale varies up to 1.5x between
    photos. Pre-canonical index CSVs are no longer read.

    ``bgr`` avoids re-reading from disk when the caller already has the image
    decoded, which is the runner's case: 12 MP per photo.
    """
    if measure_plate:
        if bgr is None:
            bgr = read_image(image_path)
        if bgr is None:
            return Calibration(
                origin=ORIGIN_NONE, diameter_mm=diameter_mm, warning=Warning(IMAGE_UNREADABLE)
            )
        cal = calibrate_from_plate(bgr, diameter_mm, height_plate_mm, distance_camera_mm)
        if cal.valid:
            return cal
        if um_por_px_manual > 0:
            return Calibration(
                um_por_px=um_por_px_manual,
                origin=ORIGIN_MANUAL,
                diameter_mm=diameter_mm,
                warning=Warning(
                    PLATE_MEASURE_FAILED,
                    cause=cal.warning.code if cal.warning is not None else None,
                ),
            )
        return cal

    if um_por_px_manual > 0:
        return Calibration(
            um_por_px=um_por_px_manual, origin=ORIGIN_MANUAL, diameter_mm=diameter_mm
        )
    return Calibration(diameter_mm=diameter_mm)


# Fraction of the fitted radius up to which the plate area is considered usable.
#
# The circle is fitted to the EXTERNAL edge of the ring, and the wall measures about 2 mm
# over a radius of 50: the inner edge falls at 0.960. The cutoff is 0.95,
# inside that wall.
#
# THAT NO PARTICLE IS LOST IS MEASURED, not assumed: on
# the 100 hand-annotated particles of this study, the most peripheral one is at
# 0.828 of the radius and none goes beyond it. Between 0.83 and 0.95 there is ample margin,
# so the cutoff excludes the ring at no cost in sample. If on another material the
# sediment ever came to rest against the wall, this number would have to be reviewed
# with the same measurement.
FRACTION_AREA_USEFUL = 0.95


def radius_fraction(cal, x: float, y: float) -> float | None:
    """At what fraction of the plate radius the point (x, y) falls.

    None when that photo has no fitted circle, which happens when the scale
    came in by hand or was inherited from the index.
    """
    if cal is None or not getattr(cal, "radius_px", 0):
        return None
    return float(np.hypot(x - cal.cx, y - cal.cy)) / float(cal.radius_px)


def about_the_ring(cal, x1: float, y1: float, x2: float, y2: float) -> bool:
    """Whether the box (x1,y1,x2,y2) falls on the plate wall or outside it.

    The CENTRE of the box is checked, not a corner: a legitimate particle
    touching the usable edge has corners that poke out, and discarding it for
    that would lose real sample. The ring, in contrast, falls entirely
    outside.

    Returns False when there is no fitted circle: without knowing where the
    plate is, nothing can be claimed to be outside, and discarding on a hunch
    would be inventing a rejection.
    """
    f = radius_fraction(cal, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return f is not None and f > FRACTION_AREA_USEFUL


def summary_batch(calibrations) -> dict:
    """Scale statistics of the batch, for the methods section of the report.

    Dispersion is the reported figure: if the photos do not share a scale, a
    size histogram pooled over all of them mixes units.
    """
    ums = [c.um_por_px for c in calibrations if c.valid]
    if not ums:
        return {"n": 0, "origins": {}, "warnings": []}
    a = np.array(ums, dtype=float)
    origins: dict[str, int] = {}
    for c in calibrations:
        if c.valid:
            origins[c.origin] = origins.get(c.origin, 0) + 1
    return {
        "n": len(ums),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        # ddof=1 (sample standard deviation): with n photos the estimate is of the
        # spread of the population of photos, not merely a description of these n.
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "variation": float(a.max() / a.min()) if a.min() > 0 else 0.0,
        "origins": origins,
        # Warnings are stable codes plus structured parameters, never prose.
        # The list is groupable and countable by ``code``.
        "warnings": [warning_to_dict(c.warning) for c in calibrations if c.warning],
    }
