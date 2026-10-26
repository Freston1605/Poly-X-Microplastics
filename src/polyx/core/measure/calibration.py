"""Fixes the um/px of a photo using the Petri plate as a length reference.

The external diameter of the plate is known -- 100 mm by default -- so its
radius in pixels determines the scale of THAT photo. It is needed because the
shooting distance changes between shots: in this material the um/px range from
31.3 to 33.9, and without the correction the same particle would be reported
with different sizes depending on which photo it landed on.

Hough alone is not usable as a reference: on some plates the radius was off by
up to 12%, and that error enters the scale in full. Here Hough only provides
the approximate centre; the radius comes from tracing the ring edge in 720
directions and fitting a circle by least squares while discarding outliers.
The fit targets the EXTERNAL edge of the ring, the one that matches the
nominal diameter.

The algorithm used to live in ``paper/detectar_placa.py`` and was moved here
so the application and the paper script measure with the same code. Two
implementations of the same measurement diverge, and then the figure in the
report stops being the figure in the paper."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

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
ORIGIN_PLATE = "placa"      # fitted on the ring of this same photo
ORIGIN_INDEX = "indice"    # inherited from the crop via `indice.csv`
ORIGIN_MANUAL = "manual"    # typed by hand in `Parametros`
ORIGIN_NONE = "ninguna"  # not calibrated; sizes cannot be reported


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
    warning: str = ""

    @property
    def valid(self) -> bool:
        return self.um_por_px > 0

    def description(self) -> str:
        """Short phrase for the interface and the report."""
        if not self.valid:
            return "sin calibrar"
        if self.origin == ORIGIN_PLATE:
            return (f"{self.um_por_px:.4f} µm/px — placa de "
                    f"{self.diameter_mm:g} mm, r={self.radius_px:.0f} px "
                    f"({self.n_points} puntos de borde)")
        if self.origin == ORIGIN_INDEX:
            return f"{self.um_por_px:.4f} µm/px — heredada del recorte"
        return f"{self.um_por_px:.4f} µm/px — introducida a mano"


def read_image(path_2: str | Path) -> Optional[np.ndarray]:
    """Reads with ``fromfile`` because folder names carry accents and parentheses.

    ``cv2.imread`` fails silently with non-ASCII paths on Windows and returns
    None, which here would be confused with "there is no plate".
    """
    try:
        return cv2.imdecode(np.fromfile(str(path_2), dtype=np.uint8),
                            cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def hough_approximate(bgr: np.ndarray) -> Optional[tuple]:
    """Approximate centre and radius. Only a seed for the fine fit."""
    height, width = bgr.shape[:2]
    scale = 900 / max(height, width)
    small = cv2.resize(bgr, (int(width * scale), int(height * scale)))
    gris = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    side = min(gris.shape)
    c = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, dp=1.5, minDist=side,
                         param1=120, param2=60,
                         minRadius=int(side * 0.22), maxRadius=int(side * 0.52))
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
    profiles = gris[ys, xs].astype(float)          # (radius, angle)

    base = np.median(gris[int(max(0, cy - r * 0.5)):int(cy + r * 0.5),
                          int(max(0, cx - r * 0.5)):int(cx + r * 0.5)])
    points = []
    for k in range(N_ANGLES):
        col = profiles[:, k]
        peak = int(np.argmax(col))
        if col[peak] <= base + 12:                 # no ring seen here
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
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return cx, cy, r


def adjustment_robust(pts: np.ndarray, iterations: int = 3) -> Optional[tuple]:
    """Fits and discards distant points; the ring has reflections and occlusions."""
    for _ in range(iterations):
        if len(pts) < 30:
            return None
        cx, cy, r = adjust_circle(pts)
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        pts = pts[d < max(3.0, 2.0 * np.median(d))]
    return adjust_circle(pts) + (len(pts),)


def correct_parallax(um_por_px: float, height_plate_mm: float,
                      distance_camera_mm: float) -> tuple:
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


def calibrate_from_plate(bgr: np.ndarray,
                         diameter_mm: float = DIAMETER_PLATE_MM,
                         height_plate_mm: float = 0.0,
                         distance_camera_mm: float = 0.0) -> Calibration:
    """Measures the plate ring and returns the scale for this photo.

    It returns an invalid Calibracion -- with ``aviso`` explaining why -- when
    the plate cannot be found, instead of raising: in a batch of 69 photos the
    run must be able to continue with the rest.
    """
    if bgr is None or bgr.size == 0:
        return Calibration(warning="imagen vacia o ilegible")
    if diameter_mm <= 0:
        return Calibration(warning="diametro nominal invalido")

    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    seed = hough_approximate(bgr)
    if seed is None:
        return Calibration(diameter_mm=diameter_mm,
                           warning="no se encontro la placa en la foto")
    hx, hy, hr = seed

    pts = edge_points(gris, hx, hy, hr)
    fine = adjustment_robust(pts) if len(pts) >= 30 else None
    if fine is None:
        # Hough as a last resort: it serves to avoid losing the photo, but its radius
        # is precisely what is unreliable, so a warning is recorded.
        cx, cy, r, n = hx, hy, hr, 0
        warning = "el ajuste fino del anillo fallo; escala tomada de Hough (poco fiable)"
    else:
        cx, cy, r, n = fine
        warning = ""

    detour = 100.0 * (r - hr) / hr if hr > 0 else 0.0
    if not warning and abs(detour) > 8:
        warning = f"Hough erraba {detour:+.1f}% respecto del ajuste fino"

    if r <= 0:
        return Calibration(diameter_mm=diameter_mm, warning="radio ajustado nulo")

    um_aro = (diameter_mm * 1000.0) / (2 * r)
    um_base, factor = correct_parallax(um_aro, height_plate_mm,
                                        distance_camera_mm)
    # No warning when no correction is applied: it was measured on this material and the effect
    # is undetectable (see `corregir_paralaje()`). Warning on every photo would
    # raise an alarm over a bias that the data bound below 1.3%.
    return Calibration(um_por_px=um_base,
                       origin=ORIGIN_PLATE, cx=cx, cy=cy, radius_px=r,
                       n_points=n, detour_hough_pct=detour,
                       diameter_mm=diameter_mm, factor_parallax=factor,
                       warning=warning)


def carry_index(path_csv: str | Path) -> Dict[str, float]:
    """Reads a CSV mapping image name to its already-calibrated um/px.

    Meant for the crops: the full plate does not appear in a crop, so its
    scale cannot be re-measured on it and must be inherited from the photo the
    crop came from. ``cortar_placas.py`` leaves that mapping in
    ``indice.csv``, and ``detectar_placa.py`` leaves the one for full plates
    in ``calibracion_placas.csv``.

    Both headers are accepted because they are the two files the pipeline
    produces; it returns ``{}`` if the file does not exist or lacks the
    column.
    """
    path_2 = Path(path_csv)
    if not path_2.is_file():
        return {}
    map_value: Dict[str, float] = {}
    try:
        with path_2.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name_2 = row.get("recorte") or row.get("placa") or ""
                try:
                    um = float(row.get("um_por_px", "") or 0)
                except ValueError:
                    continue
                if name_2 and um > 0:
                    map_value[name_2] = um
    except (OSError, csv.Error):
        return {}
    return map_value


NAMES_INDEX = ("indice.csv", "calibracion_placas.csv")


def search_index(image_path: str | Path, levels: int = 3) -> Dict[str, float]:
    """Finds a calibration CSV next to the images, walking some levels up.

    It is searched for instead of required because the paper pipeline writes
    the CSV into the crops folder or the one above it, and having to point at
    it by hand is exactly the step that gets forgotten; forgetting it raises
    no error, it produces a report with the wrong scale.
    """
    folder = Path(image_path)
    folder = folder.parent if folder.is_file() else folder
    for _ in range(max(1, levels)):
        for name_2 in NAMES_INDEX:
            map_value = carry_index(folder / name_2)
            if map_value:
                return map_value
        if folder.parent == folder:
            break
        folder = folder.parent
    return {}


def resolver(image_path: str | Path,
             index: Optional[Dict[str, float]] = None,
             um_por_px_manual: float = 0.0,
             diameter_mm: float = DIAMETER_PLATE_MM,
             measure_plate: bool = False,
             bgr: Optional[np.ndarray] = None,
             height_plate_mm: float = 0.0,
             distance_camera_mm: float = 0.0) -> Calibration:
    """Scale of an image, in order of reliability.

    1. ``indice`` -- already calibrated against the plate, per photo. It is
       what the crops have and costs nothing.
    2. measure the plate on the photo itself, if requested and visible.
    3. the value typed in Parametros, which applies a single value to the
       whole batch.

    Order matters: the manual value comes last because it is unique for the
    whole batch, and in this material the true scale varies up to 1.5x between
    photos.

    ``bgr`` avoids re-reading from disk when the caller already has the image
    decoded, which is the runner's case: 12 MP per photo.
    """
    name_2 = Path(image_path).name
    if index and name_2 in index:
        return Calibration(um_por_px=index[name_2], origin=ORIGIN_INDEX,
                           diameter_mm=diameter_mm)

    if measure_plate:
        if bgr is None:
            bgr = read_image(image_path)
        cal = calibrate_from_plate(bgr, diameter_mm, height_plate_mm,
                                   distance_camera_mm)
        if cal.valid:
            return cal
        if um_por_px_manual > 0:
            return Calibration(um_por_px=um_por_px_manual, origin=ORIGIN_MANUAL,
                               diameter_mm=diameter_mm,
                               warning=f"no se pudo medir la placa ({cal.warning})")
        return cal

    if um_por_px_manual > 0:
        return Calibration(um_por_px=um_por_px_manual, origin=ORIGIN_MANUAL,
                           diameter_mm=diameter_mm)
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


def radius_fraction(cal, x: float, y: float) -> Optional[float]:
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
        return {"n": 0, "origenes": {}}
    a = np.array(ums, dtype=float)
    origins: Dict[str, int] = {}
    for c in calibrations:
        if c.valid:
            origins[c.origin] = origins.get(c.origin, 0) + 1
    return {
        "n": len(ums),
        "min": float(a.min()),
        "max": float(a.max()),
        "media": float(a.mean()),
        # ddof=1 (sample standard deviation): with n photos the estimate is of the
        # spread of the population of photos, not merely a description of these n.
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "mediana": float(np.median(a)),
        "variacion": float(a.max() / a.min()) if a.min() > 0 else 0.0,
        "origenes": origins,
        "avisos": [c.warning for c in calibrations if c.warning],
    }
