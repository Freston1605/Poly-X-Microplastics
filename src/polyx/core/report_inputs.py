"""Report input helpers (moved out of the GUI)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from polyx.core.report_metrics import _classes_views

_RE_PLATE = re.compile(r"^(\d+)\.(\d+)(x*)$")
_STATION = {"1": "Chiu Chiu", "2": "Taira", "3": "Desembocadura"}
_ORDER_STATION = ["Chiu Chiu", "Taira", "Desembocadura"]
_PLATE = {"": "a", "x": "b", "xx": "c"}


def _sample_of(path_2) -> tuple | None:
    """(``estacion``, ``tramo``, ``placa``) parsed from the name, or None if it does
    not follow the naming pattern.

    The report cannot demand this naming: if the images come from another
    study, the grouped table is simply skipped and the per-image table
    stays, which can always be built.
    """
    stem = Path(path_2).stem.split("__")[0]
    m = _RE_PLATE.match(stem)
    if not m:
        return None
    range_value, witness, suffix = m.group(1), m.group(2), m.group(3)
    return (
        _STATION.get(witness, f"Testigo {witness}"),
        int(range_value),
        _PLATE.get(suffix, suffix),
    )


def _clue_order(path_2) -> tuple:
    """Sorts by ``estacion``/``tramo``/``placa`` when the name allows it."""
    m = _sample_of(path_2)
    if m is None:
        return (1, 0, 0, "", Path(path_2).name)
    station, range_value, plate = m
    idx = _ORDER_STATION.index(station) if station in _ORDER_STATION else 99
    return (0, idx, range_value, plate, Path(path_2).name)


def _lot_forms(state, all_results):
    """Detections with a measured shape, minus the ones over the plate ring.

    Returns ``(formas, n_en_el_aro, n_sin_forma)``: the (detection, image
    path) pairs whose shape was measured and did not fall on the ring, how
    many detections were set apart for falling on the ring, and how many
    predictions could not be segmented at all. The depth profile, the forma
    section and the particle fichas all read the same list, so this runs
    exactly once per report. The ``polyx.core.calibration`` import stays
    lazy inside this body: it must not load at module import time (S01
    laziness gate).
    """
    # ── What falls on the plate ring is not sample ──
    # The ring is a bright, continuous band, and the detector confuses it with
    # material. Since it also comes out huge, it entered among the largest: in one
    # report it occupied an example card as `PET fibra` of 5978 um, which is
    # more than the 5 mm limit by which a microplastic is defined.
    #
    # The fitted circle was already known -- the scale comes from it -- and it only needed
    # to be asked. It is removed from the SIZE and from the cards, not from the count:
    # removing detections from the count is a study decision and not the
    # report generator's. What is done is to declare how many were set apart.
    from polyx.core.measure.calibration import about_the_ring

    _cals = getattr(state, "calibrations", None) or {}

    def _en_el_aro(det, path_2) -> bool:
        return about_the_ring(_cals.get(Path(path_2).name), det.x1, det.y1, det.x2, det.y2)

    # Each particle with its source image, so it can be found later.
    _shaped = [
        (p_, r.image_path)
        for r in all_results
        for p_ in r.predictions
        if getattr(p_, "aspect", None)
    ]
    shapes = [(d, path_2) for d, path_2 in _shaped if not _en_el_aro(d, path_2)]
    n_en_el_aro = len(_shaped) - len(shapes)
    n_formless = sum(
        1 for r in all_results for p_ in r.predictions if getattr(p_, "aspect", None) is None
    )
    return shapes, n_en_el_aro, n_formless


def _batch_metrics(state, solo_images):
    """Compute the batch-wide inputs every section builder consumes.

    Verbatim hoist (S03/T01) of the precomputation block that used to live
    inline in ``_generate_report_impl``: same empty-set filter semantics,
    same computation order, same ``Counter`` loops. Returns a plain tuple —
    no :class:`~typing.NamedTuple` and no new imports, so the lazy-import
    gate (``test_importes_diferidos_informe.py``) sees nothing new and no
    heavy module (``ultralytics``/``cv2``/``morfologia``/``calibracion``)
    gets touched at module level.
    """
    if solo_images is None:
        results_2 = dict(state.results)
    else:
        chosen = {Path(p) for p in solo_images}
        results_2 = {
            mi: [r for r in rs if Path(r.image_path) in chosen] for mi, rs in state.results.items()
        }

    all_results = [r for rs in results_2.values() for r in rs]
    total_imgs = len({r.image_path for r in all_results})
    total_dets = sum(len(r.predictions) for r in all_results)
    confs = [p.conf for r in all_results for p in r.predictions]
    sizes = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
    avg_conf = sum(confs) / len(confs) if confs else 0
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    any_gt = any(r.has_gt for r in all_results)

    classes_batch = _classes_views(all_results)

    # ── Distribution by class ──
    per_class = Counter()
    for r in all_results:
        for p in r.predictions:
            per_class[p.class_name] += 1

    return (
        results_2,
        all_results,
        total_imgs,
        total_dets,
        confs,
        sizes,
        avg_conf,
        avg_size,
        any_gt,
        classes_batch,
        per_class,
    )


# Legacy anchor names, to avoid breaking links to reports already
# distributed. A new section does NOT have to appear here: its own id already
# serves as an anchor. This dictionary was a second list to keep in
# sync with SECCIONES, and adding `profundidad` crashed with
# KeyError -- exactly what SECCIONES avoids by being a single list.
