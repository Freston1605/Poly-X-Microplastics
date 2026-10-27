"""Samples, ordering, sampling and batch aggregation for the report.

Plate/station naming (Loa river), ``_sample_of``, ``_clue_order``,
``choose_records`` (the biased fiber-aware sample), ``_lot_forms`` (which
detections feed the shape/fichas sections, minus the ring) and
``_batch_metrics`` (the precomputation block every section builder reads).
These keep their lazy imports (``polyx.core.measure.calibration`` inside
``_lot_forms``) so the import-laziness gate stays satisfied.

Re-exported from ``polyx.gui.report._common`` on purpose: the package
surface, ``render.py`` and the section builders import them from there, and
nothing external needs to know this module exists.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from polyx.core.report_metrics import _classes_views

# The plates from the Loa river are named `<tramo>.<testigo>` with an `x`/`xx` suffix for
# the second and third plate of the same stretch. Crops add `__f<row>c<col>`.
_RE_PLATE = __import__("re").compile(r"^(\d+)\.(\d+)(x*)$")
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


_MAX_FOLDER_PER_PHOTO = 6


def choose_records(shapes: list[tuple], max_records: int) -> list[tuple]:
    """Which particles are shown with their measurement drawn, and in what order.

    ``formas`` is a list of ``(Detection, ruta)`` pairs. At most
    ``max_fichas`` of them are returned.

    Before, the FIRST ones by file name order entered, which had two bad
    consequences:

      * The LARGEST particle — the one backing any claim about maximum
        size — could be left out for living in a high-numbered photo.
      * Fibers never showed up in practice. They are ~1% of the material,
        so the cut arrived long before finding one. And the fiber is
        exactly the case where the geodesic diameter acts: if no
        measurement is seen, there is no way to judge whether that method
        works.

    Now the sample is split between the two morphotypes, in halves, and
    within each one size rules. That way the largest of each type always
    enters.

    The split is not rigid: if there are not enough fibers — the normal
    case — the biggest particles fill the gap instead of returning a
    short sample. And the other way around too.

    It is a deliberately BIASED sample, not representative of the batch;
    the report declares that where it shows it.
    """
    if max_records <= 0:
        return []

    def size_2(par) -> float:
        return par[0].length_um or 0.0

    fibers = sorted([t for t in shapes if t[0].morphotype == "fibra"], key=size_2, reverse=True)
    others = sorted([t for t in shapes if t[0].morphotype != "fibra"], key=size_2, reverse=True)

    half = max(1, max_records // 2)
    chosen = fibers[:half]
    chosen += others[: max_records - len(chosen)]
    # Gaps left by the scarce group are covered by the other one, so as not to deliver a
    # shorter sample than requested when material is available.
    if len(chosen) < max_records:
        ya = {id(d) for d, _ in chosen}
        leftovers = [t for t in fibers + others if id(t[0]) not in ya]
        chosen += leftovers[: max_records - len(chosen)]
    return chosen


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
