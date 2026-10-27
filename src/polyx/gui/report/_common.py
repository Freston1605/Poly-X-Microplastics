"""HTML report generator for the detection study (self-contained, base64 images).

Contents:
  1. Abstract (executive summary)
  2. Methods (model, params, calibration, device)
  3. General results (class, conf, size, P/R/F1)
  4. Per-model summary
  5. Error analysis (confusion matrix + gallery)
  6. Model comparison
  7. Full gallery per annotated image
  8. Per-sample and per-plastic-type count
  9. Bibliographic references
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# matplotlib in non-interactive mode
import matplotlib
import numpy as np

matplotlib.use("Agg")

from polyx.core.i18n import language, tr
from polyx.core.report_metrics import _classes_views
from polyx.gui import theme as T

# Machine provenance (equipment table, checkpoint training info) lives in
# machine.py. It is re-exported HERE, not imported by its readers, because
# the section builders read it through this module at call time and that is
# what the byte-identity fixtures patch.
from .machine import _team as _team
from .machine import _training_of as _training_of

# HTML primitives (escape, stylesheet, base64 embeds, tables) live in
# markup.py; re-exported here because that is where the section builders
# and the package surface import them from.
from .markup import (
    _WIDTH_GALLERY as _WIDTH_GALLERY,
)
from .markup import (
    _esc as _esc,
)
from .markup import (
    _img_b64 as _img_b64,
)
from .markup import (
    _report_css as _report_css,
)
from .markup import (
    _table_count as _table_count,
)
from .markup import (
    _table_width as _table_width,
)
from .markup import (
    _uri_gallery as _uri_gallery,
)

# The plates from the Loa river are named `<tramo>.<testigo>` with an `x`/`xx` suffix for
# the second and third plate of the same stretch. Crops add `__f<row>c<col>`.
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


# Three presentation colours of the report (fixed palette: a report is
# printed and archived, so it must not follow the UI theme).
_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


_MAX_GROUPS_BOXPLOT = 12
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


def _record_particle(det, image_path_2, um_por_px: float | None, title: str) -> str:
    """Crop of a particle with its mask and the px-to-µm count.

    The complete conversion is shown — pixels measured, the scale of
    THAT photo and the result — because a size in µm without the count
    that produced it cannot be checked, and this is the particle that
    carries the most weight in any conclusion about maximum size.
    """
    if det is None or not image_path_2:
        return ""
    try:
        # cv2 is imported here and not at the top: this module does so in the rest of
        # the functions to avoid loading OpenCV when only text is requested.
        import cv2

        from polyx.core.measure.calibration import read_image
        from polyx.core.measure.morphology import box_margin, segment

        bgr = read_image(image_path_2)
        if bgr is None:
            return ""
        # THE SAME MARGIN THAT segmentar() USES, not a similar one. The mask comes
        # cropped with margen_de_caja(); if it is cropped here with another number, the
        # contour is drawn from a different origin and ends up displaced over the
        # photo. It already happened with a 0.35 copied by hand against the real 0.50, and it was seen
        # by eye in the report: `la mascara quedo corrida`.
        mg = box_margin(det.x2 - det.x1, det.y2 - det.y1)
        x1 = max(0, int(det.x1) - mg)
        y1 = max(0, int(det.y1) - mg)
        x2 = min(bgr.shape[1], int(det.x2) + mg)
        y2 = min(bgr.shape[0], int(det.y2) + mg)
        sub = bgr[y1:y2, x1:x2].copy()
        if sub.size == 0:
            return ""
        mask = segment(bgr, det.x1, det.y1, det.x2, det.y2)
        ov = sub.copy()
        if mask is not None:
            h = min(ov.shape[0], mask.shape[0])
            w = min(ov.shape[1], mask.shape[1])
            # Contour and not fill: covering the particle in green prevents judging
            # whether the mask really follows it, which is exactly what is shown.
            cont, _ = cv2.findContours(mask[:h, :w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov[:h, :w], cont, -1, (0, 255, 0), max(1, min(ov.shape[:2]) // 90))
        par = np.hstack([sub, ov])
        scale = min(4.0, 620 / max(1, par.shape[1]))
        if scale > 1:
            par = cv2.resize(par, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", par)
        if not ok:
            return ""
        img = f"<img src='data:image/png;base64,{_img_b64(buf.tobytes())}' style='max-width:100%'>"
    except Exception:  # noqa: BLE001
        # A particle that cannot be cropped cannot bring down the report
        # entirely: the card is illustrative and the tables are already computed.
        return ""

    length_px_2 = (det.length_um / um_por_px) if (det.length_um and um_por_px) else None
    account = ""
    if length_px_2 and um_por_px:
        account = (
            f"<p style='font-family:monospace;font-size:10pt'>"
            f"{length_px_2:.1f} px &times; {um_por_px:.4f} µm/px = "
            f"<strong>{det.length_um:.0f} µm</strong></p>"
        )
        # The three measurements together: if the equivalent rectangle jumps
        # relative to the other two, the particle edge is jagged and the
        # size deserves a look.
        others = []
        if det.feret_um:
            others.append(
                f"<tr><td>{tr('Feret máximo (cuerda)')}</td><td>{det.feret_um:.0f}</td></tr>"
            )
        if det.geodesic_um:
            others.append(
                f"<tr><td>{tr('Geodésico (sigue la curva)')}</td><td>{det.geodesic_um:.0f}</td></tr>"
            )
        if others:
            account += (
                f"<table class='data'><tr>{tr('<th>Medida</th><th>µm</th>')}</tr>"
                f"{''.join(others)}</table>"
                f"<p>{tr('Se reporta la mayor de las dos primeras')}"
                + (
                    tr(", en este caso por <strong>{metodo}</strong>").format(
                        metodo=tr(det.method_length)
                    )
                    if det.method_length
                    else ""
                )
                + ".</p>"
            )
    return (
        f"<h3>{NUM}.{title}</h3>"
        + tr(
            "<p>A la izquierda la partícula; a la derecha, en verde, el "
            "contorno que se midió sobre ella.</p>"
        )
        + f"{img}{account}"
    )


SECTIONS = [
    ("resumen", "Resumen"),
    ("metodos", "Métodos"),
    ("calibracion", "Calibración de escala"),
    ("forma", "Forma y talla de las partículas"),
    ("talla_carpetas", "Talla por carpeta y por foto (comparación entre sitios)"),
    ("profundidad", "Perfil en profundidad del testigo"),
    ("fichas", "Ficha de partículas medidas (muestra)"),
    ("resultados", "Resultados generales"),
    ("modelos", "Resumen por modelo"),
    ("errores", "Análisis de errores"),
    ("comparacion", "Comparación entre modelos"),
    ("galeria", "Galería por imagen"),
    ("conteo", "Conteo por muestra y tipo de plástico"),
    ("referencias", "Referencias bibliográficas"),
]

IDS_SECTIONS = [s[0] for s in SECTIONS]

# Frequently used combinations. They exist because the real problem was not that choosing
# was impossible, but that choosing ten checkboxes one by one is tedious and
# nobody does it: with a preset the short report is one click away.
PRESETS = {
    "completo": IDS_SECTIONS,
    "resumen": ["resumen", "calibracion", "forma", "conteo", "galeria"],
    "metodologico": [
        "resumen",
        "metodos",
        "calibracion",
        "forma",
        "fichas",
        "modelos",
        "errores",
        "comparacion",
        "referencias",
    ],
}

# Marker that is replaced by the final section number during assembly.
# Numbers cannot be written by hand in the titles because they depend on which
# other sections were requested.
NUM = "@@N@@"


# ── Report section builders ──
# M004/S02: _generate_report_impl was a 1200+ line function that
# built each section inline. Each block is extracted into a module-level
# builder that returns ONLY the HTML body, without its <h2>: the title and the
# number are still set by the assembler (NUM -> `N`), which is the only one that
# knows which sections survived the filter. The byte identity of the render
# against the frozen baseline (tests/fixtures/baseline_informe/) is the slice
# contract: changing a literal, a tr(), an _esc or the order of computation breaks
# tests/test_render_reporte_benigno.py.


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
_ANCHORS = {
    "resumen": "abstract",
    "metodos": "methods",
    "calibracion": "calib",
    "resultados": "results",
    "modelos": "models",
    "errores": "errors",
    "comparacion": "compare",
    "galeria": "gallery",
    "referencias": "refs",
}


def anchor(sid: str) -> str:
    """Anchor name for a section id: legacy alias if any, the id itself otherwise."""
    return _ANCHORS.get(sid, sid)


def _document_html(alive, active) -> str:
    """Assemble the final document: cover header, TOC, numbered sections, footer.

    ``vivas`` arrives as (id, translated title, body) tuples in SECCIONES
    order, already filtered to the requested sections that have content.
    The enumeration (SECCIONES order, start=1), the ``<h2>`` titles and the
    ``{NUM}`` placeholder substitution all live here because this helper is
    the only place that knows the final order; moving or reordering them is
    byte-visible in the frozen render (MEM090).
    """
    toc_html = "".join(f"<li><a href='#{anchor(sid)}'>{title}</a></li>" for sid, title, _ in alive)
    sections_html = "".join(
        f"<h2 id='{anchor(sid)}'>{i}. {title}</h2>\n{body.replace(NUM, str(i))}\n"
        for i, (sid, title, body) in enumerate(alive, start=1)
    )

    return f"""<!doctype html>
<html lang='{language()}'><head><meta charset='utf-8'>
<title>{tr("Informe de detección")} · Poly-X</title>
<style>{_report_css()}</style></head><body>
<div class='container'>

<header class='cover'>
  <div class='kicker'>Poly-X · {tr("Informe de detección")}</div>
  <h1>{tr("Informe de detección de microplásticos")}<br>{tr("por fluorescencia Nile Red")}</h1>
  <p class='meta'><strong>{tr("Autor:")}</strong> Cristofher Ferrada &middot;
    <strong>{tr("Modelos:")}</strong> {_esc(", ".join(s.alias for s in active)) or "—"}</p>
</header>

<div class='toc'>
  <h3>📑 {tr("Índice")}</h3>
  <ol>{toc_html}</ol>
</div>

{sections_html}
<footer>
  © Cristofher Ferrada · {tr("Generado por Poly-X")}<br>
  {tr("Suite de detección de microplásticos por fluorescencia Nile Red (254 nm) e IA (YOLO v8/v11).")}
</footer>
</div></body></html>
"""
