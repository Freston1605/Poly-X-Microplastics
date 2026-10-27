"""Report assembly: builds the document from the section constructors.

The orchestration layer of the report: gathers the batch precomputations,
calls every section constructor in order and wraps the result in the
document shell. It is the only module that knows the section list, which
keeps the dependency direction one-way (render -> sections -> figures ->
_common) instead of the cycle the package used to carry.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from polyx.core.i18n import tr

from ._common import IDS_SECTIONS, SECTIONS, _batch_metrics, _document_html, _lot_forms
from .sections import (
    _section_calibration,
    _section_comparison,
    _section_count,
    _section_errors,
    _section_gallery,
    _section_methods,
    _section_models,
    _section_records,
    _section_references,
    _section_results,
    _section_shape,
    _section_summary,
    _section_verdict,
    section_depth,
)

__all__ = ["_generate_report_impl", "generate_report"]


def _generate_report_impl(
    state,
    output_path: Path,
    include_refs: bool = True,
    include_gallery: bool = True,
    max_gallery: int = 60,
    max_records: int = 12,
    solo_images=None,
    sections=None,
) -> Path:
    """Generates the HTML report. `state` is a ``DetectorState`` with results.

    ``max_gallery`` limits how many images are embedded in the gallery.
    Images go into the HTML itself in base64, so without a cap a large
    batch produced an unmanageable file (14.5 MB with just 4 images).

    ``solo_imagenes`` narrows the report down to a set of paths. With
    ``None`` the whole batch is in. The filter is applied before anything
    is computed, so that the totals, the charts and the confusion matrix
    describe exactly the same subset being shown; if only the gallery
    were filtered, the figures above would talk about one batch and the
    images about another.

    ``secciones`` is the set of :data:`SECCIONES` identifiers to include;
    ``None`` includes all of them. A requested section that has no
    content — errors without ground truth, for example — is skipped
    anyway: ticking it does not invent data that does not exist.
    """
    requested = set(IDS_SECTIONS if sections is None else sections)
    # The two old checkboxes still take precedence if passed as False, so as not to
    # change the meaning of calls that already existed.
    if not include_refs:
        requested.discard("referencias")
    if not include_gallery:
        requested.discard("galeria")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")  # noqa: F841
    active = state.active_models()

    # ── Batch metrics ──
    # The precomputation block now lives in _metricas_de_lote; the impl only
    # unpacks. The tuple order is the order in which the assembler
    # consumes them, so the unpacking names the same variables as
    # always.
    (
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
    ) = _batch_metrics(state, solo_images)

    # ── Error analysis: matrix + per-class metrics (main model) ──
    err_section = _section_errors(state, results_2, active, any_gt)

    # ── Verdict and confidence sweep ──
    verdict_html = _section_verdict(results_2, state)

    # ── Models side by side, photo by photo ──
    compare_html = _section_comparison(state, results_2)

    # ── Side-by-side gallery: Prediction vs Ground Truth ──
    gallery_html = _section_gallery(
        results_2, state, classes_batch, max_gallery=max_gallery, include_gallery=include_gallery
    )

    # ── Aggregate figures ──
    figures_html = _section_results(per_class, confs, sizes)

    # ── Methods ──
    methods_html, team_html, methods_for = _section_methods(
        state, active, all_results, total_imgs, total_dets, any_gt
    )

    count_html = _section_count(results_2, state, active, classes_batch)

    refs_html = _section_references(include_refs)

    # ── Scale calibration ──
    calib_html = _section_calibration(state)

    # ── Shape and size ──
    # It is separated from `Resultados` because it answers another question: not how many
    # particles there are, but what shape they have. Morphotype and curvature are
    # variables that the microplastics literature reports and that the detector
    # box cannot give. `formas` is computed once because the depth profile
    # and the cards also consume it.
    shapes, n_en_el_aro, n_formless = _lot_forms(state, all_results)
    # The depth profile only exists if the names carry the stretch.
    depth_html = section_depth(shapes, all_results) if shapes else ""
    shape_html, by_html_folder = _section_shape(state, shapes, n_en_el_aro, n_formless)
    # ── Card for each particle ──
    records_html = _section_records(state, shapes, max_records)

    # ── Assemble HTML ──
    # Each block is (id, body). The body does NOT carry its <h2>: the title and
    # number are set by the assembler, which is the only one that knows which sections
    # survived the filter and therefore which number belongs to each one.
    bodies = {
        "resumen": _section_summary(total_imgs, total_dets, avg_conf, avg_size, active, any_gt),
        "metodos": f"""
<table class='data'><tr>{tr("<th>Parámetro</th><th>Valor</th>")}</tr>{methods_html}</table>
{team_html}
{methods_for}""",
        "calibracion": calib_html,
        "forma": shape_html,
        "talla_carpetas": by_html_folder,
        "profundidad": depth_html,
        "fichas": records_html,
        "resultados": figures_html,
        "modelos": _section_models(state, results_2, all_results, any_gt),
        "errores": err_section,
        "comparacion": f"{verdict_html}\n{compare_html}",
        "galeria": gallery_html,
        "conteo": count_html,
        "referencias": refs_html,
    }

    # An id in ``pedidas`` with an empty body is dropped here: the checkbox was ticked but
    # there was nothing to fill it with (errors without ground truth, count without samples).
    # tr() on the title: SECCIONES stores the Spanish version, which is the
    # dictionary key. The interface already translated it for its checkboxes; the
    # document did not, and came out with the titles in Spanish even though the rest did not.
    alive = [
        (sid, tr(title), bodies.get(sid, ""))
        for sid, title in SECTIONS
        if sid in requested and (bodies.get(sid) or "").strip()
    ]

    # Cover, table of contents, section numbering and footer live in _documento_html:
    # the impl remains the ordered list of calls that feeds cuerpos.
    html = _document_html(alive, active)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_report(
    state,
    output_path: Path,
    include_refs: bool = True,
    include_gallery: bool = True,
    max_gallery: int = 60,
    max_records: int = 12,
    solo_images=None,
    sections=None,
) -> Path:
    """Generates the HTML report with an observability wrapper.

    Generates the HTML report, measuring it and recording report.generate /
    report.finish / report.error in the detector JSONL. Re-throws failures:
    the caller decides what to do, the log keeps the diagnosis.
    """
    from polyx.core.observability import logger_for, timer

    n_results = sum(len(rs) for rs in (state.results or {}).values())
    with timer(
        logger_for("detector"),
        "report",
        results_2=n_results,
        output=str(output_path),
        sections=sorted(sections) if sections else None,
        solo_images=(len(solo_images) if solo_images is not None else None),
    ):
        return _generate_report_impl(
            state,
            output_path,
            include_refs=include_refs,
            include_gallery=include_gallery,
            max_gallery=max_gallery,
            max_records=max_records,
            solo_images=solo_images,
            sections=sections,
        )
