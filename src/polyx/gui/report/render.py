"""Report assembly: builds the document from the section constructors.

The orchestration layer of the report: gathers the batch precomputations,
calls every section constructor in order and wraps the result in the
document shell. It is the only module that knows the section list, which
keeps the dependency direction one-way (render -> sections -> figures ->
_common) instead of the cycle the package used to carry.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from pathlib import Path

from polyx.core.detection_run import RunResult, RunStatus
from polyx.core.i18n import _, pgettext

from .batch import compute_batch_metrics, compute_lot_forms
from .registry import IDS_SECTIONS, SECTIONS, _document_html
from .sections import (
    _run_status_note,
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


def _section_title(message_id: str) -> str:
    """Resolve a section title from the registry.

    ``Summary`` is also the Trainer tab, where the translation differs; the
    report section keeps its own wording. The context is applied here, at
    render time, so the module-level registry still stores message ids only.
    """
    if message_id == "Summary":
        return pgettext("report section", "Summary")
    return _(message_id)


def _generate_report_impl(
    run: RunResult,
    output_path: Path,
    include_refs: bool = True,
    include_gallery: bool = True,
    max_gallery: int = 60,
    max_records: int = 12,
    only_images: Collection[Path | str] | None = None,
    sections: Collection[str] | None = None,
) -> Path:
    """Generates the HTML report from a canonical ``RunResult``.

    ``max_gallery`` limits how many images are embedded in the gallery.
    Images go into the HTML itself in base64, so without a cap a large
    batch produced an unmanageable file (14.5 MB with just 4 images).

    ``only_images`` narrows the report down to a set of paths. With
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
    active = list(run.models)
    status_note = _run_status_note(run)

    # ── Batch metrics ──
    # The named precompute lives in polyx.gui.report.batch; the assembler reads
    # its fields by name instead of unpacking a positional tuple.
    batch = compute_batch_metrics(run, only_images)

    # ── Error analysis: matrix + per-class metrics (main model) ──
    err_section = _section_errors(run, batch.results_by_model, batch.has_gt)

    # ── Verdict and confidence sweep ──
    verdict_html = _section_verdict(batch.results_by_model, run)

    # ── Models side by side, photo by photo ──
    compare_html = _section_comparison(run, batch.results_by_model)

    # ── Side-by-side gallery: Prediction vs Ground Truth ──
    gallery_html = _section_gallery(
        batch.results_by_model,
        run,
        batch.classes,
        max_gallery=max_gallery,
        include_gallery=include_gallery,
    )

    # ── Aggregate figures ──
    figures_html = _section_results(batch.per_class, batch.confidences, batch.sizes_um)

    # ── Methods ──
    methods_html, team_html, methods_for = _section_methods(
        run,
        active,
        batch.all_results,
        batch.image_count,
        batch.detection_count,
        batch.has_gt,
    )

    count_html = _section_count(batch.results_by_model, run, batch.classes)

    refs_html = _section_references(include_refs)

    # ── Scale calibration ──
    calib_html = _section_calibration(run)

    # ── Shape and size ──
    # It is separated from `Resultados` because it answers another question: not how many
    # particles there are, but what shape they have. Morphotype and curvature are
    # variables that the microplastics literature reports and that the detector
    # box cannot give. `shapes` is computed once because the depth profile
    # and the cards also consume it.
    lot = compute_lot_forms(run, batch)
    # The depth profile only exists if the names carry the stretch.
    depth_html = section_depth(lot.shapes, batch.all_results) if lot.shapes else ""
    shape_html, by_html_folder = _section_shape(run, lot.shapes, lot.ring_count, lot.formless_count)
    # ── Card for each particle ──
    records_html = _section_records(run, lot.shapes, max_records)

    # ── Assemble HTML ──
    # Each block is (id, body). The body does NOT carry its <h2>: the title and
    # number are set by the assembler, which is the only one that knows which sections
    # survived the filter and therefore which number belongs to each one.
    bodies = {
        "resumen": _section_summary(
            batch.image_count,
            batch.detection_count,
            batch.mean_confidence,
            batch.mean_size_um,
            active,
            batch.has_gt,
        ),
        "metodos": f"""
<table class='data'><tr>{_("<th>Parameter</th><th>Value</th>")}</tr>{methods_html}</table>
{team_html}
{methods_for}""",
        "calibracion": calib_html,
        "forma": shape_html,
        "talla_carpetas": by_html_folder,
        "profundidad": depth_html,
        "fichas": records_html,
        "resultados": figures_html,
        "modelos": _section_models(run, batch.results_by_model, batch.all_results, batch.has_gt),
        "errores": err_section,
        "comparacion": f"{verdict_html}\n{compare_html}",
        "galeria": gallery_html,
        "conteo": count_html,
        "referencias": refs_html,
    }

    # An id in ``pedidas`` with an empty body is dropped here: the checkbox was ticked but
    # there was nothing to fill it with (errors without ground truth, count without samples).
    # ``SECTIONS`` holds message ids. The interface already resolved them for its
    # checkboxes; the document resolves them here against the active catalog, so a
    # saved report keeps the language it was generated in.
    alive = [
        (sid, _section_title(title), bodies.get(sid, ""))
        for sid, title in SECTIONS
        if sid in requested and (bodies.get(sid) or "").strip()
    ]

    # Cover, table of contents, section numbering and footer live in _documento_html:
    # the impl remains the ordered list of calls that feeds cuerpos.
    html = _document_html(alive, active, status_note)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_report(
    run: RunResult,
    output_path: Path,
    include_refs: bool = True,
    include_gallery: bool = True,
    max_gallery: int = 60,
    max_records: int = 12,
    only_images: Collection[Path | str] | None = None,
    sections: Collection[str] | None = None,
) -> Path:
    """Generates the HTML report from a canonical ``RunResult``.

    A completed, aborted or failed run is reportable: a partial batch is real
    data, and the document carries a visible status banner so it is never
    mistaken for a finished one. A run still in progress is refused because it
    has not finished writing its results.

    Generates the HTML report, measuring it and recording report.generate /
    report.finish / report.error in the detector JSONL. Re-throws failures:
    the caller decides what to do, the log keeps the diagnosis.
    """
    from polyx.core.provenance import logger_for, timer

    status = RunStatus.parse(run.status)
    if status is RunStatus.RUNNING:
        raise ValueError(
            _("report requires a finished run, not %(status)r") % {"status": status.value}
        )
    n_results = sum(len(rs) for rs in (run.results or {}).values())
    with timer(
        logger_for("detector"),
        "report",
        results_2=n_results,
        output=str(output_path),
        sections=sorted(sections) if sections else None,
        solo_images=(len(only_images) if only_images is not None else None),
    ):
        return _generate_report_impl(
            run,
            output_path,
            include_refs=include_refs,
            include_gallery=include_gallery,
            max_gallery=max_gallery,
            max_records=max_records,
            only_images=only_images,
            sections=sections,
        )
