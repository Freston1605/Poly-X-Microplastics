"""Report assembly: builds the document from the section builders.

The orchestration layer of the report: gathers the named precomputes, builds
one :class:`~polyx.gui.report.context.ReportContext`, and calls only the
builders the caller requested, in catalogue order. It is the only module that
knows the id -> builder mapping and the section list, which keeps the
dependency direction one-way (render -> sections -> figures -> batch) instead
of the cycle the package used to carry.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path

from polyx.core.detection_run import RunResult, RunStatus
from polyx.core.i18n import _, pgettext

from .batch import LotForms, compute_batch_metrics, compute_lot_forms
from .context import ReportContext
from .registry import IDS_SECTIONS, SECTIONS, _document_html
from .sections import (
    ShapeBodies,
    _run_status_note,
    build_calibration,
    build_comparison,
    build_count,
    build_depth,
    build_errors,
    build_gallery,
    build_methods,
    build_models,
    build_records,
    build_references,
    build_results,
    build_shape,
    build_size_by_folder,
    build_summary,
    compute_shape_bodies,
)

__all__ = ["_generate_report_impl", "generate_report"]

# The id -> builder mapping. Section order stays in ``SECTIONS``; this mapping
# only answers "which builder produces this section's body". The coverage
# invariant ``set(SECTION_BUILDERS) == set(IDS_SECTIONS)`` is pinned by the
# section tests.
SECTION_BUILDERS: dict[str, Callable[[ReportContext], str]] = {
    "resumen": build_summary,
    "metodos": build_methods,
    "calibracion": build_calibration,
    "forma": build_shape,
    "talla_carpetas": build_size_by_folder,
    "profundidad": build_depth,
    "fichas": build_records,
    "resultados": build_results,
    "modelos": build_models,
    "errores": build_errors,
    "comparacion": build_comparison,
    "galeria": build_gallery,
    "conteo": build_count,
    "referencias": build_references,
}


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

    ``sections`` is the set of :data:`SECTIONS` identifiers to include;
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

    active = list(run.models)
    status_note = _run_status_note(run)

    # ── Named precomputes ──
    # The batch metrics and the lot shapes live in polyx.gui.report.batch; the
    # shape pass runs once and hands back both of its section bodies. The lot
    # and shape precomputes are paid for only by the sections that read them:
    # a summary-only report never runs the shape pass (US20).
    batch = compute_batch_metrics(run, only_images)
    needs_lot = bool({"forma", "talla_carpetas", "profundidad", "fichas"} & requested)
    needs_shape_bodies = bool({"forma", "talla_carpetas"} & requested)
    lot = (
        compute_lot_forms(run, batch)
        if needs_lot
        else LotForms(shapes=[], ring_count=0, formless_count=0)
    )
    shape_bodies = (
        compute_shape_bodies(run, lot)
        if needs_shape_bodies
        else ShapeBodies(shape="", by_folder="")
    )
    ctx = ReportContext(
        run=run,
        batch=batch,
        lot=lot,
        shape_bodies=shape_bodies,
        max_gallery=max_gallery,
        max_records=max_records,
    )

    # ── Assemble HTML ──
    # Each block is (id, translated title, body). The body does NOT carry its
    # <h2>: the title and number are set by the document shell, which is the
    # only one that knows which sections survived the filter and therefore
    # which number belongs to each one. Only requested builders run, and an id
    # whose body is empty is dropped: the checkbox was ticked but there was
    # nothing to fill it with (errors without ground truth, for example).
    alive = []
    for sid, title in SECTIONS:
        if sid not in requested:
            continue
        body = SECTION_BUILDERS[sid](ctx)
        if body.strip():
            alive.append((sid, _section_title(title), body))

    # Cover, table of contents, section numbering and footer live in
    # _document_html: the impl remains the ordered list of calls that feeds it.
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
