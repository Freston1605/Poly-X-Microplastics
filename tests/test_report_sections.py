"""One builder per report section, fed by a ``ReportContext`` (#85).

Every section id in the catalogue has exactly one builder in
``polyx.gui.report.sections`` and one entry in the assembler's
``SECTION_BUILDERS`` mapping. A test builds the context from the benign
fixture with the same precomputes production uses and calls one builder at a
time, so a broken section names itself instead of surfacing as a whole-document
diff.

The baseline is the semantic gate: section body against section body, never the
whole document and never a hash of it. Bodies that carry the ``{NUM}`` numbering
placeholder cannot be compared literally (the baseline has the resolved number),
so those assert the placeholder instead.

The baseline was captured in English, so the autouse fixture selects English
through ``i18n.set_language`` and restores the suite's language afterwards.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

# English baseline: read-only in the suite. Recapture it with
# tools/audit/capture_report_fixtures.py only after a conscious report change.
BASELINE_HTML = (
    Path(__file__).resolve().parent / "fixtures" / "baseline_informe" / "benigna" / "informe.html"
)

# The baseline figures are matplotlib base64 PNGs whose bytes change with the
# OS renderer/fonts even at the same pinned version. Tests that check an
# isolated SECTION BODY against the whole baseline normalise the figures to a
# CONSTANT marker (<FIG>): the isolated body renumbers its ordinal marker from
# 1, while the baseline numbers them from their global position, so an ordinal
# marker does not match between the two. The position of each figure inside the
# whole report is not checked here: this gate fixes each section's structure,
# not the whole document.
PNG_URI_RE = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")


def _normalise_figures(html: str) -> str:
    """Replace every PNG data-URI with the constant marker ``<FIG>``."""
    return PNG_URI_RE.sub("data:image/png;base64,<FIG>", html)


# The baseline already normalised with the constant marker (only structure is
# checked): the single place that reads the file. Section bodies are matched
# against this variant, each one normalised with the same function.
BASELINE_TXT = _normalise_figures(BASELINE_HTML.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _baseline_language():
    """Select the baseline language (en) and restore the suite's afterwards.

    The baseline HTML was captured in English; the section bodies compared
    against it need the same language. The switch goes through the public
    ``set_language`` seam, and POLYX_LOG_DIR is already isolated by the autouse
    fixture in tests/conftest.py (decision D007); no report is generated here,
    so there is no logger to silence.
    """
    from polyx.core import i18n

    previous = i18n.language()
    assert i18n.set_language("en")
    yield
    i18n.set_language(previous)


def _benign_state(tmp_path: Path):
    """The shared benign ``RunResult``, written under ``tmp_path``."""
    from tests.report_fixtures import estado_benigno as benign_state

    return benign_state(directorio=tmp_path / "photos")


def _report_context(state):
    """Build the context exactly as the assembler does, from named precomputes."""
    from polyx.gui.report import batch, sections
    from polyx.gui.report.context import ReportContext

    metrics = batch.compute_batch_metrics(state, None)
    lot = batch.compute_lot_forms(state, metrics)
    return ReportContext(
        run=state,
        batch=metrics,
        lot=lot,
        shape_bodies=sections.compute_shape_bodies(state, lot),
        max_gallery=60,
        max_records=12,
    )


def _benign_context(tmp_path: Path):
    """The benign state plus its report context, with the baseline recipe."""
    return _report_context(_benign_state(tmp_path))


def test_report_context_carries_the_named_inputs(tmp_path):
    """The frozen context carries exactly the six named inputs."""
    from polyx.gui.report.context import ReportContext

    assert [field.name for field in fields(ReportContext)] == [
        "run",
        "batch",
        "lot",
        "shape_bodies",
        "max_gallery",
        "max_records",
    ]
    context = _benign_context(tmp_path)
    assert context.max_gallery == 60
    assert context.max_records == 12
    with pytest.raises(FrozenInstanceError):
        context.max_gallery = 0


def test_shape_bodies_is_a_frozen_named_result():
    """The shape section's two bodies come from one named, immutable result."""
    from polyx.gui.report.sections import ShapeBodies

    assert [field.name for field in fields(ShapeBodies)] == ["shape", "by_folder"]
    bodies = ShapeBodies(shape="shape", by_folder="by-folder")
    assert (bodies.shape, bodies.by_folder) == ("shape", "by-folder")
    with pytest.raises(FrozenInstanceError):
        bodies.shape = "other"


def test_section_builders_cover_every_section_id():
    """``SECTION_BUILDERS`` keys are exactly the catalogue's section ids."""
    import polyx.gui.report.render as render
    from polyx.gui.report.registry import IDS_SECTIONS

    assert set(render.SECTION_BUILDERS) == set(IDS_SECTIONS)
    assert len(render.SECTION_BUILDERS) == len(IDS_SECTIONS)
    assert all(callable(builder) for builder in render.SECTION_BUILDERS.values())


def test_every_builder_returns_a_string(tmp_path):
    """Every builder accepts the context and returns a string body."""
    import polyx.gui.report.render as render

    context = _benign_context(tmp_path)
    for section_id, builder in render.SECTION_BUILDERS.items():
        body = builder(context)
        assert isinstance(body, str), f"build {section_id} did not return a string"


def test_summary_body_matches_the_baseline(tmp_path):
    """Summary: non-empty, str, no <h2>, and literal in the baseline."""
    import polyx.gui.report.sections as sections

    body = sections.build_summary(_benign_context(tmp_path))
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert _normalise_figures(body) in BASELINE_TXT


def test_errors_body_is_empty_without_ground_truth(tmp_path):
    """Errors: the benign batch has no GT, so the legitimate body is ''."""
    import polyx.gui.report.sections as sections

    body = sections.build_errors(_benign_context(tmp_path))
    assert body == ""
    assert "<h2 id='errors'>" not in BASELINE_TXT


def test_models_body_names_the_alias(tmp_path):
    """Per-model summary: non-empty, names the benign alias, no <h2>."""
    import polyx.gui.report.sections as sections

    body = sections.build_models(_benign_context(tmp_path))
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert "m1" in body  # tests.report_fixtures.ALIAS_BENIGNO
    assert _normalise_figures(body) in BASELINE_TXT


def test_comparison_body_without_a_second_model_explains_why(tmp_path):
    """Comparison: one model means no comparison, but the caption is rendered."""
    import polyx.gui.report.sections as sections

    body = sections.build_comparison(_benign_context(tmp_path))
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert _normalise_figures(body) in BASELINE_TXT


def test_gallery_body_is_empty_without_prediction_bytes(tmp_path):
    """Gallery: the fixture publishes no annotated bytes, so the body is ''."""
    import polyx.gui.report.sections as sections

    body = sections.build_gallery(_benign_context(tmp_path))
    assert body == ""
    assert "<h2 id='gallery'>" not in BASELINE_TXT


def test_results_body_carries_the_aggregate_figures(tmp_path):
    """Results: body with embedded figures (base64), no <h2>."""
    import polyx.gui.report.sections as sections

    body = sections.build_results(_benign_context(tmp_path))
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert "base64," in body
    assert _normalise_figures(body) in BASELINE_TXT


def test_methods_body_composes_its_three_parts(tmp_path):
    """Methods: parameters table, team block and redacted paragraph in one body."""
    import polyx.gui.report.sections as sections
    from polyx.gui.report.registry import NUM

    body = sections.build_methods(_benign_context(tmp_path))
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert "<table class='data'>" in body
    assert "Computing hardware" in body
    assert "<blockquote" in body
    # The table sub-headings are numbered by the assembler, never here.
    assert NUM in body


def test_references_body_is_the_static_list(tmp_path):
    """References: the bibliography list, no <h2>."""
    import polyx.gui.report.sections as sections

    body = sections.build_references(_benign_context(tmp_path))
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert "<ol" in body
    assert _normalise_figures(body) in BASELINE_TXT


def test_calibration_body_is_empty_without_calibrations(tmp_path):
    """Calibration: the benign state has no calibrations, so the body is ''."""
    import polyx.gui.report.sections as sections

    assert sections.build_calibration(_benign_context(tmp_path)) == ""
    assert "<h2 id='calib'>" not in BASELINE_TXT


def test_count_body_uses_the_numbering_placeholder(tmp_path):
    """Counts: non-empty body with the ``{NUM}`` sub-heading placeholder."""
    import polyx.gui.report.sections as sections
    from polyx.gui.report.registry import NUM

    body = sections.build_count(_benign_context(tmp_path))
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert NUM in body


def test_depth_body_is_empty_without_a_depth_stretch(tmp_path):
    """Depth: the benign names carry no depth stretch, so the body is ''."""
    import polyx.gui.report.sections as sections

    assert sections.build_depth(_benign_context(tmp_path)) == ""
    assert "<h2 id='profundidad'>" not in BASELINE_TXT


def test_shape_and_folder_bodies_come_from_the_shared_precompute(tmp_path):
    """Shape: both bodies are the named precompute, built in one pass."""
    import polyx.gui.report.sections as sections
    from polyx.gui.report.registry import NUM

    context = _benign_context(tmp_path)
    shape_body = sections.build_shape(context)
    folder_body = sections.build_size_by_folder(context)
    assert isinstance(shape_body, str) and shape_body
    assert "<h2" not in shape_body
    assert NUM in shape_body
    assert folder_body == ""
    assert shape_body == context.shape_bodies.shape
    assert folder_body == context.shape_bodies.by_folder


def test_builders_return_the_context_shape_bodies_verbatim(tmp_path):
    """Both shape builders read the context sentinel; neither recomputes it."""
    import polyx.gui.report.sections as sections
    from polyx.gui.report.sections import ShapeBodies

    context = dataclasses.replace(
        _benign_context(tmp_path),
        shape_bodies=ShapeBodies(shape="<shape-body>", by_folder="<folder-body>"),
    )
    assert sections.build_shape(context) == "<shape-body>"
    assert sections.build_size_by_folder(context) == "<folder-body>"


def test_records_body_cards_and_sample_cap(tmp_path):
    """Records: cards with base64 images; empty with no shapes or a zero cap."""
    import polyx.gui.report.sections as sections

    context = _benign_context(tmp_path)
    body = sections.build_records(context)
    assert isinstance(body, str) and body
    assert "<h2" not in body
    assert "base64," in body
    # Grouped by image: the photo name arrives escaped in an <h3>.
    assert "<h3>placa_muestra_a.png</h3>" in body
    assert _normalise_figures(body) in BASELINE_TXT
    # Boundaries: no shapes or a zero cap produce the empty body.
    no_shapes = dataclasses.replace(context, lot=dataclasses.replace(context.lot, shapes=[]))
    assert sections.build_records(no_shapes) == ""
    assert sections.build_records(dataclasses.replace(context, max_records=0)) == ""


def test_the_assembler_is_a_short_section_loop():
    """The impl stays under 150 lines and delegates to ``SECTION_BUILDERS``.

    The line cap is measured with ast over the source (not the live module) so
    a new inline computation cannot grow the assembler unnoticed. The name
    checks prove it iterates ``SECTIONS`` through the mapping and no longer
    calls the private section helpers directly.
    """
    import polyx.gui.report.render as render

    impl_path = Path(render._generate_report_impl.__code__.co_filename).resolve()
    source = impl_path.read_text(encoding="utf-8")
    impl = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_generate_report_impl"
    )

    line_count = impl.end_lineno - impl.lineno + 1
    assert line_count <= 150, (
        f"_generate_report_impl grew to {line_count} lines (cap 150): "
        "an inline computation block is back and must move to a helper"
    )

    names = {node.id for node in ast.walk(impl) if isinstance(node, ast.Name)}
    assert "SECTION_BUILDERS" in names
    assert "SECTIONS" in names

    called = {
        node.func.id
        for node in ast.walk(impl)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    direct_helpers = {
        name for name in called if name.startswith("_section_") and name != "_section_title"
    }
    assert not direct_helpers, f"the assembler still calls private helpers: {direct_helpers}"


def test_the_assembler_builds_only_the_requested_sections(tmp_path, monkeypatch):
    """A section the user did not tick is not computed; empty ones are dropped."""
    import polyx.gui.report.render as render

    state = _benign_state(tmp_path)
    called: list[str] = []

    def recording(section_id, builder):
        def wrapped(context):
            called.append(section_id)
            return builder(context)

        return wrapped

    for section_id in list(render.SECTION_BUILDERS):
        monkeypatch.setitem(
            render.SECTION_BUILDERS,
            section_id,
            recording(section_id, render.SECTION_BUILDERS[section_id]),
        )

    render._generate_report_impl(state, tmp_path / "summary.html", sections=["resumen"])
    assert called == ["resumen"]

    called.clear()
    render._generate_report_impl(state, tmp_path / "errors.html", sections=["errores"])
    assert called == ["errores"]
    html = (tmp_path / "errors.html").read_text(encoding="utf-8")
    assert "<h2 id='errors'>" not in html


def test_the_assembler_precomputes_only_what_the_sections_need(tmp_path, monkeypatch):
    """The lot shapes and shape bodies run only for sections that read them."""
    import polyx.gui.report.render as render

    calls: list[str] = []
    real_lot = render.compute_lot_forms
    real_bodies = render.compute_shape_bodies

    def recording_lot(*args, **kwargs):
        calls.append("lot")
        return real_lot(*args, **kwargs)

    def recording_bodies(*args, **kwargs):
        calls.append("bodies")
        return real_bodies(*args, **kwargs)

    monkeypatch.setattr(render, "compute_lot_forms", recording_lot)
    monkeypatch.setattr(render, "compute_shape_bodies", recording_bodies)

    state = _benign_state(tmp_path)
    render._generate_report_impl(state, tmp_path / "summary.html", sections=["resumen"])
    assert calls == []

    calls.clear()
    render._generate_report_impl(state, tmp_path / "depth.html", sections=["profundidad"])
    assert calls == ["lot"]

    calls.clear()
    render._generate_report_impl(state, tmp_path / "shape.html", sections=["forma"])
    assert calls == ["lot", "bodies"]


def test_include_refs_false_drops_the_references_section(tmp_path):
    """``include_refs=False`` removes the anchor and the bibliography."""
    from polyx.gui.report_html import generate_report

    state = _benign_state(tmp_path)
    included = generate_report(state, tmp_path / "refs.html", sections=["referencias"]).read_text(
        encoding="utf-8"
    )
    assert "<h2 id='refs'>" in included
    assert "doi.org/10.1371" in included

    excluded = generate_report(
        state,
        tmp_path / "no-refs.html",
        sections=["referencias"],
        include_refs=False,
    ).read_text(encoding="utf-8")
    assert "<h2 id='refs'>" not in excluded
    assert "doi.org/10.1371" not in excluded


def test_include_gallery_false_drops_the_gallery_section(tmp_path):
    """``include_gallery=False`` removes the anchor and the side-by-side blocks."""
    import cv2
    import numpy as np

    from polyx.gui.report_html import generate_report

    state = _benign_state(tmp_path)
    ok, buf = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
    assert ok
    pred_path = tmp_path / "pred.png"
    pred_path.write_bytes(buf.tobytes())
    for rows in state.results.values():
        for row in rows:
            row.pred_path = pred_path

    included = generate_report(state, tmp_path / "gallery.html", sections=["galeria"]).read_text(
        encoding="utf-8"
    )
    assert "<h2 id='gallery'>" in included
    assert "Model prediction" in included

    excluded = generate_report(
        state,
        tmp_path / "no-gallery.html",
        sections=["galeria"],
        include_gallery=False,
    ).read_text(encoding="utf-8")
    assert "<h2 id='gallery'>" not in excluded
    assert "Model prediction" not in excluded
