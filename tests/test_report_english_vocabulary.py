"""Reports speak English ids, presets and anchors (spec #182, ticket #184).

The report surface is a machine contract: the section id is the HTML anchor,
and the preset key is the name a caller passes to choose a combination. Both
must read the same on every machine, so they are English literals, not
translated prose. Tests observe the emitted HTML through ``generate_report``
and the catalog through its public ``SECTIONS`` / ``PRESETS`` names; no private
helper is called.
"""

from __future__ import annotations

from pathlib import Path

# English ids of the sections the benign fixture actually fills. These are
# literals from the spec's replacement list, not values read from the code.
ENGLISH_IDS = (
    "summary",
    "methods",
    "shape",
    "cards",
    "results",
    "models",
    "comparison",
    "counts",
    "references",
)

# Ids retired by the hard cut. None may appear as a section id, a preset key
# or an HTML anchor in a newly generated report.
SPANISH_IDS = (
    "resumen",
    "metodos",
    "calibracion",
    "forma",
    "talla_carpetas",
    "profundidad",
    "fichas",
    "resultados",
    "modelos",
    "errores",
    "comparacion",
    "galeria",
    "conteo",
    "referencias",
    "completo",
    "metodologico",
)

# Legacy anchor aliases that were emitted through ``anchor``. New reports use
# the section id itself, so these spellings must be gone. The four English
# ones (methods/models/results/gallery) are now the ids themselves and are
# checked through the sections that produce them, not as aliases.
LEGACY_ONLY_ANCHORS = ("abstract", "calib", "compare", "refs")


def _benign_state(tmp_path: Path):
    from tests.report_fixtures import estado_benigno as benign_state

    return benign_state(directorio=tmp_path / "photos")


def test_generated_report_uses_english_ids_as_anchors(tmp_path):
    """Every emitted section anchor is its English id, in TOC and heading."""
    from polyx.gui.report_html import generate_report

    html = generate_report(_benign_state(tmp_path), tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    for sid in ENGLISH_IDS:
        assert f"<h2 id='{sid}'>" in html, sid
        assert f"<a href='#{sid}'>" in html, sid


def test_generated_report_emits_no_spanish_or_legacy_anchor(tmp_path):
    """The hard cut removes Spanish ids and the legacy alias table from HTML."""
    from polyx.gui.report_html import generate_report

    html = generate_report(_benign_state(tmp_path), tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    for token in (*SPANISH_IDS, *LEGACY_ONLY_ANCHORS):
        assert f"id='{token}'" not in html, token
        assert f"href='#{token}'" not in html, token


def test_spanish_section_ids_are_not_accepted(tmp_path):
    """A Spanish id selects no section: it is not part of the catalog."""
    from polyx.gui.report_html import IDS_SECTIONS, generate_report

    for sid in SPANISH_IDS:
        assert sid not in IDS_SECTIONS

    html = generate_report(
        _benign_state(tmp_path), tmp_path / "empty.html", sections=["resumen"]
    ).read_text(encoding="utf-8")
    assert "<h2" not in html


def test_preset_keys_are_english():
    """Only the three English preset keys exist; their values are valid ids."""
    from polyx.gui.report_html import IDS_SECTIONS, PRESETS

    assert set(PRESETS) == {"complete", "summary", "methodological"}
    for key, sections in PRESETS.items():
        assert set(sections) <= set(IDS_SECTIONS), key


def test_schema_ids_are_english_in_the_catalog():
    """The section catalog itself carries English ids, not Spanish ones."""
    from polyx.gui.report_html import IDS_SECTIONS

    assert "summary" in IDS_SECTIONS
    assert "resumen" not in IDS_SECTIONS


def test_figure_folder_names_are_english():
    """Labelled-photo folders are English machine names on every machine."""
    from polyx.gui.detector.pages.report import ReportPage

    assert ReportPage.FOLDER_MANUAL == "manual_count"
    assert ReportPage.FOLDER_MODEL == "model_detection"
    assert ReportPage.FOLDER_BOTH == "both_overlay"


def test_figure_folder_names_are_not_spanish():
    """The Spanish folder spellings are gone from the page contract."""
    from polyx.gui.detector.pages.report import ReportPage

    values = {ReportPage.FOLDER_MANUAL, ReportPage.FOLDER_MODEL, ReportPage.FOLDER_BOTH}
    assert not values & {"conteo_manual", "deteccion_modelo", "ambas_superpuestas"}
