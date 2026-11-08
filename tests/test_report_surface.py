"""The report package exposes the catalog and the entry point, nothing else.

``polyx.gui.report`` is internal: its public surface is exactly
``generate_report``, ``SECTIONS``, ``IDS_SECTIONS``, ``PRESETS`` and
``anchor``. The compatibility shim ``polyx.gui.report_html`` exposes the same
five names, so the GUI page, the CLI and the audit tool keep importing from
the shim while the section builders, figures and helpers stay free to change.

The list is written by hand here on purpose: removing a name from ``__all__``
or from the shim must break this test, not a consumer in production. The
builder-coverage invariant (one builder per catalog id) is pinned here too,
because it is the contract that replaces the retired hand-written name list.
"""

from __future__ import annotations

from pathlib import Path

# The five names ARE the contract: do not edit this tuple to make the test
# pass; first find the consumer that still needs the name.
EXPECTED: tuple[str, ...] = (
    "IDS_SECTIONS",
    "PRESETS",
    "SECTIONS",
    "anchor",
    "generate_report",
)


def test_the_contract_list_is_intact():
    """Five names, no duplicates and sorted: the list does not drift on its own."""
    assert len(EXPECTED) == 5
    assert len(set(EXPECTED)) == len(EXPECTED)
    assert list(EXPECTED) == sorted(EXPECTED)


def test_the_package_exports_the_five_names():
    """``polyx.gui.report`` exports the catalog plus the entry point, name by name."""
    import polyx.gui.report as package

    assert sorted(package.__all__) == list(EXPECTED)
    assert len(package.__all__) == len(set(package.__all__))
    missing = [name for name in EXPECTED if not hasattr(package, name)]
    assert not missing, f"the package does not export: {missing}"


def test_the_package_exports_no_private_names():
    """Internal wiring is not API: no underscore name reaches the surface."""
    import polyx.gui.report as package

    assert not [name for name in package.__all__ if name.startswith("_")]


def test_the_shim_exports_the_same_five_names():
    """``polyx.gui.report_html`` still exposes the surface consumers import."""
    import polyx.gui.report_html as shim

    assert sorted(shim.__all__) == list(EXPECTED)
    assert len(shim.__all__) == len(set(shim.__all__))
    missing = [name for name in EXPECTED if not hasattr(shim, name)]
    assert not missing, f"the shim does not export: {missing}"


def test_the_shim_exports_no_private_names():
    """The shim mirrors the package, so it carries no underscore names either."""
    import polyx.gui.report_html as shim

    assert not [name for name in shim.__all__ if name.startswith("_")]


def test_report_css_stays_out_of_the_surface():
    """``REPORT_CSS`` was replaced by the lazy ``_report_css()``; it stays out."""
    import polyx.gui.report as package
    import polyx.gui.report_html as shim

    assert "REPORT_CSS" not in EXPECTED
    assert "REPORT_CSS" not in package.__all__
    assert "REPORT_CSS" not in shim.__all__
    assert not hasattr(package, "REPORT_CSS")
    assert not hasattr(shim, "REPORT_CSS")


def test_the_shim_points_at_the_package_file():
    """``report_html.__file__`` resolves to the package, not to a second module."""
    import polyx.gui.report_html as shim

    assert Path(shim.__file__).parent.name == "report"


def test_every_catalog_section_has_exactly_one_builder():
    """Builder coverage: adding a catalog id without a builder fails here."""
    import polyx.gui.report.registry as registry
    import polyx.gui.report.render as render

    assert set(render.SECTION_BUILDERS) == set(registry.IDS_SECTIONS)
    assert len(render.SECTION_BUILDERS) == len(registry.IDS_SECTIONS)
