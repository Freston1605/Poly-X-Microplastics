"""Import-smoke test for GUI detector pages.

Guards against regressions where a page imports a name that no longer exists
in its source module (see PR review on the core refactor). The main suite
does not otherwise import these modules, so a broken import would otherwise
pass silently.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

PAGES = [
    "polyx.gui.detector.pages.gt_manual",
    "polyx.gui.detector.pages.images",
    "polyx.gui.detector.pages.models",
    "polyx.gui.detector.pages.parameters",
    "polyx.gui.detector.pages.execute",
    "polyx.gui.detector.pages.results",
    "polyx.gui.detector.pages.errors",
    "polyx.gui.detector.pages.compare",
    "polyx.gui.detector.pages.report",
    "polyx.gui.detector.recargar_gt",
    "polyx.gui.detector.review_dialog",
]


@pytest.mark.parametrize("module_name", PAGES)
def test_detector_page_imports(module_name: str) -> None:
    import importlib

    importlib.import_module(module_name)