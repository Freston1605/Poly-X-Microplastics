"""HTML report package: the one-way section-builder DAG.

``render -> sections -> figures -> (batch/machine/markup/samples/particle/registry)``
is the dependency direction. Each module has a single responsibility:
batch (named batch and lot-shape precomputes), machine (provenance), markup
(HTML primitives and CSS), samples (sampling and ordering), particle
(particle cards), registry (section catalog, anchors, document assembly).

This module is the package's public surface: the entry point and the section
catalog, nothing else. ``generate_report`` is the seam callers render through;
``SECTIONS`` / ``IDS_SECTIONS`` / ``PRESETS`` / ``anchor`` are the catalog the
GUI page and the section tests read. Every other name is implementation and
lives in its owning module. ``polyx.gui.report_html`` is a compatibility shim
over this surface; the contract is pinned by ``tests/test_report_surface.py``.
"""

from .registry import IDS_SECTIONS, PRESETS, SECTIONS, anchor
from .render import generate_report

__all__ = [
    "IDS_SECTIONS",
    "PRESETS",
    "SECTIONS",
    "anchor",
    "generate_report",
]
