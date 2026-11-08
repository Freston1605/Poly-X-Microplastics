"""The named inputs every report section builder receives (#85).

``ReportContext`` is frozen so a builder cannot rebind a field while it
renders, and the assembler builds one instance per report. Freezing protects
the bindings, not the contents: the lists and dicts the context points at are
the batch's own containers. It lives in its own module because both sides of
the section seam need it: the assembler
(:mod:`polyx.gui.report.render`) constructs it and the builders
(:mod:`polyx.gui.report.sections`) annotate their parameter with it. Keeping it
here avoids a runtime import cycle between those two modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polyx.core.detection_run import RunResult

    from .batch import BatchMetrics, LotForms
    from .sections import ShapeBodies


@dataclass(frozen=True)
class ReportContext:
    """Everything a section builder may read, computed once by the assembler.

    ``batch`` and ``lot`` are the named precomputes; ``shape_bodies`` is the
    shape pass's two section bodies, produced once and shared by the ``forma``
    and ``talla_carpetas`` builders. ``max_gallery`` and ``max_records`` are the
    render caps, so a builder never has to know which keyword the caller used.
    """

    run: RunResult
    batch: BatchMetrics
    lot: LotForms
    shape_bodies: ShapeBodies
    max_gallery: int
    max_records: int
