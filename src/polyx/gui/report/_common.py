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

# matplotlib in non-interactive mode
import matplotlib

matplotlib.use("Agg")

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

# Particle card with mask overlay lives in particle.py.
# Particle card with mask overlay lives in particle.py.
from .particle import _record_particle as _record_particle

# Section registry, presets, anchors and document shell live in registry.py;
# re-exported here for the package surface and call sites.
from .registry import (
    _ANCHORS as _ANCHORS,
)
from .registry import (
    IDS_SECTIONS as IDS_SECTIONS,
)
from .registry import (
    NUM as NUM,
)
from .registry import (
    PRESETS as PRESETS,
)
from .registry import (
    SECTIONS as SECTIONS,
)
from .registry import (
    _document_html as _document_html,
)
from .registry import (
    anchor as anchor,
)

# Samples/ordering/sampling/batch aggregation live in samples.py;
# re-exported here to keep the package surface and call sites stable.
from .samples import (
    _MAX_FOLDER_PER_PHOTO as _MAX_FOLDER_PER_PHOTO,
)
from .samples import (
    _ORDER_STATION as _ORDER_STATION,
)
from .samples import (
    _PLATE as _PLATE,
)
from .samples import (
    _RE_PLATE as _RE_PLATE,
)
from .samples import (
    _STATION as _STATION,
)
from .samples import (
    _batch_metrics as _batch_metrics,
)
from .samples import (
    _clue_order as _clue_order,
)
from .samples import (
    _lot_forms as _lot_forms,
)
from .samples import (
    _sample_of as _sample_of,
)
from .samples import (
    choose_records as choose_records,
)

# Three presentation colours of the report (fixed palette: a report is
# printed and archived, so it must not follow the UI theme).
_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


_MAX_GROUPS_BOXPLOT = 12


