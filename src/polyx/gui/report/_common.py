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
from .machine import _team, _training_of

# HTML primitives (escape, stylesheet, base64 embeds, tables) live in
# markup.py; re-exported here because that is where the section builders
# and the package surface import them from.
from .markup import _WIDTH_GALLERY, _esc, _img_b64, _report_css, _table_count, _table_width, _uri_gallery

# Particle card with mask overlay lives in particle.py.
from .particle import _record_particle

# Section registry, presets, anchors and document shell live in registry.py;
# re-exported here for the package surface and call sites.
from .registry import _ANCHORS, IDS_SECTIONS, NUM, PRESETS, SECTIONS, _document_html, anchor

# Samples/ordering/sampling/batch aggregation live in samples.py;
# re-exported here to keep the package surface and call sites stable.
from .samples import (
    _MAX_FOLDER_PER_PHOTO,
    _ORDER_STATION,
    _PLATE,
    _RE_PLATE,
    _STATION,
    _batch_metrics,
    _clue_order,
    _lot_forms,
    _sample_of,
    choose_records,
)

# Three presentation colours of the report (fixed palette: a report is
# printed and archived, so it must not follow the UI theme).
_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


_MAX_GROUPS_BOXPLOT = 12
