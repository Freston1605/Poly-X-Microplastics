"""HTML report package: the one-way section-builder DAG.

``render -> sections -> figures -> _common`` is the dependency direction.
This module is the package's public surface: it re-exports, from the module
that owns each one, exactly the names the pre-split ``polyx.gui.report_html``
module exposed. The 76-name contract is pinned by
``tests/test_report_surface.py``; ``polyx.gui.report_html`` is the
compatibility shim over this surface.
"""

from polyx.core.report_metrics import (
    _ORDER_CLASSES,
    _classes_views,
    _count_per_class,
    _pr_f1,
)

from polyx.gui import theme as T

# Three presentation colours of the report (fixed palette: a report is
# printed and archived, so it must not follow the UI theme).
_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


_MAX_GROUPS_BOXPLOT = 12

from .figures import (
    _fig_agreement_by_image,
    _fig_boxplot_groups,
    _fig_class_distribution,
    _fig_comparison_metrics,
    _fig_confidence_hist,
    _fig_confusion_matrix,
    _fig_metrics_by_class,
    _fig_profile_depth,
    _fig_size_hist,
    _fig_sizes_by_section,
    _fig_sweep,
    _fig_to_b64,
    _figure_calibration,
)
from .machine import _team, _training_of
from .markup import _WIDTH_GALLERY, _esc, _img_b64, _table_count, _table_width, _uri_gallery
from .particle import _record_particle
from .registry import _ANCHORS, IDS_SECTIONS, NUM, PRESETS, SECTIONS, _document_html, anchor
from .render import (
    _generate_report_impl,
    generate_report,
)
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
from .sections import (
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
    _section_team,
    _section_verdict,
    _size_per_folder,
    section_depth,
)
from .stats import (
    _NAME_MORPHOTYPE,
    _TABLE_T95,
    _average_ic95,
    _chi2_sf,
    _kruskal_wallis,
    _morphotype,
    _p_permutation,
    _rankdata,
    _spearman,
    _sweep_confidence,
    _t_critical95,
    _text_kruskal,
    _text_trend,
)

__all__ = [
    "IDS_SECTIONS",
    "NUM",
    "PRESETS",
    "SECTIONS",
    "_ANCHORS",
    "_MAX_FOLDER_PER_PHOTO",
    "_MAX_GROUPS_BOXPLOT",
    "_NAME_MORPHOTYPE",
    "_ORDER_CLASSES",
    "_ORDER_STATION",
    "_PALETTE",
    "_PLATE",
    "_RE_PLATE",
    "_STATION",
    "_TABLE_T95",
    "_WIDTH_GALLERY",
    "_average_ic95",
    "_batch_metrics",
    "_chi2_sf",
    "_classes_views",
    "_clue_order",
    "_count_per_class",
    "_document_html",
    "_esc",
    "_fig_agreement_by_image",
    "_fig_boxplot_groups",
    "_fig_class_distribution",
    "_fig_comparison_metrics",
    "_fig_confidence_hist",
    "_fig_confusion_matrix",
    "_fig_metrics_by_class",
    "_fig_profile_depth",
    "_fig_size_hist",
    "_fig_sizes_by_section",
    "_fig_sweep",
    "_fig_to_b64",
    "_figure_calibration",
    "_generate_report_impl",
    "_img_b64",
    "_kruskal_wallis",
    "_lot_forms",
    "_morphotype",
    "_p_permutation",
    "_pr_f1",
    "_rankdata",
    "_record_particle",
    "_sample_of",
    "_section_calibration",
    "_section_comparison",
    "_section_count",
    "_section_errors",
    "_section_gallery",
    "_section_methods",
    "_section_models",
    "_section_records",
    "_section_references",
    "_section_results",
    "_section_shape",
    "_section_summary",
    "_section_team",
    "_section_verdict",
    "_size_per_folder",
    "_spearman",
    "_sweep_confidence",
    "_t_critical95",
    "_table_count",
    "_table_width",
    "_team",
    "_text_kruskal",
    "_text_trend",
    "_training_of",
    "_uri_gallery",
    "anchor",
    "choose_records",
    "generate_report",
    "section_depth",
]
