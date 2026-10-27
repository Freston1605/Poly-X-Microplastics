"""Statistics helpers re-exported from ``polyx.core.stats``.

The subset of ``polyx.core.stats`` that the report's public surface exposes
(see ``polyx.gui.report.__all__``): the package re-exports them from here so
the report has a single entry point for its statistics helpers.
"""

from polyx.core.stats import (
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
    "_NAME_MORPHOTYPE",
    "_TABLE_T95",
    "_average_ic95",
    "_chi2_sf",
    "_kruskal_wallis",
    "_morphotype",
    "_p_permutation",
    "_rankdata",
    "_spearman",
    "_sweep_confidence",
    "_t_critical95",
    "_text_kruskal",
    "_text_trend",
]
