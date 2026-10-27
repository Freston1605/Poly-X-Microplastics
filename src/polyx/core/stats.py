"""Statistical helpers for the report (moved out of the GUI)."""

from __future__ import annotations

import math

import numpy as np

from polyx.core.i18n import tr
from polyx.core.quality.metrics import match_image

_TABLE_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}

_NAME_MORPHOTYPE = {"fibra": "Fibra", "fragmento": "Partícula"}


def _chi2_sf(h: float, df: int) -> float:
    """P(X > h) of a chi-square with ``df`` degrees of freedom, without scipy.

    Built from the recurrence of the upper incomplete gamma function
    Q(a+1,x) = Q(a,x) + x^a e^-x / Gamma(a+1), stepping a in increments
    of 0.5 up from the closed cases (erfc for odd df, exp for even df).
    df here is always a small integer (number of groups - 1), so the
    recurrence converges in a few steps.
    """
    if h <= 0 or df < 1:
        return 1.0
    xa = h / 2.0
    if df % 2 == 1:
        q = math.erfc(math.sqrt(xa))
        a0, n = 0.5, (df - 1) // 2
    else:
        q = math.exp(-xa)
        a0, n = 1.0, (df - 2) // 2
    if n <= 0:
        return min(1.0, max(0.0, q))
    term = (xa**a0) * math.exp(-xa) / math.gamma(a0 + 1.0)
    q += term
    for k in range(1, n):
        term *= xa / (a0 + k)
        q += term
    return min(1.0, max(0.0, q))


# Two-tailed Student t table, 95% (alpha=0.05), for the confidence
# interval of the batch mean scale. Same reason as _chi2_sf for not
# using scipy.stats.t. Exact standard table values for df 1-30, plus
# the usual reference points 40/60/120 for large batches.


def _t_critical95(df: int) -> float:
    """Two-tailed critical t (95%) for ``df`` degrees of freedom.

    Beyond gl=30 there is no entry per gl: the smallest tabulated step
    that is >= the real one is used. Since t decreases with the degrees
    of freedom, that step gives a slightly wider interval than the exact
    one, never narrower — conservative rather than optimistic.
    """
    if df <= 0:
        return float("nan")
    if df in _TABLE_T95:
        return _TABLE_T95[df]
    for threshold in (40, 60, 120):
        if df <= threshold:
            return _TABLE_T95[threshold]
    return 1.960


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Rank of each value, averaging the rank across ties."""
    order = np.argsort(a, kind="mergesort")
    sorted_value = a[order]
    ranges_ord = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_value[j + 1] == sorted_value[i]:
            j += 1
        ranges_ord[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranges = np.empty(len(a), dtype=float)
    ranges[order] = ranges_ord
    return ranges


def _average_ic95(values_2: list[float]) -> tuple[float, float, int] | None:
    """(mean, margin of the 95% CI, n), or None when there is nothing to estimate
    from.

    The margin is t-critical * sample stdev / sqrt(n); the CI is reported
    as mean ± margin. With a single value there is no dispersion to
    estimate.
    """
    a = np.asarray(values_2, dtype=float)
    n = len(a)
    if n < 2:
        return None
    average = float(a.mean())
    std = float(a.std(ddof=1))
    margin = _t_critical95(n - 1) * std / math.sqrt(n)
    return average, margin, n


def _kruskal_wallis(groups: list[np.ndarray]) -> tuple[float, float, int] | None:
    """(H, p, degrees of freedom), or None without enough data.

    Includes the standard correction for ties (common here: sizes are
    rounded to one decimal). Without the correction, a batch with many
    repeated values inflates H artificially.
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return None
    all_value_2 = np.concatenate(groups)
    n_total = len(all_value_2)
    if n_total < 3:
        return None
    ranges = _rankdata(all_value_2)
    h = 0.0
    idx = 0
    for g in groups:
        n_i = len(g)
        r_i = ranges[idx : idx + n_i].sum()
        h += (r_i * r_i) / n_i
        idx += n_i
    h = (12.0 / (n_total * (n_total + 1))) * h - 3 * (n_total + 1)
    _, counts = np.unique(all_value_2, return_counts=True)
    draw = np.sum(counts**3 - counts)
    denom = n_total**3 - n_total
    if denom > 0 and draw > 0:
        c = 1.0 - draw / denom
        if c > 0:
            h = h / c
    df = len(groups) - 1
    p = _chi2_sf(h, df)
    return h, p, df


def _text_kruskal(result: tuple[float, float, int] | None, n_groups: int) -> str:
    if result is None:
        return tr(
            "Sin datos suficientes para una prueba estadística (hace falta "
            "más de una observación por grupo)."
        )
    h, p, df = result
    frase = (
        tr("<strong>diferencia significativa</strong>")
        if p < 0.05
        else tr("sin diferencia significativa")
    )
    return tr(
        "Kruskal-Wallis entre los {n} grupos: {frase} (H={h}, gl={gl}, "
        "p={p}). No dice CUÁL grupo difiere de cuál, solo que no todos "
        "comparten la misma distribución de talla."
    ).format(n=n_groups, frase=frase, h=f"{h:.2f}", gl=df, p=f"{p:.3g}")


def _spearman(x, y) -> float | None:
    """Rank correlation between two series. None when it cannot be computed.

    Ranks and not Pearson because here there is no reason to expect a
    LINEAR relation: the question is whether size (or count) rises or
    falls with depth, not at what rate. Also, the particle count has a
    long tail — a few plates with hundreds — that would dominate a
    Pearson correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _p_permutation(x, y, n_perm: int = 20000, seed: int = 0) -> float | None:
    """Two-sided p of the correlation, shuffling one of the two series.

    By permutation and not by the usual t approximation: with 7 or 8
    intervals that approximation does not hold, and here n is the number
    of INTERVALS, not of particles. Shuffling assumes nothing about the
    distribution and answers exactly the question at hand — how often
    chance produces a correlation this big.

    The seed is fixed so the report is reproducible: the same batch has
    to give the same number every time it is generated.
    """
    rho = _spearman(x, y)
    if rho is None:
        return None
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    extremes = 0
    for _ in range(n_perm):
        r = _spearman(x, rng.permutation(y))
        if r is not None and abs(r) >= abs(rho) - 1e-12:
            extremes += 1
    # +1 in numerator and denominator: without it, p could come out 0, which does not exist
    # in a permutation test.
    return (extremes + 1) / (n_perm + 1)


def _text_trend(x, y, que: str) -> str:
    """Sentence with the correlation, its p and the n. Empty when it cannot be computed."""
    rho = _spearman(x, y)
    if rho is None:
        return ""
    p = _p_permutation(x, y)
    direction = tr("crece") if rho > 0 else tr("decrece")
    if p is not None and p < 0.05:
        # Capital initial letter: the sentence goes after the period that closes the station
        # name, and in lowercase it reads like a truncated line.
        verdict = tr("<strong>{que} {sentido} con la profundidad</strong>").format(
            que=que[0].upper() + que[1:], sentido=direction
        )
    else:
        verdict = tr("No se detecta tendencia en {que} con la profundidad").format(que=que)
    # The operator travels with the value: with `p = {p}` and a small p it printed
    # `p = < 0.001`, with two signs in a row.
    #
    # Decimal point and not comma: the rest of the report figures come from
    # Python formats and carry a point; mixing them reads like a typo.
    p_txt = "< 0.001" if p is not None and p < 0.001 else f"= {p:.3f}"
    return tr("{veredicto} (Spearman ρ = {rho}, p {p}, n = {n} tramos).").format(
        veredicto=verdict, rho=f"{rho:+.2f}", p=p_txt, n=len(x)
    )


def _morphotype(value: str | None, lowercase: bool = False) -> str:
    """Presentable name for a morphotype. '—' when it could not be determined."""
    if not value:
        return "—"
    name_2 = tr(_NAME_MORPHOTYPE.get(value, value))
    return name_2.lower() if lowercase else name_2


def _sweep_confidence(results_2, state, iou_thr: float, conf_current: float):
    """Recomputes P/R/F1 at different confidence thresholds, without re-inferring.

    The saved predictions already passed the threshold the run used, so
    raising it is just filtering and re-pairing. Lowering it is NOT
    possible: those detections were never computed. That is why the sweep
    starts at the threshold used and states that limitation.

    Returns {alias: [(conf, precision, recall, f1, n_det), ...]}.
    """
    thresholds = [round(u, 2) for u in np.arange(conf_current, 0.96, 0.05)]
    output_2 = {}
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None:
            continue
        rs = [r for r in results_2.get(mi, []) if r.has_gt]
        if not rs:
            continue
        rows_2 = []
        for u in thresholds:
            tp = fp = fn = mc = nd = 0
            for r in rs:
                preds = [p for p in r.predictions if p.conf >= u]
                nd += len(preds)
                m = match_image(preds, r.gt, iou_thr=iou_thr)
                tp += m.tp
                fp += m.fp
                fn += m.fn
                mc += m.miscls
            # Strict: misclassification penalizes on both sides.
            p = tp / (tp + fp + mc) if (tp + fp + mc) else 0.0
            rec = tp / (tp + fn + mc) if (tp + fn + mc) else 0.0
            f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
            rows_2.append((u, p, rec, f1, nd))
        output_2[slot.alias] = rows_2
    return output_2
