"""Stats helpers."""

from ._common import *
import polyx.gui.report._common as _common

globals().update({k: v for k, v in vars(_common).items() if not k.startswith("__")})


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
