"""Figures."""

from ._common import *
from .stats import *
import polyx.gui.report._common as _common

globals().update({k: v for k, v in vars(_common).items() if not k.startswith("__")})


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ────────────────────────────────────────────────────────────────────
# Count by polymer and by sample
# ────────────────────────────────────────────────────────────────────
# Presentation order of the polymers. Any class that appears in the
# data and is not listed here is appended at the end, so a dataset with other classes
# does not silently lose columns.


def _fig_sweep(sweep: dict[str, list]) -> str:
    """F1 curve against the confidence threshold, one line per model."""
    if not sweep:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for alias, rows_2 in sweep.items():
        ax.plot(
            [f[0] for f in rows_2],
            [f[3] for f in rows_2],
            marker="o",
            markersize=3,
            label=alias,
            linewidth=1.6,
        )
        best = max(rows_2, key=lambda f: f[3])
        ax.plot(
            [best[0]],
            [best[3]],
            marker="*",
            markersize=13,
            zorder=5,
            color=ax.lines[-1].get_color(),
        )
    ax.set_xlabel(tr("Umbral de confianza"))
    ax.set_ylabel(tr("F1 (con clase)"))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_comparison_metrics(data_2: list[tuple]) -> str:
    """Grouped bars of P/R/F1 per model, under the two criteria.

    ``datos``: ``[(alias, p_loc, r_loc, f1_loc, p_cls, r_cls, f1_cls), ...]``.

    Both panels are drawn together, not only the permissive one: the gap
    between both is the confusion between polymers, and seeing them side
    by side keeps the good number from being read as the full performance.
    """
    if not data_2:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    labels_2 = [tr("Precisión"), "Recall", "F1"]
    x = np.arange(len(labels_2))
    width_2 = 0.8 / max(1, len(data_2))
    for panel, (ax, offset, title) in enumerate(
        ((axes[0], 1, tr("Localización")), (axes[1], 4, tr("Con clase")))
    ):
        for i, row in enumerate(data_2):
            vals = row[offset : offset + 3]
            pos = x - 0.4 + width_2 * (i + 0.5)
            bars = ax.bar(pos, vals, width_2, label=row[0], color=_PALETTE[i % len(_PALETTE)])
            if panel == 1 or len(data_2) <= 2:
                ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_2, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(tr("Valor"))
    # The legend goes below the figure: inside the axes it overlapped the
    # bars, which with these values almost reach the ceiling in both halves.
    handlers, names_2 = axes[0].get_legend_handles_labels()
    fig.legend(
        handlers,
        names_2,
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=len(data_2),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_agreement_by_image(alias_a: str, alias_b: str, pares: list[tuple]) -> str:
    """Detections per image of one model against the other, with the diagonal.

    A batch total is dominated by a few dense plates, so two models can
    tie on the sum and yet disagree photo by photo. Here every point is
    an image: on the diagonal both count the same, and the distance to it
    says how far apart they are and on which plates.

    Symmetric-log scale because most plates have 0-10 particles and a few
    go over 500; on a linear scale the bulk of the points clumps at the
    origin.
    """
    if not pares:
        return ""
    a = [p[0] for p in pares]
    b = [p[1] for p in pares]
    max_size = max(max(a), max(b), 1)  # noqa: PLW3301
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, max_size], [0, max_size], "--", color="#999", linewidth=1, label="acuerdo exacto")
    ax.scatter(a, b, s=26, alpha=0.75, color=T.DOC.ACCENT, edgecolor="white", linewidth=0.5)
    # The plates that differ the most are labelled: those are the ones to go and inspect.
    for na, nb, name_2 in sorted(pares, key=lambda p: -abs(p[0] - p[1]))[:3]:
        if abs(na - nb) > 0:
            ax.annotate(
                name_2,
                (na, nb),
                fontsize=7,
                color=T.DOC.INK2,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(-1, max_size * 1.6)
    ax.set_ylim(-1, max_size * 1.6)
    ax.set_xlabel(f"Detecciones · {alias_a}")
    ax.set_ylabel(f"Detecciones · {alias_b}")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_metrics_by_class(by_model: dict[str, dict[str, tuple]]) -> str:
    """F1 per class and per model: where each one wins.

    The global F1 averages polymers of very different performance — PET
    separates well, PP and LDPE share a tone and get confused — so a
    model can win on the aggregate and lose on the class that matters.
    """
    if not by_model:
        return ""
    classes = []
    for m in by_model.values():
        for c in m:
            if c not in classes:
                classes.append(c)
    if not classes:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(classes))
    width_2 = 0.8 / max(1, len(by_model))
    for i, (alias, metrics) in enumerate(by_model.items()):
        vals = [metrics.get(c, (0, 0, 0))[2] for c in classes]
        bars = ax.bar(
            x - 0.4 + width_2 * (i + 0.5),
            vals,
            width_2,
            label=alias,
            color=_PALETTE[i % len(_PALETTE)],
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(tr("F1 por clase"))
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_class_distribution(per_class_counts: dict[str, int]) -> str:
    if not per_class_counts:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = list(per_class_counts.keys())
    vals = list(per_class_counts.values())
    colors = [T.CLASS_COLOR_HEX.get(n, "#888") for n in names]
    ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(tr("Detecciones"))
    ax.set_title(tr("Distribución por clase"))
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_confidence_hist(confs: list[float]) -> str:
    if not confs:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(confs, bins=20, color=T.DOC.ACCENT, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(tr("Confianza"))
    ax.set_ylabel(tr("Frecuencia"))
    ax.set_title(tr("Histograma de confianza"))
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_size_hist(sizes_um: list[float]) -> str:
    if not sizes_um:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(sizes_um, bins=25, color=T.DOC.WARN, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(tr("Diámetro equivalente (μm)"))
    ax.set_ylabel(tr("Frecuencia"))
    ax.set_title(tr("Distribución de tamaños"))
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_sizes_by_section(lengths_per_class: dict[str, list[float]]) -> str:
    """Stacked size histogram, in bands, with one bar per class.

    Stacked and not overlaid: the question it answers is how many
    particles sit in each size band and which polymer they belong to;
    with overlaid bars the front one hides the back one.
    """
    all_value_2 = [v for vals in lengths_per_class.values() for v in vals]
    if not all_value_2:
        return ""
    a = np.array(all_value_2, dtype=float)
    # Stretches on a logarithmic scale: sizes range from tens to
    # thousands of micrometres, and on a linear scale everything piles up in the first
    # bar and the tail becomes illegible.
    lo, hi = max(a.min(), 1.0), a.max()
    edges = np.logspace(np.log10(lo), np.log10(hi * 1.02), 13)

    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=140)
    bottom_2 = np.zeros(len(edges) - 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges) * 0.86
    for cls in _ORDER_CLASSES + [c for c in lengths_per_class if c not in _ORDER_CLASSES]:
        vals = lengths_per_class.get(cls)
        if not vals:
            continue
        account, _ = np.histogram(np.array(vals, dtype=float), bins=edges)
        ax.bar(
            centers,
            account,
            width=widths,
            bottom=bottom_2,
            label=cls,
            color=T.CLASS_COLOR_HEX.get(cls, "#8b949e"),
            edgecolor="white",
            linewidth=0.6,
        )
        bottom_2 += account
    ax.set_xscale("log")
    ax.set_xlabel(tr("talla (µm, dimensión mayor siguiendo la curva)"))
    ax.set_ylabel(tr("partículas"))
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Size across groups (folder, or photo within a folder) ──
#
# The paper pipeline handled `by station` by parsing a
# filename pattern (N.T from one specific study). That does not work for an arbitrary
# batch that someone loads in the Detector. The only thing the app knows about
# ANY batch is which folder each photo came from, so the contrast is
# generalized to that: by folder, and by photo within each folder.
#
# The test (Kruskal-Wallis) is implemented here by hand, without scipy: scipy is not
# a declared project dependency -- it arrives indirectly with ultralytics, and
# relying on it would break the installation the day ultralytics stops
# shipping it (the same reason morfologia.py avoids scipy). The H statistic
# and the p value were verified against scipy.stats.kruskal on a hundred
# random cases, with and without ties, and they match to machine precision.


def _fig_boxplot_groups(
    groups: dict[str, list[float]], title: str, xlabel: str, max_groups: int = _MAX_GROUPS_BOXPLOT
) -> str:
    """Boxplot of size by group (folder, or photo inside a folder).

    Truncated to the ``max_grupos`` with the most particles: a boxplot
    with more than a dozen boxes becomes illegible before it contributes
    anything, and the groups with few particles say the least anyway.
    """
    items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    truncated = len(items) > max_groups
    items = items[:max_groups]
    items.sort(key=lambda kv: kv[0])

    data_2 = [np.asarray(v, dtype=float) for _, v in items]
    labels_2 = [f"{k}\n(n={len(v)})" for k, v in items]

    fig, ax = plt.subplots(figsize=(max(5.5, 1.15 * len(items) + 2), 4.2), dpi=140)
    bp = ax.boxplot(
        data_2,
        tick_labels=labels_2,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.35, markeredgewidth=0),  # noqa: C408
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(T.DOC.ACCENT)
        patch.set_alpha(0.30)
        patch.set_edgecolor(T.DOC.ACCENT)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)
    ax.set_yscale("log")
    ax.set_ylabel(tr("talla (µm, escala log)"))
    ax.set_xlabel(xlabel)
    title_complete = title + (
        tr(" (las {n} con más partículas)").format(n=max_groups) if truncated else ""
    )
    ax.set_title(title_complete, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="both", alpha=0.2)
    rota = len(items) > 5
    plt.setp(ax.get_xticklabels(), rotation=30 if rota else 0, ha="right" if rota else "center")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_profile_depth(profiles: dict) -> str:
    """Core sample profile: depth on the vertical axis, pointing down.

    It is the convention of any sediment profile, and it is not cosmetic:
    with depth on the X axis one has to turn the head to read the core.
    """
    if not profiles:
        return ""
    # One line per station, with the report colours.
    colors_2 = [T.DOC.ACCENT, T.DOC.OK, T.DOC.VIO, T.DOC.WARN, T.DOC.ERR]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.6), dpi=140, sharey=True)
    for i, (station, rows_2) in enumerate(sorted(profiles.items())):
        ranges_2 = [f[0] for f in rows_2]
        color = colors_2[i % len(colors_2)]
        ax1.plot(
            [f[1] for f in rows_2],
            ranges_2,
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=station,
            color=color,
        )
        medians = [f[2] for f in rows_2]
        ax2.plot(
            medians, ranges_2, marker="o", markersize=4, linewidth=1.4, label=station, color=color
        )
        # Interquartile range as a band: the median alone hides whether the
        # size distribution widens or narrows with depth.
        ax2.fill_betweenx(
            ranges_2,
            [f[3] for f in rows_2],
            [f[4] for f in rows_2],
            alpha=0.15,
            color=color,
            linewidth=0,
        )
    ax1.set_xlabel(tr("partículas por tramo"))
    ax2.set_xlabel(tr("talla mediana (µm)  ·  banda = rango intercuartílico"))
    ax1.set_ylabel(tr("tramo (mayor = más profundo)"))
    ax1.invert_yaxis()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if len(profiles) > 1:
        ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── How each morphotype is named for the user ──
# The code distinguishes `fibra` and `fragmento` (morfologia.FIBRA / FRAGMENTO) and
# that does NOT change: it is the classification used to measure and to decide whether the length
# comes from Feret or from the geodesic path. What changes is the word that is printed.
#
# It lives in a function and is not repeated in every table because it already happened once: it was
# renamed only in the cards and the report ended up calling `particula` and
# `fragmento` the same thing on different pages.
#
# NOTE FOR THE PAPER: in the microplastics literature the standard term is
# `fragment` (fibre / fragment / film / pellet). For the English manuscript
# it is better to go back to `fragment`; here `particula` is used by the author's decision,
# because in this material almost everything is a fragment and repeating the word in every
# card added nothing.
# Translation happens at the CALL SITE, not here: a tr() in the module body is resolved at
# import time and would keep the language of that moment.


def _figure_calibration(state) -> str:
    """Photo of a plate with the fitted circle drawn on top.

    The report used to declare the scale —
    `«32.86 µm/px, placa de 100 mm»` — but never SHOW it. The whole size chain of the study hangs
    off that number, and until now it had to be taken on faith: if the
    fit had grabbed the wrong edge (the plate's shadow, the ring's
    reflection, the inner filter's edge) nothing in the document would
    betray it.

    With the circumference drawn over the photo, one glance shows
    whether the fit landed where it should. It is the same criterion
    already applied to each particle in the ``fichas``: a measurement
    that cannot be seen being measured cannot be verified.

    The MIDDLE-scale photo is chosen, not the first: the first one can be
    exactly the batch outlier, and then the figure would illustrate the
    rare case.

    Returns "" when there is no measured plate to draw on.
    """
    from polyx.core.measure.calibration import DIAMETER_PLATE_MM, ORIGIN_PLATE

    cals = getattr(state, "calibrations", None) or {}
    measurements = [
        (name_2, c)
        for name_2, c in cals.items()
        if getattr(c, "valid", False)
        and getattr(c, "origin", "") == ORIGIN_PLATE
        and getattr(c, "radius_px", 0) > 0
    ]
    if not measurements:
        return ""

    # The median-scale one represents the batch; the first one by name does not.
    measurements.sort(key=lambda t: t[1].um_por_px)
    name_2, cal = measurements[len(measurements) // 2]

    # It is necessary to go back to the original photo: the calibration stores cx/cy/radius but
    # not the pixels. It is searched among the analysed images by name.
    path_2 = None
    for r in [x for rs in state.results.values() for x in rs]:
        if Path(r.image_path).name == name_2:
            path_2 = Path(r.image_path)
            break
    if path_2 is None or not path_2.exists():
        return ""

    try:
        import cv2

        from polyx.core.measure.calibration import read_image

        bgr = read_image(path_2)
        if bgr is None:
            return ""

        vis = bgr.copy()
        cx, cy, r_px = float(cal.cx), float(cal.cy), float(cal.radius_px)
        # Line widths proportional to the photo size: a 2 px stroke is
        # invisible in a 4096 image, and a 20 px one covers the edge that
        # is precisely what we want to show.
        thickness = max(2, int(round(min(vis.shape[:2]) / 500)))  # noqa: RUF046

        # Fitted circumference + centre.
        cv2.circle(vis, (int(cx), int(cy)), int(r_px), (0, 255, 0), thickness)
        cv2.drawMarker(
            vis, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, thickness * 14, thickness
        )
        # The diameter, which is the known magnitude (100 mm nominal).
        cv2.line(
            vis, (int(cx - r_px), int(cy)), (int(cx + r_px), int(cy)), (0, 210, 255), thickness
        )

        scale = 900.0 / max(vis.shape[:2])
        if scale < 1:
            vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return ""
        img64 = base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:  # noqa: BLE001
        # An illustrative figure cannot bring down the whole report.
        return ""

    diam_px = 2.0 * r_px
    diam_mm = getattr(cal, "diameter_mm", 0) or DIAMETER_PLATE_MM
    return f"""
<h3>{NUM}.1 {tr("Cómo se obtiene la escala, sobre una placa real")}</h3>
<p>{tr("En verde, la circunferencia que el ajuste encontró para el borde externo del anillo; en amarillo, su diámetro. Ese diámetro es la magnitud conocida")}
—<strong>{diam_mm:g} mm</strong>—{tr(" y de ahí sale todo lo demás.")}</p>
<img src='data:image/jpeg;base64,{img64}' style='max-width:100%'>
<p style='font-family:monospace;font-size:10pt'>
{diam_mm:g} mm ÷ {diam_px:.1f} px = <strong>{cal.um_por_px:.4f} µm/px</strong>
&nbsp;&nbsp;<span style='font-family:inherit;color:#656d76'>({_esc(name_2)},
{tr("{n} puntos de borde").format(n=cal.n_points)})</span></p>
<p>{tr("Conviene mirar si el círculo verde sigue el borde del anillo y no otra cosa —la sombra de la placa, un reflejo, o el borde del filtro de dentro—. Si cayera mal, <strong>todas</strong> las tallas de esa foto saldrían escaladas por el mismo factor equivocado, y ninguna otra cifra del informe lo delataría.")}</p>"""


def _fig_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> str:
    labels = list(class_names) + ["background"]  # noqa: RUF005
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(tr("Predicción"))
    ax.set_ylabel("Ground Truth")
    ax.set_title(tr("Matriz de confusión"))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_b64(fig)
