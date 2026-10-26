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

import base64
import html
import io
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

# matplotlib in non-interactive mode
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polyx.core.i18n import language, tr
from polyx.core.quality.assignment import NO_ASSIGNABLE, class_id_reported, resume
from polyx.core.quality.metrics import aggregate_per_class, confusion_matrix

from . import theme as T

# ────────────────────────────────────────────────────────────────────
# Report styles (identical to the manual)
# ────────────────────────────────────────────────────────────────────
REPORT_CSS = """
:root{--ink:#1f2328;--ink2:#424a53;--ink3:#656d76;--muted:#8c959f;
--rule:#d0d7de;--rule_soft:#eaeef2;--bg:#ffffff;--bg_soft:#f6f8fa;
--accent:#0969da;--accent_d:#0550ae;
--ok:#1f6b5e;--warn:#9a6700;--err:#cf222e;--vio:#6639ba;}
*{box-sizing:border-box;}
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:var(--ink);
background:var(--bg);margin:0;padding:0;line-height:1.55;}
.container{max-width:1100px;margin:0 auto;padding:40px 48px 80px;}
header.cover{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:28px;}
.kicker{font-size:10pt;letter-spacing:.14em;text-transform:uppercase;
color:var(--accent_d);font-weight:600;margin-bottom:8px;}
h1{font-size:28pt;margin:4px 0 8px;line-height:1.18;}
h2{font-size:18pt;margin-top:38px;padding-bottom:8px;border-bottom:1px solid var(--rule);}
h3{font-size:13pt;margin-top:22px;color:var(--ink2);}
h4{font-size:11pt;margin-top:14px;color:var(--ink2);}
.meta{font-size:10pt;color:var(--ink3);}
.toc{background:var(--bg_soft);border-left:3px solid var(--accent);
padding:14px 20px;margin:18px 0 28px;}
.toc h3{margin:0 0 8px;color:var(--ink);}
.toc ol{margin:0;padding-left:22px;font-size:10.5pt;}
.toc a{color:var(--ink2);text-decoration:none;}
.toc a:hover{color:var(--accent);}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0;}
.kpi{background:var(--bg);border:1px solid var(--rule);border-radius:8px;
padding:14px 16px;}
.kpi .v{font-size:22pt;font-weight:700;color:var(--ink);margin:4px 0 6px;}
.kpi .l{font-size:9pt;font-weight:600;color:var(--ink3);letter-spacing:1.4px;
text-transform:uppercase;}
.kpi .b{height:3px;background:var(--accent);border-radius:2px;}
table.data{width:100%;border-collapse:collapse;font-size:10pt;margin:12px 0;}
table.data th{background:var(--bg_soft);text-align:left;padding:8px 10px;
border-bottom:1px solid var(--rule);font-weight:600;}
table.data td{padding:7px 10px;border-bottom:1px solid var(--rule_soft);}
table.data tr:last-child td{border-bottom:none;}
table.data td.r{text-align:right;font-variant-numeric:tabular-nums;}
.fig{margin:18px 0;}
.fig img{width:100%;border:1px solid var(--rule);border-radius:6px;display:block;}
.caption{font-size:9pt;color:var(--ink3);margin-top:6px;text-align:center;}
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0;}
.gallery .item{border:1px solid var(--rule);border-radius:6px;overflow:hidden;background:var(--bg_soft);}
.gallery .item img{width:100%;display:block;}
.gallery .item .cap{font-size:8.5pt;color:var(--ink3);padding:5px 8px;}
/* Comparación lado a lado: Predicción vs Ground Truth */
.compare{border:1px solid var(--rule);border-radius:8px;overflow:hidden;
margin:18px 0;background:var(--bg_soft);}
.compare-pair{display:grid;grid-template-columns:1fr 1fr;gap:0;}
.compare-pair figure{margin:0;border-right:1px solid var(--rule);background:var(--bg);}
.compare-pair figure:last-child{border-right:none;}
.compare-pair img{width:100%;display:block;}
.compare-pair figcaption{font-size:9pt;color:var(--ink2);font-weight:600;
padding:7px 10px;text-align:center;border-top:1px solid var(--rule_soft);}
.compare-pair figcaption .sub{display:block;font-weight:400;color:var(--ink3);font-size:8.5pt;margin-top:2px;}
.compare-meta{font-size:9pt;color:var(--ink2);padding:9px 12px;
border-top:1px solid var(--rule);font-variant-numeric:tabular-nums;}
.compare-meta .tag{display:inline-block;margin-right:10px;}
/* Conteo por polimero de una sola foto, bajo el par de imagenes. */
table.conteo{width:100%;border-collapse:collapse;font-size:9.5pt;
border-top:1px solid var(--rule);background:var(--bg);}
table.conteo th{background:var(--bg_soft);text-align:right;padding:6px 12px;
font-weight:600;color:var(--ink2);border-bottom:1px solid var(--rule_soft);}
table.conteo th:first-child{text-align:left;}
table.conteo td{padding:5px 12px;text-align:right;
font-variant-numeric:tabular-nums;border-bottom:1px solid var(--rule_soft);}
table.conteo td:first-child{text-align:left;font-weight:600;}
table.conteo tr.tot td{border-bottom:none;background:var(--bg_soft);font-weight:700;}
table.conteo td.dif-pos{color:var(--err);}
table.conteo td.dif-neg{color:var(--accent_d);}
.nogt{display:flex;align-items:center;justify-content:center;min-height:180px;
color:var(--muted);font-size:10pt;text-align:center;padding:16px;
background:repeating-linear-gradient(45deg,#fbfcfd,#fbfcfd 12px,#f0f2f4 12px,#f0f2f4 24px);}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:8.5pt;
font-weight:600;letter-spacing:.04em;}
.b-blue{background:#ddf4ff;color:var(--accent_d);}
.b-green{background:#d8f5ea;color:var(--ok);}
.b-amber{background:#fff8c5;color:var(--warn);}
.b-red{background:#ffebe9;color:var(--err);}
.b-violet{background:#ece0fd;color:var(--vio);}
footer{border-top:1px solid var(--rule);margin-top:60px;padding-top:18px;
font-size:9.5pt;color:var(--ink3);}
@media print{.container{max-width:none;padding:24px;}
h2{page-break-after:avoid;}.fig,.gallery .item{page-break-inside:avoid;}
/* Que la foto y su tabla no queden en paginas distintas del PDF. */
.compare{page-break-inside:avoid;}}
"""


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
_ORDER_CLASSES = ["PET", "PP", "LDPE"]

# The plates from the Loa river are named `<tramo>.<testigo>` with an `x`/`xx` suffix for
# the second and third plate of the same stretch. Crops add `__f<row>c<col>`.
_RE_PLATE = re.compile(r"^(\d+)\.(\d+)(x*)$")
_STATION = {"1": "Chiu Chiu", "2": "Taira", "3": "Desembocadura"}
_ORDER_STATION = ["Chiu Chiu", "Taira", "Desembocadura"]
_PLATE = {"": "a", "x": "b", "xx": "c"}


def _count_per_class(detections) -> Counter:
    """Particles by reported polymer.

    Uses the *reported* class, not plain ``class_name``: under active
    abstention, a particle whose confidence does not support the polymer
    counts under `«no asignable»`. It is still a detected particle — just no
    polymer is attributed to it.
    """
    c = Counter()
    for d in detections:
        c[class_id_reported(d)] += 1
    return c


def _classes_views(results_planes) -> list[str]:
    """Classes present in the batch, in presentation order.

    Taken from the detections **and** from the Ground Truth: a polymer the
    model never detects would still be a table row, with a zero, which is
    exactly the figure worth looking at.
    """
    views = {class_id_reported(d) for r in results_planes for d in list(r.predictions) + list(r.gt)}
    order = [c for c in _ORDER_CLASSES if c in views]
    # `«no asignable»` goes at the end and not alphabetically among the polymers: it is not
    # a polymer, it is the absence of one.
    rest = sorted(views - set(order) - {NO_ASSIGNABLE})
    return order + rest + ([NO_ASSIGNABLE] if NO_ASSIGNABLE in views else [])


def _sample_of(path_2) -> tuple | None:
    """(``estacion``, ``tramo``, ``placa``) parsed from the name, or None if it does
    not follow the naming pattern.

    The report cannot demand this naming: if the images come from another
    study, the grouped table is simply skipped and the per-image table
    stays, which can always be built.
    """
    stem = Path(path_2).stem.split("__")[0]
    m = _RE_PLATE.match(stem)
    if not m:
        return None
    range_value, witness, suffix = m.group(1), m.group(2), m.group(3)
    return (
        _STATION.get(witness, f"Testigo {witness}"),
        int(range_value),
        _PLATE.get(suffix, suffix),
    )


def _clue_order(path_2) -> tuple:
    """Sorts by ``estacion``/``tramo``/``placa`` when the name allows it."""
    m = _sample_of(path_2)
    if m is None:
        return (1, 0, 0, "", Path(path_2).name)
    station, range_value, plate = m
    idx = _ORDER_STATION.index(station) if station in _ORDER_STATION else 99
    return (0, idx, range_value, plate, Path(path_2).name)


def _esc(v) -> str:
    """Escapes an interpolated VALUE into HTML: <, > and &; quotes stay.

    ``quote=False`` is deliberate: report values are not placed between
    attribute delimiters, and escaping quotes too would keep intact the
    single quote of a real file name with apostrophes (the byte identity
    of the benign render depends on that). The template literals (td, tr,
    span, styles) never go through here: only the data the batch provides.
    """
    return html.escape(str(v), quote=False)


def _table_count(manual: Counter | None, detected: Counter, classes: list[str]) -> str:
    """Per-photo table: particles by polymer, counted by hand and by the model.

    It is a count, not an evaluation: the model column is **all** of its
    detections for that polymer, without discounting false positives or
    matching box to box. That is what the error analysis is for.
    """
    there_is_a_manual = manual is not None
    rows_2 = ""
    for c in classes:
        det = detected.get(c, 0)
        if there_is_a_manual:
            man = manual.get(c, 0)
            dif = det - man
            css = "dif-pos" if dif > 0 else ("dif-neg" if dif < 0 else "")
            rows_2 += (
                f"<tr><td>{_esc(tr(c))}</td><td>{man}</td><td>{det}</td>"
                f"<td class='{css}'>{dif:+d}</td></tr>"
            )
        else:
            rows_2 += f"<tr><td>{_esc(tr(c))}</td><td>{det}</td></tr>"

    tot_det = sum(detected.get(c, 0) for c in classes)
    if there_is_a_manual:
        tot_man = sum(manual.get(c, 0) for c in classes)
        dif = tot_det - tot_man
        css = "dif-pos" if dif > 0 else ("dif-neg" if dif < 0 else "")
        rows_2 += (
            f"<tr class='tot'><td>Total</td><td>{tot_man}</td>"
            f"<td>{tot_det}</td><td class='{css}'>{dif:+d}</td></tr>"
        )
        cabecera = tr(
            "<tr><th>Polímero</th><th>Conteo manual</th>"
            "<th>Detectadas por el modelo</th><th>Diferencia</th></tr>"
        )
    else:
        rows_2 += f"<tr class='tot'><td>Total</td><td>{tot_det}</td></tr>"
        cabecera = tr("<tr><th>Polímero</th><th>Detectadas por el modelo</th></tr>")

    return f"<table class='conteo'>{cabecera}{rows_2}</table>"


# Maximum width of gallery images, in pixels. They are embedded as base64
# inside the HTML, so their size is the size of the report: without downscaling, four
# 4096 px images weighed 14.5 MB and a batch of several hundred produced a
# gigabyte file that no browser opens.
_WIDTH_GALLERY = 1100


def _uri_gallery(data: bytes, width_max: int = _WIDTH_GALLERY) -> str:
    """Data URI for a gallery image: always JPEG, re-scaled when wide.

    It is re-encoded **always**, not only when shrinking. The annotated
    ones are generated in PNG (lossless), a terrible format for
    photography: one plate used to weigh ~3.6 MB and four images produced
    a 14.5 MB report. As JPEG the same image drops an order of magnitude
    with no visible difference.

    Returns the full URI with its MIME type, not just the base64:
    declaring it as PNG after re-encoding would stop the browser from
    showing it.
    """
    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            if arr.shape[1] > width_max:
                height = int(round(arr.shape[0] * width_max / arr.shape[1]))  # noqa: RUF046
                arr = cv2.resize(arr, (width_max, height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:  # noqa: BLE001, S110
        pass
    return "data:image/png;base64," + _img_b64(data)


def _team() -> list[tuple]:
    """Machine components, so the report can state where it was run."""
    import platform

    rows_2 = [
        (tr("Sistema operativo"), f"{platform.system()} {platform.release()}"),
        (tr("Procesador"), platform.processor() or "—"),
    ]
    try:
        import psutil

        rows_2.append((tr("Memoria RAM"), f"{psutil.virtual_memory().total / 1024**3:.1f} GB"))
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            rows_2 += [
                ("GPU", props.name),
                ("VRAM", f"{props.total_memory / 1024**3:.1f} GB"),
                ("Compute capability", f"{props.major}.{props.minor}"),
                (
                    "PyTorch",
                    f"{torch.__version__} ({'ROCm' if getattr(torch.version, 'hip', None) else 'CUDA'} {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)})",
                ),
            ]
        else:
            rows_2 += [("GPU", tr("no disponible")), ("PyTorch", torch.__version__)]
    except Exception:  # noqa: BLE001, S110
        pass
    return rows_2


def _training_of(weight: Path) -> dict | None:
    """Reads from the checkpoint with which configuration and result the model
    was trained.

    Ultralytics keeps ``train_args`` and ``train_metrics`` inside the .pt
    itself, so the report can declare imgsz, batch, epochs and metrics
    without requiring the run folder to still exist.
    """
    try:
        import torch

        ck = torch.load(str(weight), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    args = ck.get("train_args") or {}
    met = ck.get("train_metrics") or {}
    if not args and not met:
        return None
    return {
        "base": args.get("model", "—"),
        "imgsz": args.get("imgsz", "—"),
        "batch": args.get("batch", "—"),
        "epocas": args.get("epochs", "—"),
        "optimizador": args.get("optimizer", "—"),
        "lr0": args.get("lr0", "—"),
        "amp": args.get("amp", "—"),
        "precision": met.get("metrics/precision(B)"),
        "recall": met.get("metrics/recall(B)"),
        "map50": met.get("metrics/mAP50(B)"),
        "map": met.get("metrics/mAP50-95(B)"),
    }


def _section_team(active) -> str:
    """Table of components and of how each loaded model was trained."""
    rows_eq = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in _team())
    html = (
        f"<h3>@@N@@.1 {tr('Equipo de cómputo')}</h3>"
        f"<table class='data'><tr>{tr('<th>Componente</th><th>Detalle</th>')}</tr>"
        f"{rows_eq}</table>"
    )

    training_runs = []
    for s in active:
        if s.path is None:
            continue
        info = _training_of(Path(s.path))
        if info:
            training_runs.append((s.alias, info))
    if not training_runs:
        return html

    def _n(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    rows_2 = ""
    for alias, i in training_runs:
        rows_2 += (
            f"<tr><td>{_esc(alias)}</td><td>{_esc(i['base'])}</td>"
            f"<td class='r'>{_esc(i['imgsz'])}</td><td class='r'>{_esc(i['batch'])}</td>"
            f"<td class='r'>{_esc(i['epocas'])}</td><td>{_esc(i['optimizador'])}</td>"
            f"<td class='r'>{_n(i['precision'])}</td><td class='r'>{_n(i['recall'])}</td>"
            f"<td class='r'>{_n(i['map50'])}</td><td class='r'>{_n(i['map'])}</td></tr>"
        )

    html += (
        f"<h3>@@N@@.2 {tr('Entrenamiento de cada modelo')}</h3>"
        + tr(
            "<p>Configuración y métricas de validación con que se entrenó cada peso, "
            "leídas del propio archivo <code>.pt</code>.</p>"
        )
        + "<table class='data'>"
        + tr(
            "<tr><th>Modelo</th><th>Arquitectura base</th><th>imgsz</th><th>batch</th>"
            "<th>épocas</th><th>optimizador</th><th>Precisión</th><th>Recall</th>"
            "<th>mAP@50</th><th>mAP@50-95</th></tr>"
        )
        + f"{rows_2}</table>"
    )
    return html


def _sweep_confidence(results_2, state, iou_thr: float, conf_current: float):
    """Recomputes P/R/F1 at different confidence thresholds, without re-inferring.

    The saved predictions already passed the threshold the run used, so
    raising it is just filtering and re-pairing. Lowering it is NOT
    possible: those detections were never computed. That is why the sweep
    starts at the threshold used and states that limitation.

    Returns {alias: [(conf, precision, recall, f1, n_det), ...]}.
    """
    from polyx.core.quality.metrics import match_image

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


_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


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

_MAX_GROUPS_BOXPLOT = 12
_MAX_FOLDER_PER_PHOTO = 6


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
_NAME_MORPHOTYPE = {"fibra": "Fibra", "fragmento": "Partícula"}


def _morphotype(value: str | None, lowercase: bool = False) -> str:
    """Presentable name for a morphotype. '—' when it could not be determined."""
    if not value:
        return "—"
    name_2 = tr(_NAME_MORPHOTYPE.get(value, value))
    return name_2.lower() if lowercase else name_2


def choose_records(shapes: list[tuple], max_records: int) -> list[tuple]:
    """Which particles are shown with their measurement drawn, and in what order.

    ``formas`` is a list of ``(Detection, ruta)`` pairs. At most
    ``max_fichas`` of them are returned.

    Before, the FIRST ones by file name order entered, which had two bad
    consequences:

      * The LARGEST particle — the one backing any claim about maximum
        size — could be left out for living in a high-numbered photo.
      * Fibers never showed up in practice. They are ~1% of the material,
        so the cut arrived long before finding one. And the fiber is
        exactly the case where the geodesic diameter acts: if no
        measurement is seen, there is no way to judge whether that method
        works.

    Now the sample is split between the two morphotypes, in halves, and
    within each one size rules. That way the largest of each type always
    enters.

    The split is not rigid: if there are not enough fibers — the normal
    case — the biggest particles fill the gap instead of returning a
    short sample. And the other way around too.

    It is a deliberately BIASED sample, not representative of the batch;
    the report declares that where it shows it.
    """
    if max_records <= 0:
        return []

    def size_2(par) -> float:
        return par[0].length_um or 0.0

    fibers = sorted([t for t in shapes if t[0].morphotype == "fibra"], key=size_2, reverse=True)
    others = sorted([t for t in shapes if t[0].morphotype != "fibra"], key=size_2, reverse=True)

    half = max(1, max_records // 2)
    chosen = fibers[:half]
    chosen += others[: max_records - len(chosen)]
    # Gaps left by the scarce group are covered by the other one, so as not to deliver a
    # shorter sample than requested when material is available.
    if len(chosen) < max_records:
        ya = {id(d) for d, _ in chosen}
        leftovers = [t for t in fibers + others if id(t[0]) not in ya]
        chosen += leftovers[: max_records - len(chosen)]
    return chosen


def section_depth(shapes, results_2) -> str:
    """How many particles and of which size sit at each depth of the core.

    The interval is not just any folder: it is an ORDERED variable. That
    makes it possible to ask about TRENDS — whether there are more
    particles at the bottom than at the top, or whether the bottom ones
    are bigger — and not only whether the groups differ, which is all the
    generic per-folder comparison can answer.

    ``formas`` is ``[(Detection, ruta)]`` with a measured size;
    ``resultados`` are the ImageResult records, needed to count ALL the
    particles and not only the ones that could be measured in
    micrometers.

    Returns "" when the names do not follow the ``tramo.testigo``
    pattern: without an interval there is no depth, and the report cannot
    demand that naming.
    """
    # -- Count by stretch, summing the plates of the same stretch --
    # They are summed because they are subsamples of the same sediment mass, not
    # photographic replicates. It is the same rule as the count section.
    count: dict[tuple, int] = {}
    plates: dict[tuple, set] = {}
    for r in results_2:
        m = _sample_of(r.image_path)
        if m is None:
            continue
        station, range_value, plate = m
        count[(station, range_value)] = count.get((station, range_value), 0) + len(r.predictions)
        plates.setdefault((station, range_value), set()).add(plate)

    # -- Sizes by stretch --
    sizes_2: dict[tuple, list[float]] = {}
    for d, path_2 in shapes:
        m = _sample_of(path_2)
        if m is None or not d.length_um:
            continue
        sizes_2.setdefault((m[0], m[1]), []).append(float(d.length_um))

    if len(count) < 3:
        # With one or two stretches there is no profile to draw or trend to
        # measure; saying so is more useful than a two-point figure.
        return ""

    profiles: dict[str, list] = {}
    for (station, range_value), n in sorted(count.items()):
        vals = np.asarray(sizes_2.get((station, range_value), []), dtype=float)
        if len(vals):
            median_2 = float(np.median(vals))
            q1, q3 = (float(np.percentile(vals, 25)), float(np.percentile(vals, 75)))
        else:
            median_2 = q1 = q3 = float("nan")
        profiles.setdefault(station, []).append((range_value, n, median_2, q1, q3))

    fig = _fig_profile_depth(profiles)

    # -- Trend, by station --
    # n is the number of STRETCHES, not of particles: two particles from the same
    # plate are not independent observations of depth, and counting them
    # as such would inflate any significance. With few stretches the
    # correlation is weak by design, and that is why n is declared.
    paragraphs = []
    for station, rows_2 in sorted(profiles.items()):
        rows_2 = sorted(rows_2)  # noqa: PLW2901
        x = [f[0] for f in rows_2]
        n_part = [f[1] for f in rows_2]
        med = [f[2] for f in rows_2]
        with_size = [(a, b) for a, b in zip(x, med) if b == b]  # discards NaN  # noqa: B905
        parts = [_text_trend(x, n_part, tr("el número de partículas"))]
        if len(with_size) >= 3:
            parts.append(
                _text_trend(
                    [a for a, _ in with_size], [b for _, b in with_size], tr("la talla mediana")
                )
            )
        parts = [p for p in parts if p]
        if parts:
            paragraphs.append(f"<p><strong>{_esc(station)}.</strong> " + " ".join(parts) + "</p>")

    # -- Table --
    rows_table = ""
    for station, rows_2 in sorted(profiles.items()):
        for range_value, n, median_2, _q1, _q3 in sorted(rows_2):
            vals = sizes_2.get((station, range_value), [])
            # A stretch can have counted particles and none with a size, if
            # that photo was never calibrated. The count is shown and a dash in
            # the size columns, which is what actually happened.
            cells = [
                _esc(station),
                str(range_value),
                str(len(plates.get((station, range_value), ()))),
                str(n),
                str(len(vals)),
                f"{median_2:.0f}" if vals else "—",
                f"{max(vals):.0f}" if vals else "—",
            ]
            rows_table += (
                "<tr><td>"
                + cells[0]
                + "</td>"
                + "".join(f"<td class='r'>{c}</td>" for c in cells[1:])
                + "</tr>"
            )

    return (
        tr(
            "<p>El tramo es la <strong>profundidad</strong> a la que se tomó el "
            "sedimento, y a diferencia de una carpeta cualquiera está "
            "<em>ordenado</em>. Eso permite preguntar por una tendencia —si hay "
            "más partículas abajo que arriba, o si las de abajo son más "
            "grandes— y no solo si los grupos difieren entre sí.</p>"
        )
        + (f"<img src='data:image/png;base64,{fig}' style='max-width:100%'>" if fig else "")
        + "".join(paragraphs)
        + tr(
            "<p class='caption' style='text-align:left'>La correlación se calcula "
            "sobre los <strong>tramos</strong>, no sobre las partículas: dos "
            "partículas de la misma placa no son observaciones independientes de la "
            "profundidad, y contarlas como tales inflaría cualquier significación. "
            "El valor de p sale de barajar la serie 20 000 veces, no de una "
            "aproximación que con tan pocos tramos no valdría.</p>"
        )
        + f"<table class='data'><tr>{tr('<th>Estación</th><th>Tramo</th><th>Placas</th><th>Partículas</th><th>Con talla</th><th>Mediana (µm)</th><th>Mayor (µm)</th>')}</tr>{rows_table}</table>"
    )


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


def _record_particle(det, image_path_2, um_por_px: float | None, title: str) -> str:
    """Crop of a particle with its mask and the px-to-µm count.

    The complete conversion is shown — pixels measured, the scale of
    THAT photo and the result — because a size in µm without the count
    that produced it cannot be checked, and this is the particle that
    carries the most weight in any conclusion about maximum size.
    """
    if det is None or not image_path_2:
        return ""
    try:
        # cv2 is imported here and not at the top: this module does so in the rest of
        # the functions to avoid loading OpenCV when only text is requested.
        import cv2

        from polyx.core.measure.calibration import read_image
        from polyx.core.measure.morphology import box_margin, segment

        bgr = read_image(image_path_2)
        if bgr is None:
            return ""
        # THE SAME MARGIN THAT segmentar() USES, not a similar one. The mask comes
        # cropped with margen_de_caja(); if it is cropped here with another number, the
        # contour is drawn from a different origin and ends up displaced over the
        # photo. It already happened with a 0.35 copied by hand against the real 0.50, and it was seen
        # by eye in the report: `la mascara quedo corrida`.
        mg = box_margin(det.x2 - det.x1, det.y2 - det.y1)
        x1 = max(0, int(det.x1) - mg)
        y1 = max(0, int(det.y1) - mg)
        x2 = min(bgr.shape[1], int(det.x2) + mg)
        y2 = min(bgr.shape[0], int(det.y2) + mg)
        sub = bgr[y1:y2, x1:x2].copy()
        if sub.size == 0:
            return ""
        mask = segment(bgr, det.x1, det.y1, det.x2, det.y2)
        ov = sub.copy()
        if mask is not None:
            h = min(ov.shape[0], mask.shape[0])
            w = min(ov.shape[1], mask.shape[1])
            # Contour and not fill: covering the particle in green prevents judging
            # whether the mask really follows it, which is exactly what is shown.
            cont, _ = cv2.findContours(mask[:h, :w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov[:h, :w], cont, -1, (0, 255, 0), max(1, min(ov.shape[:2]) // 90))
        par = np.hstack([sub, ov])
        scale = min(4.0, 620 / max(1, par.shape[1]))
        if scale > 1:
            par = cv2.resize(par, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", par)
        if not ok:
            return ""
        img = f"<img src='data:image/png;base64,{_img_b64(buf.tobytes())}' style='max-width:100%'>"
    except Exception:  # noqa: BLE001
        # A particle that cannot be cropped cannot bring down the report
        # entirely: the card is illustrative and the tables are already computed.
        return ""

    length_px_2 = (det.length_um / um_por_px) if (det.length_um and um_por_px) else None
    account = ""
    if length_px_2 and um_por_px:
        account = (
            f"<p style='font-family:monospace;font-size:10pt'>"
            f"{length_px_2:.1f} px &times; {um_por_px:.4f} µm/px = "
            f"<strong>{det.length_um:.0f} µm</strong></p>"
        )
        # The three measurements together: if the equivalent rectangle jumps
        # relative to the other two, the particle edge is jagged and the
        # size deserves a look.
        others = []
        if det.feret_um:
            others.append(
                f"<tr><td>{tr('Feret máximo (cuerda)')}</td><td>{det.feret_um:.0f}</td></tr>"
            )
        if det.geodesic_um:
            others.append(
                f"<tr><td>{tr('Geodésico (sigue la curva)')}</td><td>{det.geodesic_um:.0f}</td></tr>"
            )
        if others:
            account += (
                f"<table class='data'><tr>{tr('<th>Medida</th><th>µm</th>')}</tr>"
                f"{''.join(others)}</table>"
                f"<p>{tr('Se reporta la mayor de las dos primeras')}"
                + (
                    tr(", en este caso por <strong>{metodo}</strong>").format(
                        metodo=tr(det.method_length)
                    )
                    if det.method_length
                    else ""
                )
                + ".</p>"
            )
    return (
        f"<h3>{NUM}.{title}</h3>"
        + tr(
            "<p>A la izquierda la partícula; a la derecha, en verde, el "
            "contorno que se midió sobre ella.</p>"
        )
        + f"{img}{account}"
    )


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


def _table_width(
    rows_2: list[tuple],
    classes: list[str],
    title_col: str,
    there_is_a_manual: bool,
    totals: bool = True,
) -> str:
    """Sample x polymer table with the manual and model columns side by side.

    ``filas`` is a list of (``etiqueta``, manual ``Counter``, model
    ``Counter``).
    """
    if there_is_a_manual:
        header_line = (
            f"<tr><th rowspan='2'>{_esc(title_col)}</th>"
            + "".join(
                f"<th colspan='2' style='text-align:center'>{_esc(tr(c))}</th>" for c in classes
            )
            + "<th colspan='2' style='text-align:center'>Total</th></tr><tr>"
            + "".join(
                "<th class='r'>manual</th><th class='r'>modelo</th>"
                for _ in classes + ["total"]  # noqa: RUF005
            )
            + "</tr>"
        )
    else:
        header_line = (
            f"<tr><th>{_esc(title_col)}</th>"
            + "".join(f"<th class='r'>{_esc(tr(c))}</th>" for c in classes)
            + "<th class='r'>Total</th></tr>"
        )

    body = ""
    accumulated_man, accumulated_det = Counter(), Counter()
    for label_2, man, det in rows_2:
        accumulated_man.update(man)
        accumulated_det.update(det)
        cells = ""
        for c in classes:
            if there_is_a_manual:
                cells += f"<td class='r'>{man.get(c, 0)}</td><td class='r'>{det.get(c, 0)}</td>"
            else:
                cells += f"<td class='r'>{det.get(c, 0)}</td>"
        t_det = sum(det.get(c, 0) for c in classes)
        if there_is_a_manual:
            t_man = sum(man.get(c, 0) for c in classes)
            cells += f"<td class='r'><strong>{t_man}</strong></td><td class='r'><strong>{t_det}</strong></td>"
        else:
            cells += f"<td class='r'><strong>{t_det}</strong></td>"
        body += f"<tr><td>{label_2}</td>{cells}</tr>"

    if totals and rows_2:
        cells = ""
        for c in classes:
            if there_is_a_manual:
                cells += (
                    f"<td class='r'>{accumulated_man.get(c, 0)}</td>"
                    f"<td class='r'>{accumulated_det.get(c, 0)}</td>"
                )
            else:
                cells += f"<td class='r'>{accumulated_det.get(c, 0)}</td>"
        t_det = sum(accumulated_det.get(c, 0) for c in classes)
        if there_is_a_manual:
            t_man = sum(accumulated_man.get(c, 0) for c in classes)
            cells += f"<td class='r'>{t_man}</td><td class='r'>{t_det}</td>"
        else:
            cells += f"<td class='r'>{t_det}</td>"
        body += f"<tr style='font-weight:700;background:var(--bg_soft)'><td>TOTAL</td>{cells}</tr>"

    return f"<table class='data'>{header_line}{body}</table>"


def _section_count(results_2: dict[int, list], state, active, classes: list[str]) -> str:
    """Section 8: how many particles of each polymer there are per sample.

    It is the table requested for the manuscript: count by plastic type
    and sample, with the manual count and the model count one next to the
    other. **Not** an evaluation of the detector: the model column is all
    of its detections, without discounting false positives. That reading
    lives in the error section.

    With several models loaded the first active one is used. Putting two
    models in the same cell would make the table illegible, and the
    report already compares models in its own section.
    """
    if not active:
        return ""
    main_mi = state.model_slots.index(active[0])
    rs = results_2.get(main_mi, [])
    if not rs:
        return ""

    rs = sorted(rs, key=lambda r: _clue_order(r.image_path))
    there_is_a_manual = any(r.has_gt for r in rs)

    # -- Per image --
    rows_img = []
    for r in rs:
        sample = _sample_of(r.image_path)
        label_2 = _esc(r.image_path.name)
        if sample:
            station, range_value, plate = sample
            label_2 = (
                f"{_esc(r.image_path.name)}<br>"
                f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                f"{_esc(station)} &middot; tramo {_esc(range_value)} &middot; placa {_esc(plate)}</span>"
            )
        rows_img.append(
            (
                label_2,
                _count_per_class(r.gt) if r.has_gt else Counter(),
                _count_per_class(r.predictions),
            )
        )
    table_img = _table_width(rows_img, classes, "Sample (image)", there_is_a_manual)

    # -- Per stretch and per station, if the names follow the pattern --
    grouped = [(r, _sample_of(r.image_path)) for r in rs]
    parseable = [(r, m) for r, m in grouped if m is not None]
    table_range = table_station = ""
    note_grouping = ""
    if parseable:
        per_section: dict[tuple, list] = {}
        per_station: dict[str, list] = {}
        for r, (station, range_value, _plate) in parseable:
            per_section.setdefault((station, range_value), []).append(r)
            per_station.setdefault(station, []).append(r)

        def _add(items_2):
            man, det = Counter(), Counter()
            for r in items_2:
                if r.has_gt:
                    man.update(_count_per_class(r.gt))
                det.update(_count_per_class(r.predictions))
            return man, det

        def _order_est(e):
            return _ORDER_STATION.index(e) if e in _ORDER_STATION else 99

        rows_t = []
        for clue in sorted(per_section, key=lambda k: (_order_est(k[0]), k[1])):
            station, range_value = clue
            items_2 = per_section[clue]
            man, det = _add(items_2)
            palabra = tr("imágenes") if len(items_2) != 1 else tr("imagen")
            rows_t.append(
                (
                    f"{_esc(station)} &middot; tramo {_esc(range_value)} "  # noqa: ISC004
                    f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                    f"({len(items_2)} {palabra})</span>",
                    man,
                    det,
                )
            )
        table_range = (
            "<h3>@@N@@.2 "
            + tr("Por tramo de profundidad")
            + "</h3>"
            + tr(
                "<p>Las placas del mismo tramo se <strong>suman</strong>: son "
                "submuestras de la misma masa de sedimento, no repeticiones "
                "fotográficas. El tramo es la unidad de análisis.</p>"
            )
            + _table_width(rows_t, classes, tr("Tramo"), there_is_a_manual)
        )

        rows_e = []
        for station in sorted(per_station, key=_order_est):
            man, det = _add(per_station[station])
            rows_e.append((_esc(station), man, det))
        table_station = (
            "<h3>@@N@@.3 "
            + tr("Por estación")
            + "</h3>"
            + _table_width(rows_e, classes, tr("Estación"), there_is_a_manual, totals=False)
        )

        outside = len(grouped) - len(parseable)
        if outside:
            note_grouping = (
                "<p class='caption'>"
                + tr(
                    "{n} imagen(es) no siguen la nomenclatura <code>tramo.testigo</code> y "
                    "quedan fuera de las tablas agrupadas; sí están en la tabla por "
                    "imagen."
                ).format(n=outside)
                + "</p>"
            )

    tagline = tr(", el primer modelo activo.") if len(active) > 1 else "."
    intro = (
        tr(
            "<p>Partículas contadas en cada muestra, desglosadas por polímero. "
            "La columna <em>manual</em> es la anotación humana (Ground Truth) y la "
            "columna <em>modelo</em> son todas las detecciones de "
            "<strong>{alias}</strong>{coletilla}</p>"
        ).format(alias=_esc(active[0].alias), coletilla=tagline)
        + tr(
            "<p><strong>Léase como conteo, no como evaluación.</strong> La "
            "columna del modelo no descuenta falsos positivos ni empareja caja a caja: "
            "es cuántas partículas de cada polímero reportó. Coincidir en el total no "
            "implica haber acertado partícula por partícula; para eso está el análisis "
            "de errores.</p>"
        )
        + "<h3>@@N@@.1 "
        + tr("Por imagen")
        + "</h3>"
    )
    return intro + table_img + note_grouping + table_range + table_station


# ────────────────────────────────────────────────────────────────────
# ── Report sections ────────────────────────────────────────────────────────
# The order of this list is the order of the report, and the numbering comes from here:
# if a section is unchecked the following ones renumber themselves, instead of leaving
# a gap (`4, 6, 7`) that in a deliverable document reads as an error.
#
# Adding a new section is adding a tuple: the interface draws its checkboxes
# by reading this list, so there is no second list to keep in
# sync.
SECTIONS = [
    ("resumen", "Resumen"),
    ("metodos", "Métodos"),
    ("calibracion", "Calibración de escala"),
    ("forma", "Forma y talla de las partículas"),
    ("talla_carpetas", "Talla por carpeta y por foto (comparación entre sitios)"),
    ("profundidad", "Perfil en profundidad del testigo"),
    ("fichas", "Ficha de partículas medidas (muestra)"),
    ("resultados", "Resultados generales"),
    ("modelos", "Resumen por modelo"),
    ("errores", "Análisis de errores"),
    ("comparacion", "Comparación entre modelos"),
    ("galeria", "Galería por imagen"),
    ("conteo", "Conteo por muestra y tipo de plástico"),
    ("referencias", "Referencias bibliográficas"),
]

IDS_SECTIONS = [s[0] for s in SECTIONS]

# Frequently used combinations. They exist because the real problem was not that choosing
# was impossible, but that choosing ten checkboxes one by one is tedious and
# nobody does it: with a preset the short report is one click away.
PRESETS = {
    "completo": IDS_SECTIONS,
    "resumen": ["resumen", "calibracion", "forma", "conteo", "galeria"],
    "metodologico": [
        "resumen",
        "metodos",
        "calibracion",
        "forma",
        "fichas",
        "modelos",
        "errores",
        "comparacion",
        "referencias",
    ],
}

# Marker that is replaced by the final section number during assembly.
# Numbers cannot be written by hand in the titles because they depend on which
# other sections were requested.
NUM = "@@N@@"


# ── Report section builders ──
# M004/S02: _generate_report_impl was a 1200+ line function that
# built each section inline. Each block is extracted into a module-level
# builder that returns ONLY the HTML body, without its <h2>: the title and the
# number are still set by the assembler (NUM -> `N`), which is the only one that
# knows which sections survived the filter. The byte identity of the render
# against the frozen baseline (tests/fixtures/baseline_informe/) is the slice
# contract: changing a literal, a tr(), an _esc or the order of computation breaks
# tests/test_render_reporte_benigno.py.


def _pr_f1(tp: int, fp: int, fn: int):
    """Precision, recall and F1 from the tp/fp/fn counters."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def _section_summary(
    total_imgs: int, total_dets: int, avg_conf: float, avg_size: float, active, any_gt: bool
) -> str:
    """Body of the 'resumen' section: KPI cards plus the abstract paragraph.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns the
    heading and the section number.
    """
    # Outside the paragraph f-string: it carries a plural and a numeric format,
    # and nesting them inside {tr(...)} would require quotes of the same type.
    _summary_paragraph = tr(
        "Se analizaron <strong>{imgs} imágenes</strong> con <strong>{n} {modelos}</strong> "
        "YOLO entrenado para detectar microplásticos de PET, PP y LDPE bajo "
        "fluorescencia Nile Red (254 nm). El total de detecciones fue "
        "<strong>{dets}</strong> con una confianza media de <strong>{conf}</strong>."
    ).format(
        imgs=total_imgs,
        n=len(active),
        modelos=tr("modelos") if len(active) != 1 else tr("modelo"),
        dets=total_dets,
        conf=f"{avg_conf:.3f}",
    )

    return f"""
<div class='kpis'>
  <div class='kpi'><div class='l'>{tr("Imágenes")}</div><div class='v'>{total_imgs}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{tr("Detecciones")}</div><div class='v'>{total_dets}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{tr("Conf. media")}</div><div class='v'>{avg_conf:.3f}</div><div class='b' style='background:var(--vio)'></div></div>
  <div class='kpi'><div class='l'>{tr("Tamaño medio (μm)")}</div><div class='v'>{(f"{avg_size:.1f}" if avg_size else "—")}</div><div class='b' style='background:var(--warn)'></div></div>
</div>
<p>{_summary_paragraph}
{tr("Se incluyó análisis de errores con Ground Truth (Verdaderos Positivos, Falsos Positivos, Falsos Negativos y Mal Clasificados).") if any_gt else tr("No se aportó Ground Truth, por lo que no se reportan métricas de error.")}
</p>"""


def _section_errors(state, results_2, active, any_gt: bool) -> str:
    """Body of the 'errores' section: confusion matrix plus per-class metrics.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when there is no
    ground truth or no active model: the assembler omits a section with no
    content rather than inventing data.
    """
    if not (any_gt and active):
        return ""
    main_mi = state.model_slots.index(active[0])
    rs = results_2.get(main_mi, [])
    gts_only = [r.gt for r in rs if r.has_gt]
    preds_only = [r.predictions for r in rs if r.has_gt]
    cls_ids = sorted({d.class_id for lst in gts_only + preds_only for d in lst})
    cls_names = []
    for cid in cls_ids:
        name = next(
            (d.class_name for lst in gts_only + preds_only for d in lst if d.class_id == cid),
            str(cid),
        )
        cls_names.append(name)
    cm = confusion_matrix(preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp)
    per_class_metrics = aggregate_per_class(
        preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp
    )

    cm_img = _fig_confusion_matrix(cm, cls_names)
    rows = ""
    for cid, name in zip(cls_ids, cls_names):  # noqa: B905
        cm_ = per_class_metrics[cid]
        rows += (
            f"<tr><td>{name}</td>"
            f"<td class='r'>{cm_.tp}</td><td class='r'>{cm_.fp}</td><td class='r'>{cm_.fn}</td>"
            f"<td class='r'>{cm_.precision:.3f}</td><td class='r'>{cm_.recall:.3f}</td>"
            f"<td class='r'>{cm_.f1:.3f}</td></tr>"
        )
    # The classes that the model knows but that appear neither in the GT nor in
    # the predictions are left out of the table, because they have no metric
    # to compute. They are declared anyway: a report that announces PET/PP/LDPE and
    # then shows two rows looks broken, and the reader cannot distinguish
    # `absent from the batch` from `lost along the way`.
    known = dict(getattr(getattr(active[0], "loaded", None), "names", {}) or {})
    missing = [n for cid, n in sorted(known.items()) if cid not in set(cls_ids)]
    note_missing = ""
    if missing:
        note_missing = (
            "<p class='caption' style='text-align:left'>"
            + tr(
                "Sin fila para {clases}: no aparece ni en la anotación manual ni "
                "entre las predicciones de este lote, de modo que no hay métrica "
                "que informar. El modelo sí está entrenado para esa clase."
            ).format(clases=_esc(", ".join(missing)))
            + "</p>"
        )

    return f"""
        
        <h3>@@N@@.1 {tr("Matriz de confusión")}</h3>
        <div class='fig'><img src='data:image/png;base64,{cm_img}' />
            <div class='caption'>{tr("Figura. Matriz de confusión (modelo principal: {modelo}, IoU = {iou}).").format(modelo=_esc(active[0].alias), iou=state.params.iou_tp)}</div></div>
        <h3>@@N@@.2 {tr("Precisión / Recall / F1 por clase")}</h3>
        <table class='data'><tr>{tr("<th>Clase</th>")}<th>True Positives</th><th>False Positives</th><th>False Negatives</th>
        {tr("<th>Precisión</th><th>Recall</th><th>F1</th>")}</tr>{rows}</table>
        {note_missing}
        """


def _section_models(state, results_2, all_results, any_gt: bool) -> str:
    """Body of the 'modelos' section: per-model comparison table plus F1 note.

    Returns the HTML body WITHOUT its ``<h2>`` title (header table included:
    the builder owns the complete section body the assembler used to compose).
    """
    # TWO F1 values are reported, and the difference between them is not a detail:
    #
    #         localization  tp/(tp+fp) and tp/(tp+fn). Boxes that are well placed but with
    #                                                                 the wrong class fall outside both denominators,
    #                                                                 so it measures only whether the detector finds the particle.
    #         with class    those same boxes count as a false positive of the predicted
    #                                                                 class and a false negative of the real one, which is the strict criterion.
    #
    # Publishing only the first overestimates performance, and also contradicts the
    # per-class table, which does penalize confusion. Both are declared.
    rows_models = ""
    total_mc = 0
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None:
            continue
        rs = results_2.get(mi, [])
        n_img = len({r.image_path for r in rs})
        n_det = sum(len(r.predictions) for r in rs)
        cf = [p.conf for r in rs for p in r.predictions]
        tp = sum(r.tp for r in rs)
        fp = sum(r.fp for r in rs)
        fn = sum(r.fn for r in rs)
        mc = sum(r.miscls for r in rs)
        total_mc += mc
        _, _, f1_loc = _pr_f1(tp, fp, fn)
        _, _, f1_cls = _pr_f1(tp, fp + mc, fn + mc)
        avg_cf = (sum(cf) / len(cf)) if cf else 0
        there_is_gt = any(r.has_gt for r in rs)
        c_loc = f"{f1_loc:.3f}" if there_is_gt else "—"
        c_cls = f"{f1_cls:.3f}" if there_is_gt else "—"
        rows_models += (
            f"<tr><td>{_esc(slot.alias)}</td><td class='r'>{n_img}</td><td class='r'>{n_det}</td>"
            f"<td class='r'>{avg_cf:.3f}</td>"
            f"<td class='r'>{tp}</td><td class='r'>{fp}</td><td class='r'>{fn}</td>"
            f"<td class='r'>{mc}</td>"
            f"<td class='r'>{c_loc}</td><td class='r'>{c_cls}</td></tr>"
        )

    # Note that accompanies the table, with the figures from the batch itself.
    note_miscls = ""
    if any_gt:
        tp_g = sum(r.tp for r in all_results)
        fp_g = sum(r.fp for r in all_results)
        fn_g = sum(r.fn for r in all_results)
        p_loc, r_loc, f_loc = _pr_f1(tp_g, fp_g, fn_g)
        p_cls, r_cls, f_cls = _pr_f1(tp_g, fp_g + total_mc, fn_g + total_mc)
        note_miscls = tr(
            "<p><strong>Los dos F1 miden cosas distintas.</strong> "
            "<em>Localización</em> responde si el detector encuentra la partícula "
            "(P {pl} · R {rl} · <strong>F1 {fl}</strong>). "
            "<em>Con clase</em> exige además acertar el polímero, contando cada "
            "caja mal clasificada como falso positivo de la clase predicha y "
            "falso negativo de la real "
            "(P {pc} · R {rc} · <strong>F1 {fc}</strong>).</p>"
            "<p>La diferencia corresponde a <strong>{mc}</strong> "
            "partícula(s) bien localizada(s) pero asignada(s) a la clase "
            "incorrecta. Es la cifra que concilia esta tabla con la de "
            "precisión por clase de la sección de errores.</p>"
        ).format(
            pl=f"{p_loc:.3f}",
            rl=f"{r_loc:.3f}",
            fl=f"{f_loc:.3f}",
            pc=f"{p_cls:.3f}",
            rc=f"{r_cls:.3f}",
            fc=f"{f_cls:.3f}",
            mc=total_mc,
        )

    return f"""
<table class='data'><tr>{tr("<th>Modelo</th><th>Imágenes</th><th>Detecciones</th><th>Conf. media</th>")}
<th>True Positives</th><th>False Positives</th><th>False Negatives</th>
<th>Misclassified</th><th>F1<br><span style='font-weight:400;font-size:8.5pt'>{tr("localización")}</span></th>
<th>F1<br><span style='font-weight:400;font-size:8.5pt'>{tr("con clase")}</span></th></tr>{rows_models}</table>
{note_miscls}"""


def _section_verdict(results_2, state) -> str:
    """Body of the 'veredicto' section: best model by F1-with-class, from the
    confidence sweep. Arguments are the precomputed locals the inline block
    consumed before extraction."""
    text_2 = ""
    if any(r.has_gt for rs in results_2.values() for r in rs):
        sweep = _sweep_confidence(results_2, state, state.params.iou_tp, state.params.conf)
        if sweep:
            summary = []
            for alias, rows_2 in sweep.items():
                best = max(rows_2, key=lambda f: f[3])
                current = rows_2[0]
                summary.append((alias, current, best))

            # Verdict: the winner is whoever has the best F1 with class at their best threshold.
            summary.sort(key=lambda x: x[2][3], reverse=True)
            g_alias, g_act, g_best = summary[0]  # noqa: RUF059
            text_2 = tr(
                "<p><strong>Mejor desempeño: {alias}</strong>, con F1 {f1} "
                "al umbral {umbral} (P {p} · R {r}).</p>"
            ).format(
                alias=_esc(g_alias),
                f1=f"{g_best[3]:.3f}",
                umbral=f"{g_best[0]:.2f}",
                p=f"{g_best[1]:.3f}",
                r=f"{g_best[2]:.3f}",
            )
            if len(summary) > 1:
                p_alias, _, p_best = summary[-1]
                d = g_best[3] - p_best[3]
                text_2 += tr(
                    "<p>La diferencia con {otro} es de <strong>{d}</strong> de F1. Con "
                    "un solo entrenamiento por arquitectura, una diferencia pequeña no "
                    "distingue el diseño de la red del azar de inicialización: haría "
                    "falta repetir con distintas semillas para afirmar que una es "
                    "superior.</p>"
                ).format(otro=_esc(p_alias), d=f"{d:.3f}")

            return text_2
    return text_2


def _section_comparison(state, results_2) -> str:
    """Body of the 'comparacion' section: per-photo model table, the F1 of
    localization note and the comparison figures. Arguments are the
    precomputed locals the inline block consumed before extraction."""
    # ── Models side by side, photo by photo ──
    # The global table above says which model wins overall; this one says in
    # which photos it wins. With uneven batches like those from the Loa, where one stretch
    # concentrates most of the particles, the total is decided by that stretch and
    # can hide that in the rest the other model does better.
    compare_html = ""
    active_2 = [mi for mi, s in enumerate(state.model_slots) if s.path is not None]
    if len(active_2) > 1:
        by_photo = {}
        for mi in active_2:
            for r in results_2.get(mi, []):
                by_photo.setdefault(r.image_path, {})[mi] = r

        there_is_gt = any(r.has_gt for rs in results_2.values() for r in rs)
        header_line = "<tr><th>Image</th>"
        for mi in active_2:
            alias = state.model_slots[mi].alias
            header_line += f"<th>{_esc(alias)}<br><span style='font-weight:400;font-size:8.5pt'>dets</span></th>"
            if there_is_gt:
                header_line += (
                    "<th><span style='font-weight:400;font-size:8.5pt'>"
                    "True Positives/False Positives/False Negatives</span></th>"
                )
        header_line += "</tr>"

        rows_2 = ""
        for path_2 in sorted(by_photo):
            rows_2 += f"<tr><td>{_esc(Path(path_2).name)}</td>"
            for mi in active_2:
                r = by_photo[path_2].get(mi)
                if r is None:
                    rows_2 += "<td class='r'>—</td>"
                    if there_is_gt:
                        rows_2 += "<td class='r'>—</td>"
                    continue
                rows_2 += f"<td class='r'>{len(r.predictions)}</td>"
                if there_is_gt:
                    cell = f"{r.tp}/{r.fp}/{r.fn}" if r.has_gt else "—"
                    rows_2 += f"<td class='r'>{cell}</td>"
            rows_2 += "</tr>"

        # Verdict: only declared if there is ground truth to judge with. Without it,
        # more detections is not better, and saying otherwise would be misleading.
        verdict = ""
        if there_is_gt:
            marker_2 = []
            for mi in active_2:
                rs = results_2.get(mi, [])
                tp = sum(r.tp for r in rs)
                fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs)
                pr = tp / (tp + fp) if (tp + fp) else 0
                rc = tp / (tp + fn) if (tp + fn) else 0
                marker_2.append(
                    (2 * pr * rc / (pr + rc) if (pr + rc) else 0, state.model_slots[mi].alias)
                )
            marker_2.sort(reverse=True)
            best_f1, best = marker_2[0]
            # This F1 is the permissive one (localization). It used to say only `F1 global` and
            # coexisted with the strict verdict above: two different numbers
            # called the same in the same section.
            verdict = tr(
                "<p><b>Mejor F1 de localización:</b> {alias} ({f1}) — encontrar la "
                "partícula, sin exigir que acierte el polímero. El veredicto con clase "
                "está más arriba.</p>"
            ).format(alias=_esc(best), f1=f"{best_f1:.3f}")
        else:
            verdict = (
                "<p class='caption' style='text-align:left'>"
                + tr(
                    "Sin ground truth no se puede declarar un ganador: un "
                    "modelo con más detecciones puede estar acertando o "
                    "inventando. Carga anotaciones para obtener F1 por "
                    "modelo."
                )
                + "</p>"
            )

        # ── Side-by-side figures ──
        figures_cmp = ""
        if there_is_gt:
            data_bars = []
            by_model_class: dict[str, dict[str, tuple]] = {}
            for mi in active_2:
                alias = state.model_slots[mi].alias
                rs = [r for r in results_2.get(mi, []) if r.has_gt]
                tp = sum(r.tp for r in rs)
                fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs)
                mc = sum(r.miscls for r in rs)
                p_l, r_l, f_l = _pr_f1(tp, fp, fn)
                p_c, r_c, f_c = _pr_f1(tp, fp + mc, fn + mc)
                data_bars.append((alias, p_l, r_l, f_l, p_c, r_c, f_c))

                gts_m = [r.gt for r in rs]
                preds_m = [r.predictions for r in rs]
                ids = sorted({d.class_id for lst in gts_m + preds_m for d in lst})
                if ids:
                    metrics = aggregate_per_class(preds_m, gts_m, ids, iou_thr=state.params.iou_tp)
                    names_cls = {
                        cid: next(
                            (
                                d.class_name
                                for lst in gts_m + preds_m
                                for d in lst
                                if d.class_id == cid
                            ),
                            str(cid),
                        )
                        for cid in ids
                    }
                    by_model_class[alias] = {
                        names_cls[cid]: (
                            metrics[cid].precision,
                            metrics[cid].recall,
                            metrics[cid].f1,
                        )
                        for cid in ids
                    }

            f_bars = _fig_comparison_metrics(data_bars)
            if f_bars:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_bars}' />"
                    "<div class='caption'>"
                    + tr(
                        "Figura. Precisión, Recall y F1 de cada modelo al umbral "
                        "{umbral}, en los dos criterios. La distancia entre paneles es "
                        "la confusión entre polímeros."
                    ).format(umbral=f"{state.params.conf:g}")
                    + "</div></div>"
                )

            f_class_id = _fig_metrics_by_class(by_model_class)
            if f_class_id:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_class_id}' />"
                    "<div class='caption'>"
                    + tr(
                        "Figura. F1 por clase. Un modelo puede ganar en el agregado y "
                        "perder en el polímero que interesa."
                    )
                    + "</div></div>"
                )

            # Photo-by-photo agreement between the first two models.
            if len(active_2) >= 2:
                a, b = active_2[0], active_2[1]
                pares = []
                for path_2 in sorted(by_photo):
                    ra, rb = by_photo[path_2].get(a), by_photo[path_2].get(b)
                    if ra is not None and rb is not None:
                        pares.append((len(ra.predictions), len(rb.predictions), Path(path_2).name))
                f_agreement = _fig_agreement_by_image(
                    state.model_slots[a].alias, state.model_slots[b].alias, pares
                )
                if f_agreement:
                    figures_cmp += (
                        f"<div class='fig'><img src='data:image/png;base64,{f_agreement}' />"
                        "<div class='caption'>"
                        + tr(
                            "Figura. Detecciones por imagen de un modelo frente al "
                            "otro; la diagonal es el acuerdo exacto. Escala "
                            "simétrica-logarítmica, porque casi todas las placas "
                            "tienen pocas partículas y unas pocas cientos."
                        )
                        + "</div></div>"
                    )

            # F1 curve against the threshold: it supports the choice of the operating
            # point with evidence, instead of asserting it.
            sweep = _sweep_confidence(results_2, state, state.params.iou_tp, state.params.conf)
            f_sweep = _fig_sweep(sweep)
            if f_sweep:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_sweep}' />"
                    "<div class='caption'>"
                    + tr(
                        "Figura. F1 (con clase) frente al umbral de confianza; la "
                        "estrella marca el máximo de cada modelo. La curva arranca en "
                        "{umbral}, el umbral con que se ejecutó: por debajo las "
                        "detecciones no se calcularon."
                    ).format(umbral=f"{state.params.conf:g}")
                    + "</div></div>"
                )

        compare_html = (
            tr(
                "<p>Detecciones de cada modelo imagen por imagen. El total puede "
                "estar dominado por unas pocas fotos densas, así que conviene mirar "
                "el detalle antes de elegir modelo.</p>"
            )
            + verdict
            + figures_cmp
            + f"<table class='data'>{header_line}{rows_2}</table>"
        )
    elif len(active_2) == 1:
        compare_html = (
            "<p class='caption' style='text-align:left'>"
            + tr(
                "Solo se ejecutó un modelo, así que no hay comparación. "
                "Carga un segundo modelo en la pestaña Modelos para "
                "compararlos sobre las mismas imágenes."
            )
            + "</p>"
        )
    return compare_html


def _section_gallery(
    results_2: dict[int, list],
    state,
    classes_batch,
    max_gallery: int = 60,
    include_gallery: bool = True,
) -> str:
    """Body of the 'galeria' section: side-by-side prediction vs Ground Truth.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns the
    heading and the section number. Empty when the gallery is disabled or no
    candidate image has prediction bytes.
    """
    # ── Side-by-side gallery: Prediction vs Ground Truth ──
    gallery_html = ""
    omitted = 0
    if include_gallery:
        blocks = []
        # Gallery image cap: each one is embedded as base64, so
        # without a limit a batch of several hundred produces a gigabyte HTML.
        # It is grouped by image and not by model. Flattening model by model, the
        # cap was exhausted inside the first one and the second never appeared in
        # the gallery: with two models loaded only the photos from one were seen,
        # which is precisely the visual side-by-side that is wanted. Now the cap counts
        # images, and from each one all models enter, one below the other.
        per_image: dict = {}
        for mi in sorted(results_2):
            for r in results_2[mi]:
                per_image.setdefault(r.image_path, []).append((mi, r))
        # Order by station/stretch/plate when the plate name
        # allows it: the report is read following the core downwards, not in the
        # arbitrary order in which the files were loaded.
        order_imgs = sorted(per_image, key=_clue_order)
        candidates = []
        for path_2 in order_imgs[:max_gallery]:
            candidates.extend(per_image[path_2])
        omitted = max(0, len(per_image) - max_gallery)
        for mi, r in candidates:
            slot = state.model_slots[mi]
            if True:
                # Prediction image (we prefer pred_png; otherwise the combined one)
                pred_bytes = getattr(r, "pred_png", None) or r.annotated_png
                if not pred_bytes:
                    continue
                left = (
                    f"<figure><img src='{_uri_gallery(pred_bytes)}' />"
                    f"<figcaption>{tr('Predicción del modelo')} · {_esc(slot.alias)}"
                    f"<span class='sub'>"
                    + tr("{n} detección(es) por YOLO").format(n=len(r.predictions))
                    + "</span>"
                    "</figcaption></figure>"
                )

                # Ground Truth image (control)
                gt_bytes = getattr(r, "gt_png", None)
                if r.has_gt and gt_bytes:
                    right = (
                        f"<figure><img src='{_uri_gallery(gt_bytes)}' />"
                        f"<figcaption>Ground Truth (control)"
                        f"<span class='sub'>{len(r.gt)} etiqueta(s) reales</span>"
                        f"</figcaption></figure>"
                    )
                else:
                    right = (
                        f"<figure><div class='nogt'>"
                        f"{tr('Sin Ground Truth para esta imagen')}</div>"
                        f"<figcaption>Ground Truth (control)"
                        f"<span class='sub'>{tr('no disponible')}</span>"
                        f"</figcaption></figure>"
                    )

                # Photo header: name and, if the name allows it,
                # which stretch and plate it comes from.
                sample = _sample_of(r.image_path)
                where = ""
                if sample:
                    station, range_value, plate = sample
                    where = (
                        f" &nbsp;<span class='tag'>{_esc(station)} · tramo {_esc(range_value)} "
                        f"· placa {_esc(plate)}</span>"
                    )
                meta = (
                    f"<strong>{_esc(r.image_path.name)}</strong>{where}"
                    f"&nbsp;<span class='tag'>modelo: {_esc(slot.alias)}</span>"
                )

                # Photo table: particles per polymer. Matches and
                # failures paired box by box are another question and live in
                # the error analysis section.
                table = _table_count(
                    _count_per_class(r.gt) if r.has_gt else None,
                    _count_per_class(r.predictions),
                    classes_batch,
                )

                blocks.append(
                    f"<div class='compare'><div class='compare-pair'>{left}{right}</div>"
                    f"<div class='compare-meta'>{meta}</div>{table}</div>"
                )

        if blocks:
            note = ""
            if omitted:
                n_shown = len({r.image_path for _, r in candidates})
                note = (
                    "<p class='caption'>"
                    + tr(
                        "Se muestran las primeras {n} imágenes, cada una con todos los "
                        "modelos; {fuera} quedaron fuera de la galería para que el archivo "
                        "siga siendo manejable. Las métricas de las secciones anteriores "
                        "sí incluyen todas."
                    ).format(n=n_shown, fuera=omitted)
                    + "</p>"
                )
            gallery_html = (
                tr(
                    "<p>Cada bloque muestra, a la izquierda, las detecciones del "
                    "modelo (<em>bounding boxes</em> dibujadas por YOLO con su clase y "
                    "confianza) y, a la derecha, las etiquetas reales de control "
                    "(<em>Ground Truth</em>). Esta vista lado a lado permite evaluar "
                    "visualmente dónde acertó o falló el modelo.</p>"
                )
                + note
                + "".join(blocks)
            )
    return gallery_html


def _section_results(per_class: dict[str, int], confs: list[float], sizes: list[float]) -> str:
    """Body of the ``'resultados'`` section: aggregate figures plus captions.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns
    the heading and the section number. Empty when no figure was
    produced.
    """
    # ── Aggregate figures ──
    fig_classes = _fig_class_distribution(dict(per_class))
    fig_conf = _fig_confidence_hist(confs)
    fig_size = _fig_size_hist(sizes) if sizes else ""

    figures_html = ""
    if fig_classes:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_classes}' /><div class='caption'>{tr('Figura. Distribución de detecciones por clase.')}</div></div>"
    if fig_conf:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_conf}' /><div class='caption'>{tr('Figura. Histograma de confianza.')}</div></div>"
    if fig_size:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_size}' /><div class='caption'>{tr('Figura. Distribución de tamaños (diámetro equivalente en μm).')}</div></div>"
    return figures_html


def _section_methods(
    state, active, all_results, total_imgs: int, total_dets: int, any_gt: bool
) -> tuple:
    """Body of the 'metodos' section: parameters table, team and redacted paragraph.

    Returns ``(methods_html, equipo_html, methods_para)`` — the three pieces
    the assembler's ``metodos`` entry composes — WITHOUT any ``<h2>`` title.
    The ``import ultralytics`` stays lazy inside this body: it must not load
    at module import time (S01 laziness gate).
    """
    # ── Methods ──
    p = state.params
    # When calibration is per photo (medir_placa) there is no SINGLE um_per_px
    # value to report here -- p.um_per_px stays at 0 on purpose.
    # Instead of an `—` that says nothing, the batch mean is given with its
    # 95% confidence interval: a number that can be cited in a
    # method even if the real calibration is photo by photo (that is explained in
    # detail in the `Calibracion de escala` section, with the minimum and
    # maximum that show how much it varies).
    if p.um_per_px > 0:
        um_px_txt = f"{p.um_per_px:g}"
    else:
        _ums_methods = [
            c.um_por_px
            for c in (getattr(state, "calibrations", None) or {}).values()
            if getattr(c, "valid", False)
        ]
        _mic = _average_ic95(_ums_methods)
        if _mic:
            _average_m, _margin_m, _n_m = _mic
            um_px_txt = f"{_average_m:.4f} ± {_margin_m:.4f} (IC 95%, n={_n_m} fotos)"
        elif _ums_methods:
            um_px_txt = f"{_ums_methods[0]:.4f} " + tr("(una sola foto calibrada, sin IC)")
        else:
            um_px_txt = "—"
    methods_rows = [
        (tr("Modelos cargados"), _esc(", ".join(s.alias for s in active)) or "—"),
        (tr("Confianza mínima"), f"{p.conf:g}"),
        ("IoU NMS", f"{p.iou_nms:g}"),
        (tr("IoU para emparejar Verdaderos Positivos"), f"{p.iou_tp:g}"),
        (tr("Tamaño de imagen (imgsz)"), f"{p.imgsz}"),
        (tr("Dispositivo"), p.device),
        (tr("μm por píxel"), um_px_txt),
        (
            tr("Filtro tamaño (μm)"),
            f"{p.size_min_um} – {p.size_max_um}"
            if (p.size_min_um or p.size_max_um)
            else tr("sin filtro"),
        ),
        (tr("Imágenes procesadas"), str(total_imgs)),
        (tr("Total de detecciones"), str(total_dets)),
    ]
    # Abstention: how many particles were counted without attributing a polymer to them. It goes in
    # Methods and not buried in a results table because it is an analysis
    # decision, and whoever reads the report needs to know it was taken.
    _res_assignment = resume(
        [d for r in all_results for d in r.predictions], getattr(p, "conf_assignment", 0.0)
    )
    if _res_assignment.active:
        methods_rows.append(
            (tr("Confianza mínima para asignar polímero"), f"{p.conf_assignment:g}")
        )
        methods_rows.append(
            (
                tr("Partículas sin polímero asignable"),
                f"{_res_assignment.no_assignables} / {_res_assignment.total} "  # noqa: ISC004
                f"({_res_assignment.percentage_no_assignable:.1f} %)",
            )
        )
    methods_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in methods_rows)
    team_html = _section_team(active)

    # Redacted methods paragraph, ready to copy into the manuscript
    # (reproducibility: model, version and exact parameters).
    try:
        import ultralytics

        _ul_see = ultralytics.__version__
    except Exception:  # noqa: BLE001
        _ul_see = "—"
    _model_names = _esc(", ".join(s.alias for s in active)) if active else "—"
    methods_for = (
        "<p><strong>" + tr("Resumen de la configuración empleada:") + "</strong></p>"
        "<blockquote style='border-left:3px solid var(--accent); margin:8px 0; "
        "padding:6px 14px; color:#424a53; background:#f6f8fa; border-radius:0 6px 6px 0;'>"
        + tr(
            "La detección automatizada se realizó con el modelo YOLO «{modelos}» "
            "(Ultralytics {ul}) a una resolución de entrada de {imgsz} px, umbral de "
            "confianza {conf} y supresión de no-máximos con IoU {iou}. "
        ).format(
            modelos=_model_names,
            ul=_ul_see,
            imgsz=p.imgsz,
            conf=f"{p.conf:g}",
            iou=f"{p.iou_nms:g}",
        )
        + (
            tr(
                "Las métricas de error se calcularon contra anotación manual "
                "independiente, emparejando predicciones y etiquetas con IoU ≥ {iou}. "
            ).format(iou=f"{p.iou_tp:g}")
            if any_gt
            else ""
        )
        + (
            tr("La calibración óptica fue de {um} μm/píxel. ").format(um=f"{p.um_per_px:g}")
            if p.um_per_px > 0
            else ""
        )
        + tr("Se procesaron {imgs} imágenes con un total de {dets} detecciones.").format(
            imgs=total_imgs, dets=total_dets
        )
        + "</blockquote>"
    )
    return methods_html, team_html, methods_for


def _section_references(include_refs: bool = True) -> str:
    """Body of the 'referencias' section: static list of cited papers.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when the section
    is not included, so the assembler drops it like it does today.
    """
    refs_html = ""
    if include_refs:
        refs_html = """
        
        <ol>
          <li>Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).
              Development of a new methodology for the determination of PET microplastics in sediment,
              based on microwave-assisted acid digestion.
              <em>PLoS ONE</em> 19(12): e0314520.
              <a href='https://doi.org/10.1371/journal.pone.0314520'>https://doi.org/10.1371/journal.pone.0314520</a></li>
          <li>Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).
              Evaluation of microwave-assisted acid/oxidant digestion method for the detection of
              polyethylene microplastics in <em>Merluccius gayi</em> fish by Nile Red fluorescent
              staining and image analysis. <em>J. Chil. Chem. Soc.</em> 69(1): 6082-6085.
              <a href='https://doi.org/10.4067/s0717-97072024000106082'>https://doi.org/10.4067/s0717-97072024000106082</a></li>
        </ol>
        """
    return refs_html


def _section_calibration(state) -> str:
    """Body of the 'calibracion' section: scale provenance and per-photo calibration.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when the state has
    no calibraciones and no manual scale, so the assembler drops the section
    like it does today. The ``polyx.core.calibration`` import stays lazy
    inside this body: it must not load at module import time (S01 laziness
    gate).
    """
    # ── Scale calibration ──
    # The provenance is reported, not only the number: a size in µm without saying
    # against which standard it was measured is not verifiable. If there is no calibration the
    # section is not emitted, instead of showing an `—` that looks like data.
    p = state.params
    calib_html = ""
    cals = getattr(state, "calibrations", None) or {}
    if cals:
        from polyx.core.measure.calibration import (
            BIAS_MAXIMUM_WALL_PCT,
            THICKNESS_WALL_MM,
            summary_batch,
        )

        res = summary_batch(list(cals.values()))
        if res["n"]:
            label_2 = {
                "placa": tr("medida sobre la placa Petri de esta foto"),
                "indice": tr("heredada del recorte (índice de calibración)"),
                "manual": tr("introducida a mano en Parámetros"),
            }
            rows_2 = "".join(
                f"<tr><td>{label_2.get(o, o)}</td>"
                f"<td>{k} {tr('imágenes') if k != 1 else tr('imagen')}</td></tr>"
                for o, k in sorted(res["origenes"].items())
            )
            warning = ""
            if res["variacion"] > 1.05:
                warning = tr(
                    "<p><strong>La escala no es común a todo el lote:</strong> varía "
                    "{var}× entre la foto más cercana y la más lejana. Por eso cada "
                    "imagen se convierte con su propio factor; usar uno solo para todas "
                    "daría tamaños con hasta un {pct}% de error.</p>"
                ).format(var=f"{res['variacion']:.2f}", pct=f"{100 * (res['variacion'] - 1):.0f}")
            _mic_calib = _average_ic95(
                [c.um_por_px for c in cals.values() if getattr(c, "valid", False)]
            )
            _row_average = ""
            if _mic_calib:
                _average_c, _margin_c, _n_c = _mic_calib
                _row_average = (
                    f"<tr>{tr('<td>media ± IC 95%</td>')}"
                    f"<td>{_average_c:.4f} ± {_margin_c:.4f} (n={_n_c})</td></tr>"
                )
            calib_html = f"""
<table class='data'><tr>{tr("<th>Procedencia de la escala</th><th>Imágenes</th>")}</tr>{rows_2}</table>
<table class='data'>
<tr>{tr("<th>µm por píxel</th><th>Valor</th>")}</tr>
<tr>{tr("<td>mínimo</td>")}<td>{res["min"]:.4f}</td></tr>
<tr>{tr("<td>mediana</td>")}<td>{res["mediana"]:.4f}</td></tr>
<tr>{tr("<td>máximo</td>")}<td>{res["max"]:.4f}</td></tr>
{_row_average}
</table>
{warning}
<p>{tr("El patrón de longitud es el diámetro externo nominal de la placa Petri. El radio en píxeles se obtiene ajustando un círculo por mínimos cuadrados al borde del anillo muestreado en 720 direcciones, con rechazo de atípicos; la transformada de Hough solo aporta el centro aproximado, porque su radio llega a errar un 12&nbsp;% y ese error entraría entero en todos los tamaños reportados.")}</p>

<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;background:var(--bg_soft)'>
{tr("<strong>Incertidumbre de la escala, y en qué dirección.</strong> El anillo de la placa tiene una pared de unos <strong>{pared:g}&nbsp;mm</strong> (medido sobre las fotos de este estudio: el borde interno cae en 0,960 del radio ajustado y el externo en 1,000). El diámetro nominal de una placa Petri es ambiguo a ese nivel: puede referirse al diámetro <em>externo</em> o al <em>útil interior</em>. Aquí se toma el <strong>externo</strong>, que es el borde al que ajusta el círculo. Si el nominal se refiriera al interior, la escala correcta sería un <strong>{sesgo:g}&nbsp;% mayor</strong> y todas las tallas de este informe estarían <strong>subestimadas</strong> en esa cifra. El sesgo solo puede ir en ese sentido, porque el externo es el mayor de los dos bordes posibles.").format(pared=THICKNESS_WALL_MM, sesgo=BIAS_MAXIMUM_WALL_PCT)}</p>
{_figure_calibration(state)}"""
            if res["avisos"]:
                items = "".join(f"<li>{a}</li>" for a in res["avisos"][:12] if a)
                if items:
                    calib_html += (
                        f"<p>{tr('Incidencias durante la calibración:')}</p><ul>{items}</ul>"
                    )
    elif p.um_per_px > 0:
        calib_html = tr(
            "<p>Escala única para todo el lote, introducida a mano: "
            "<strong>{um} µm/píxel</strong>. No se midió ninguna placa, "
            "de modo que este factor no está trazado a un patrón de longitud y "
            "cualquier variación en la distancia de disparo entre fotos queda sin "
            "corregir.</p>"
        ).format(um=f"{p.um_per_px:g}")
    return calib_html


def _lot_forms(state, all_results):
    """Detections with a measured shape, minus the ones over the plate ring.

    Returns ``(formas, n_en_el_aro, n_sin_forma)``: the (detection, image
    path) pairs whose shape was measured and did not fall on the ring, how
    many detections were set apart for falling on the ring, and how many
    predictions could not be segmented at all. The depth profile, the forma
    section and the particle fichas all read the same list, so this runs
    exactly once per report. The ``polyx.core.calibration`` import stays
    lazy inside this body: it must not load at module import time (S01
    laziness gate).
    """
    # ── What falls on the plate ring is not sample ──
    # The ring is a bright, continuous band, and the detector confuses it with
    # material. Since it also comes out huge, it entered among the largest: in one
    # report it occupied an example card as `PET fibra` of 5978 um, which is
    # more than the 5 mm limit by which a microplastic is defined.
    #
    # The fitted circle was already known -- the scale comes from it -- and it only needed
    # to be asked. It is removed from the SIZE and from the cards, not from the count:
    # removing detections from the count is a study decision and not the
    # report generator's. What is done is to declare how many were set apart.
    from polyx.core.measure.calibration import about_the_ring

    _cals = getattr(state, "calibrations", None) or {}

    def _en_el_aro(det, path_2) -> bool:
        return about_the_ring(_cals.get(Path(path_2).name), det.x1, det.y1, det.x2, det.y2)

    # Each particle with its source image, so it can be found later.
    _shaped = [
        (p_, r.image_path)
        for r in all_results
        for p_ in r.predictions
        if getattr(p_, "aspect", None)
    ]
    shapes = [(d, path_2) for d, path_2 in _shaped if not _en_el_aro(d, path_2)]
    n_en_el_aro = len(_shaped) - len(shapes)
    n_formless = sum(
        1 for r in all_results for p_ in r.predictions if getattr(p_, "aspect", None) is None
    )
    return shapes, n_en_el_aro, n_formless


def _size_per_folder(with_size):
    """Body of the 'talla_carpetas' section: size by folder, then by photo.

    Returns the HTML body WITHOUT its ``<h2>`` title, with the ``{NUM}.N``
    sub-header placeholders intact. Empty when fewer than two folders hold
    measured particles, so the assembler drops the section like it does
    today.
    """
    # ── Size by folder (`{tr('Talla por carpeta')}`), and by photo within each folder ──
    # Generalizes the `by station/stretch` contrast from the paper
    # pipeline (which depended on a filename pattern specific
    # to one study) to the only thing ANY batch has: which
    # folder each photo came from. It is useful for contrasting sampling sites,
    # treatments, or any other grouping that the user has
    # expressed as folders when choosing the images.
    by_html_folder = ""
    groups_folder: dict[str, list[float]] = {}
    for d, path_2 in with_size:
        folder = Path(path_2).parent.name or str(Path(path_2).parent)
        groups_folder.setdefault(folder, []).append(d.length_um)

    if len(groups_folder) >= 2:
        fig_folder = _fig_boxplot_groups(
            groups_folder, tr("Talla por carpeta"), tr("carpeta de origen")
        )
        verdict_folder = _text_kruskal(
            _kruskal_wallis(list(groups_folder.values())), len(groups_folder)
        )
        by_html_folder = (
            f"<h3>{NUM}.1 {tr('Talla por carpeta')}</h3>"
            + tr(
                "<p>Compara la talla entre las carpetas del lote analizado: "
                "cada carpeta como un grupo distinto. Útil cuando cada carpeta "
                "es un sitio de muestreo, una estación o una condición.</p>"
            )
            + f"<img src='data:image/png;base64,{fig_folder}' style='max-width:100%'>"
            f"<p>{verdict_folder}</p>"
        )

        # -- Per photo, within folders with more than one image --
        parts_photo = []
        for folder in sorted(groups_folder, key=lambda c: -len(groups_folder[c])):
            if len(parts_photo) >= _MAX_FOLDER_PER_PHOTO:
                break
            folder_photos: dict[str, list[float]] = {}
            for d, path_2 in with_size:
                if Path(path_2).parent.name == folder:
                    folder_photos.setdefault(Path(path_2).name, []).append(d.length_um)
            if len(folder_photos) < 2:
                continue
            fig_photo = _fig_boxplot_groups(
                folder_photos,
                tr("Talla por foto — {carpeta}").format(carpeta=folder),
                tr("foto"),
                max_groups=20,
            )
            verdict_photo = _text_kruskal(
                _kruskal_wallis(list(folder_photos.values())), len(folder_photos)
            )
            parts_photo.append(
                f"<p><strong>{_esc(folder)}</strong> "
                + tr("({n} fotos)").format(n=len(folder_photos))
                + "</p>"
                f"<img src='data:image/png;base64,{fig_photo}' style='max-width:100%'>"
                f"<p>{verdict_photo}</p>"
            )
        if parts_photo:
            n_folders_with_photos = sum(
                1
                for c in groups_folder
                if len({Path(r).name for d, r in with_size if Path(r).parent.name == c}) >= 2
            )
            note_tope = (
                tr(
                    " Se muestran las {n} carpetas con más partículas "
                    "de las {total} que tienen más de una foto."
                ).format(n=_MAX_FOLDER_PER_PHOTO, total=n_folders_with_photos)
                if n_folders_with_photos > _MAX_FOLDER_PER_PHOTO
                else ""
            )
            by_html_folder += (
                f"<h3>{NUM}.2 {tr('Talla por foto, dentro de cada carpeta')}</h3>"
                + tr(
                    "<p>Compara la talla entre las fotos individuales de una "
                    "misma carpeta: por ejemplo, si el tamaño cambia con la "
                    "profundidad o el momento de muestreo dentro de un mismo "
                    "sitio.{nota}</p>"
                ).format(nota=note_tope)
                + "".join(parts_photo)
            )
    return by_html_folder


def _section_shape(state, shapes, n_en_el_aro, n_formless):
    """Body of the 'forma' section, plus the companion 'talla_carpetas' body.

    Returns ``(forma_html, por_carpeta_html)``, both WITHOUT their ``<h2>``
    titles (the assembler owns them; the ``{NUM}.N`` sub-header placeholders
    stay intact). ``por_carpeta_html`` is built in the same pass because it
    shares the per-class size grouping computed here, but it is its own
    optional section in the report (see SECCIONES) and never lands inside
    ``forma_html``. ``formas`` arrives precomputed from
    :func:`_formas_de_lote` because the depth profile and the particle
    fichas read the same list.
    """
    record_elderly = None
    record_minor = None
    table_morphotype = tables = count_2 = method_2 = example = ""
    # A batch without measured particles has no forma body: with this
    # initialization the section stays empty and the assembler drops it,
    # instead of crashing with UnboundLocalError in the return below. Same
    # meaning as the por_carpeta_html one: it is its own optional section.
    shape_html = ""
    # Independent of `forma`: it is its own optional section in the report
    # (see SECCIONES below), so it does not enter forma_html.
    by_html_folder = ""

    if shapes:
        with_size = [(d, path_2) for d, path_2 in shapes if d.length_um]
        fibers = sum(1 for d, _ in shapes if d.morphotype == "fibra")
        curves = sum(1 for d, _ in shapes if d.curvature and d.curvature >= 1.15)
        pc_f = 100.0 * fibers / len(shapes)

        # ── Particles whose size warrants a look ──
        # It is declared in the section, not only in each card: whoever reads the size
        # figures has to know how many of them relied on a mask
        # that had to be cropped. It affects above all the large tail of the
        # distribution, which is precisely the one reported as `the largest
        # particle`.
        #
        # WATCH THE DIRECTION OF THE BIAS: while the mask was not cropped, the
        # leak inflated the size and the bias went UP. Now the crop
        # bounds it, so what is reported is the piece inside the box and the
        # bias goes DOWN. The text below says it in that direction.
        n_review = sum(1 for d, _ in shapes if getattr(d, "review", False))
        warning_review = ""
        if n_review:
            # One decimal below 1%: with `:.0f` a proportion of 4 out of
            # 909 came out as `(0 %)` next to `4 particulas`, which reads as if
            # the figure were broken.
            _pc = 100.0 * n_review / len(shapes)
            _pc_txt = f"{_pc:.1f}" if _pc < 1 else f"{_pc:.0f}"
            warning_review = (
                f"<p style='border-left:3px solid {T.DOC.WARN};padding:6px 12px;"
                f"background:#fff8e6'>"
                + tr(
                    "<strong>{n} de {total} partículas ({pct}&nbsp;%) tienen una talla "
                    "que pide comprobación</strong>: al segmentarlas, la máscara se "
                    "salía bastante más allá de la caja del detector, casi siempre "
                    "porque el umbral enganchó una partícula vecina o una zona "
                    "brillante del fondo. En esos casos la máscara se <em>recorta a la "
                    "caja</em>, de modo que la talla reportada no está inflada; lo que "
                    "puede es quedarse <em>corta</em>, si la partícula seguía de verdad "
                    "más allá de la caja. Van marcadas con ⚠ en sus fichas."
                ).format(n=n_review, total=len(shapes), pct=_pc_txt)
                + "</p>"
            )

        # ── What was set apart for falling on the ring ──
        if n_en_el_aro:
            from polyx.core.measure.calibration import FRACTION_AREA_USEFUL

            warning_review += (
                f"<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;"
                f"background:var(--bg_soft)'>"
                + tr(
                    "<strong>{n} detección(es) caen sobre el anillo de la placa</strong> "
                    "—más allá del {pct} % del radio ajustado— y quedan fuera de esta "
                    "sección: el aro es una banda brillante que el detector confunde "
                    "con material, y como sale enorme desplazaría a las partículas de "
                    "verdad del extremo grande. Siguen contadas en la sección de "
                    "conteo."
                ).format(n=n_en_el_aro, pct=f"{100 * FRACTION_AREA_USEFUL:.0f}")
                + "</p>"
            )

        # ── Distribution by morphotype ──
        table_morphotype = (
            warning_review
            + f"""
<table class='data'><tr>{tr("<th>Morfotipo</th><th>Cuántas</th><th>%</th>")}</tr>
<tr>{tr("<td>Fibra (relación de aspecto ≥ 3)</td>")}<td>{fibers}</td><td>{pc_f:.1f} %</td></tr>
<tr>{tr("<td>Partícula (no fibrosa)</td>")}<td>{len(shapes) - fibers}</td><td>{100 - pc_f:.1f} %</td></tr>
</table>"""
        )

        if with_size:
            lengths = np.array([d.length_um for d, _ in with_size], dtype=float)
            elderly, path_elderly = max(with_size, key=lambda t: t[0].length_um)
            minor_2, path_minor = min(with_size, key=lambda t: t[0].length_um)

            def _num(v, dec=0):
                """Cell with the value, or a dash when that magnitude was not measured."""
                return f"{v:.{dec}f}" if v else "—"

            def _row(label_3, d, path_2):
                return (
                    f"<tr><td><strong>{label_3}</strong></td><td>{_esc(d.class_name)}</td>"
                    f"<td>{_num(d.length_um)}</td><td>{_num(d.width_um)}</td>"
                    f"<td>{_num(d.area_um2)}</td><td>{_num(d.aspect, 1)}</td>"
                    f"<td>{_num(d.curvature, 2)}</td><td>{_morphotype(d.morphotype)}</td>"
                    f"<td style='font-size:9pt'>{_esc(Path(path_2).name)}</td></tr>"
                )

            tables = f"""
<h3>{NUM}.4 {tr("Partícula mayor y menor")}</h3>
<table class='data'>
<tr>{tr("<th></th><th>Clase</th><th>Largo</th><th>Ancho</th><th>Área</th><th>Aspecto</th><th>Curvatura</th><th>Morfotipo</th><th>Imagen</th>")}</tr>
{_row(tr("Mayor"), elderly, path_elderly)}
{_row(tr("Menor"), minor_2, path_minor)}
</table>

<h3>{NUM}.5 {tr("Distribución de tallas")}</h3>
<table class='data'>
<tr>{tr("<th>Estadístico</th><th>Largo (µm)</th>")}</tr>
<tr>{tr("<td>mínimo</td>")}<td>{lengths.min():.0f}</td></tr>
<tr>{tr("<td>percentil 10</td>")}<td>{np.percentile(lengths, 10):.0f}</td></tr>
<tr>{tr("<td>mediana</td>")}<td>{np.median(lengths):.0f}</td></tr>
<tr>{tr("<td>percentil 90</td>")}<td>{np.percentile(lengths, 90):.0f}</td></tr>
<tr>{tr("<td>máximo</td>")}<td>{lengths.max():.0f}</td></tr>
</table>"""

            # ── Sizes by stretch, stacked by class ──
            per_class_2: dict[str, list[float]] = {}
            for d, _path in with_size:
                per_class_2.setdefault(d.class_name, []).append(d.length_um)
            fig = _fig_sizes_by_section(per_class_2)
            if fig:
                tables += f"<img src='data:image/png;base64,{fig}' style='max-width:100%'>"

            # ── Table by class, with its largest ──
            rows_cls = ""
            for cls in _ORDER_CLASSES + [c for c in per_class_2 if c not in _ORDER_CLASSES]:
                v = sorted(per_class_2.get(cls, []))
                if not v:
                    continue
                nf = sum(
                    1 for d, _r in with_size if d.class_name == cls and d.morphotype == "fibra"
                )
                rows_cls += (
                    f"<tr><td>{cls}</td><td>{len(v)}</td>"
                    f"<td>{v[0]:.0f}</td><td>{v[len(v) // 2]:.0f}</td>"
                    f"<td>{v[-1]:.0f}</td><td>{nf}</td></tr>"
                )
            if rows_cls:
                tables += (
                    f"<h3>{NUM}.6 {tr('Talla por tipo de plástico')}</h3>"
                    f"<table class='data'><tr><th>Class</th><th>n</th>"
                    f"<th>smallest<br>(µm)</th><th>median<br>(µm)</th>"
                    f"<th>largest<br>(µm)</th><th>Fibers</th></tr>{rows_cls}</table>"
                )

            # The example card is drawn further below, AFTER explaining the
            # method: showing the crop with a yellow line on top before
            # saying that this line is the Feret diameter explains nothing.
            cal_elderly = (getattr(state, "calibrations", {}) or {}).get(Path(path_elderly).name)
            um_elderly = cal_elderly.um_por_px if cal_elderly and cal_elderly.valid else None
            record_elderly = (elderly, path_elderly, um_elderly)

            # The smallest one is shown too, not only tabulated: a minimum size
            # without a photo cannot be distinguished from noise, or the shape card
            # and a reader has no way to check that there really is a
            # particle there and not a background speck that was badly segmented.
            cal_minor = (getattr(state, "calibrations", {}) or {}).get(Path(path_minor).name)
            um_minor = cal_minor.um_por_px if cal_minor and cal_minor.valid else None
            record_minor = (minor_2, path_minor, um_minor)
            by_html_folder = _size_per_folder(with_size)

        # ── Fibre/fragment count per photo ──
        # It goes inside the forma section: it answers `how many fibres and how many
        # fragments are there on each plate`, which is the question asked of the
        # table when reviewing photo by photo.
        por_img: dict[str, list] = {}
        for d, path_2 in shapes:
            por_img.setdefault(Path(path_2).name, []).append(d)
        rows_img = ""
        for name_2 in sorted(por_img):
            ds = por_img[name_2]
            nf = sum(1 for d in ds if d.morphotype == "fibra")
            lengths_i = [d.length_um for d in ds if d.length_um]
            rows_img += (
                f"<tr><td style='font-size:9pt'>{_esc(name_2)}</td><td>{len(ds)}</td>"
                f"<td>{nf}</td><td>{len(ds) - nf}</td>"
                f"<td>{(f'{np.median(lengths_i):.0f}' if lengths_i else '—')}</td>"
                f"<td>{(f'{max(lengths_i):.0f}' if lengths_i else '—')}</td></tr>"
            )
        # The first column is renamed `Total` and not `Partículas`: with the
        # nomenclature change there would be two columns with the same name, one
        # as a sum and another as a morphotype. `Total = Fibras + Partículas`.
        count_2 = (
            f"<h3>{NUM}.7 {tr('Recuento por imagen')}</h3>"
            f"<table class='data'><tr>"
            f"{tr('<th>Imagen</th><th>Total</th><th>Fibras</th><th>Partículas</th>')}"
            f"{tr('<th>Largo mediano<br>(µm)</th><th>Mayor<br>(µm)</th>')}"
            f"</tr>{rows_img}</table>"
        )

        # An `—` in the size columns is not an analysis failure: the
        # particle WAS measured, in pixels, but without µm/px there is no way to
        # turn it into micrometres. Before, the dash appeared without explanation and
        # it looked as if the measurement had failed. It says what is missing and how
        # to fix it, instead of inventing a number or saying nothing.
        no_scale = [n for n, ds in por_img.items() if not any(d.length_um for d in ds)]
        if no_scale:
            all_value = len(no_scale) == len(por_img)
            which_2 = (
                tr("<strong>Ninguna imagen tiene escala</strong>")
                if all_value
                else tr(
                    "<strong>{n} de {total} imágenes no tienen escala</strong> ({cuales})"
                ).format(
                    n=len(no_scale),
                    total=len(por_img),
                    cuales=", ".join(sorted(no_scale)[:6]) + (", …" if len(no_scale) > 6 else ""),
                )
            )
            count_2 += (
                "<p>"
                + which_2
                + tr(
                    ", y por eso su largo aparece como «—». Las partículas <em>sí</em> "
                    "se midieron —el conteo y la forma de la tabla son correctos—, pero "
                    "una medida en píxeles no se puede pasar a micrómetros sin saber "
                    "cuántos µm mide cada píxel.</p>"
                )
                + tr(
                    "<p>Para obtenerlo, en <em>Parámetros</em>: activa <strong>«Medir "
                    "la placa Petri»</strong> —cada foto obtiene su propia escala del "
                    "anillo de la placa, que es lo más fiable— o escribe un valor de "
                    "<strong>µm/píxel</strong>"
                )
                + (
                    "."
                    if all_value
                    else tr(
                        ". Si ya está activo, en esas fotos concretas no se encontró el "
                        "anillo de la placa: revisa que se vea entero en el encuadre."
                    )
                )
                + "</p>"
            )

        # Outside the f-string: inside it we would have to nest quotes of the same type.
        pct_curves = f"{100.0 * curves / len(shapes):.1f}"
        method_2 = f"""
<h3>{NUM}.1 {tr("Cómo se mide el largo")}</h3>
{tr("<p>Las magnitudes se miden sobre la <strong>máscara de cada partícula</strong>, no sobre la caja del detector. La caja de una partícula alargada está casi vacía y depende de cómo haya caído: una fibra tumbada en diagonal tiene caja cuadrada, de modo que medir sobre la caja la reportaría como si no fuera fibra, y con una talla equivocada. La máscara se obtiene umbralizando cada recorte a medio camino entre el nivel del fondo —la mediana del anillo que rodea la caja— y el de la partícula —el percentil 90 dentro de ella—.</p>")}

{tr("<p><strong>El criterio general es la línea recta más larga que cabe en la partícula</strong>, es decir la mayor distancia entre dos puntos de su contorno: el <em>diámetro de Feret máximo</em>. Para una partícula de forma irregular pero no doblada, esa recta es su talla, y tiene dos propiedades que la hacen fiable: no depende de la orientación con que la partícula haya caído, y un borde dentado no la altera.</p>")}

{tr("<p><strong>Esa recta deja de servir cuando la partícula está contorsionada.</strong> En una fibra doblada la distancia entre sus extremos es la cuerda, no su longitud: en un arco de media circunferencia se queda un 35&nbsp;% corta. Para esos casos se mide el <em>diámetro geodésico</em>, que es el camino más largo que cabe <strong>dentro</strong> de la partícula. Al no poder salirse de la máscara, ese camino rodea la curva y devuelve la longitud recorrida.</p>")}

<table class='data'>
<tr>{tr("<th>Forma de la partícula</th><th>Qué se reporta como largo</th>")}</tr>
<tr>{tr("<td>Compacta o irregular, pero no doblada</td><td>Feret máximo (la recta más larga)</td>")}</tr>
<tr>{tr("<td>Alargada y contorsionada (fibra)</td><td>Diámetro geodésico (sigue la curva)</td>")}</tr>
</table>

{tr("<p>La distinción se hace por el <strong>grosor</strong>: solo se usa el geodésico si el largo supera al menos cuatro veces el grosor máximo inscrito de la partícula. La razón es que en una partícula gruesa cualquier concavidad obliga al camino geodésico a bordearla en vez de atravesarla, y entonces el número se infla; en una fibra delgada eso no puede ocurrir.</p>")}

{tr("<p>Contrastado con formas de talla conocida —rectas, rectas giradas 45&nbsp;°, arcos de 60, 120 y 180&nbsp;°, un círculo y una recta de borde dentado— el largo así medido da un <strong>error mediano del 0,8&nbsp;% y del 4,7&nbsp;% en el peor caso</strong>.</p>")}

{tr("<p><strong>{n} partículas ({pct} %) están contorsionadas</strong>, entendiendo por tal que su largo supera en más de un 15&nbsp;% su extensión en línea recta.</p>").format(n=curves, pct=pct_curves)}

{tr("<p><em>Partículas en contacto.</em> Dos partículas que se tocan forman una sola mancha, y medirlas juntas sumaría sus tallas. Se separan por <em>watershed</em> sobre la transformada de distancia: el centro de cada una queda lejos del fondo y el cuello que las une queda cerca, de modo que el corte cae por el cuello. Sobre círculos de talla conocida las separa hasta un 27&nbsp;% de solapamiento del diámetro, sin partir ninguna partícula de una sola pieza.</p>")}

{tr("<p><em>Limitaciones conocidas.</em> Dos partículas solapadas más allá de un 40&nbsp;% de su diámetro se siguen midiendo como una sola: a esa altura ya no hay un cuello por el que cortar. Y en una fibra muy enroscada el camino geodésico ataja por el interior de cada codo, subestimando la longitud hasta un 19&nbsp;% en el caso más cerrado que se ensayó.</p>")}"""
        if n_formless:
            method_2 += tr(
                "<p>En {n} partículas no se pudo separar la partícula del fondo; su "
                "talla proviene de la caja y no es comparable con el resto.</p>"
            ).format(n=n_formless)

        # ── The examples, once the method is explained ──
        # Both, not only the largest: a minimum size without a photo cannot be
        # distinguished by eye from badly segmented noise, and it is the one most easily
        # questioned in a review.
        if record_elderly is not None:
            example = _record_particle(
                *record_elderly, f"2 {tr('Un ejemplo: la partícula mayor, medida')}"
            )
        if record_minor is not None:
            example += _record_particle(
                *record_minor, f"3 {tr('Un ejemplo: la partícula menor, medida')}"
            )

        # Section order: first how it is measured, then a measured case, and
        # finally the figures. The other way round, the tables arrive before anything has
        # explained what `largo` means here.
        shape_html = method_2 + example + table_morphotype + tables + count_2
    return shape_html, by_html_folder


def _section_records(state, shapes: list[tuple], max_records: int = 12) -> str:
    """Body of the 'fichas' section: one annotated card per sampled particle.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns the
    heading and the section number. Empty when there are no measured particles
    or the cap is zero. The ``cv2`` / ``morfologia`` / ``calibracion`` imports
    stay lazy inside this body, exactly as lazy as when the block was inline
    (S01 fresh-interpreter laziness gate).
    """
    # One entry per particle, with its number, its crop and the measurement drawn
    # on top. That is what makes it possible to check particle by particle that the size
    # reported corresponds to what is in the photo, instead of having to
    # trust the table.
    records_html = ""
    if shapes and max_records:
        import cv2

        from polyx.core.measure.morphology import draw_measurement

        chosen = choose_records(shapes, max_records)
        n_fibers_sample = sum(1 for d, _ in chosen if d.morphotype == "fibra")
        por_img_f: dict[str, list] = {}
        for d, path_2 in chosen:
            por_img_f.setdefault(str(path_2), []).append(d)
        parts, made = [], 0
        for path_2 in sorted(por_img_f):
            bgr = None
            try:
                from polyx.core.measure.calibration import read_image

                bgr = read_image(path_2)
            except Exception:  # noqa: BLE001
                bgr = None
            if bgr is None:
                continue
            cal = (getattr(state, "calibrations", {}) or {}).get(Path(path_2).name)
            um = cal.um_por_px if cal and cal.valid else None
            cards = ""
            for d in sorted(por_img_f[path_2], key=lambda z: z.number or 0):
                if made >= max_records:
                    break
                img, m = draw_measurement(bgr, d)
                if img is None or m is None or not m.ok:
                    continue
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    continue
                made += 1
                account = ""
                if d.length_um and um:
                    account = (
                        f"{m.length_px:.1f} px &times; {um:.4f} = "
                        f"<strong>{d.length_um:.0f} µm</strong>"
                    )
                # d.aspecto/morfotipo are filled by segmentation alone (they do not need
                # calibration), but d.ancho_um only if the photo had a valid um_por_px
                # when it was measured -- a photo with no detected plate leaves
                # aspecto with a value and ancho_um at None. Without this guard, that
                # combination crashes the `:.0f` format on None.
                detail = [f"ancho {d.width_um:.0f} µm"] if d.width_um else []
                if d.aspect:
                    detail.append(f"aspecto {d.aspect:.1f}")
                detail.append(tr("medido por {metodo}").format(metodo=tr(m.method)))
                # If an appreciable part of the line falls outside, it is stated with a
                # number. Below 5% it is contour discretization and
                # saying it would only add noise to the card.
                if getattr(m, "feret_outside", 0) > 0.05:
                    detail.append(
                        tr("{pct} % de la recta fuera del contorno").format(
                            pct=f"{100 * m.feret_outside:.0f}"
                        )
                    )
                color = "#1f6b5e" if d.morphotype == "fibra" else "#656d76"
                label_morph = _morphotype(d.morphotype, lowercase=True)
                # The warning is SHOWN. Before, it was computed and thrown away: a size
                # inflated by several particles in contact reached the report
                # as a clean number, and whoever read it had no way to know.
                # It is read from the Detection, NOT from the Morfologia returned by
                # dibujar_medicion: that one comes from medir(), and the warning is set by
                # medir_deteccion() when measuring against the box diagonal. In
                # the Morfologia here the field is always False, so
                # looking at it meant the `⚠` never appeared.
                alerta = ""
                if getattr(d, "review", False):
                    alerta = (
                        f"<div style='font-size:8.5pt;color:{T.DOC.WARN};margin-top:3px'>"
                        f"⚠ {d.warning_shape or tr('la máscara se salía de la caja y se recortó a ella')}</div>"
                    )
                cards += (
                    f"<div style='display:inline-block;vertical-align:top;margin:0 14px 18px 0;"
                    f"max-width:290px'>"
                    f"<div style='font-weight:700'>#{d.number} · {_esc(d.class_name)} · "
                    f"<span style='color:{color}'>{label_morph}</span></div>"
                    f"<img src='data:image/png;base64,{_img_b64(buf.tobytes())}' style='max-width:100%'>"
                    f"<div style='font-size:9pt;color:#656d76'>{account}<br>"
                    f"{' · '.join(detail)}</div>{alerta}</div>"
                )
            if cards:
                parts.append(f"<h3>{_esc(Path(path_2).name)}</h3>{cards}")
        if parts:
            # It is declared HOW the sample was chosen. Without this, a reader could
            # take it as representative of the batch and it is not: it is deliberately biased
            # towards the large ones and towards fibres.
            n_others_sample = made - n_fibers_sample
            if n_fibers_sample:
                criterion = tr(
                    "Se muestran las <strong>{fibras} fibra(s) y las {otras} "
                    "partículas más grandes</strong>"
                ).format(fibras=n_fibers_sample, otras=n_others_sample)
            else:
                criterion = tr(
                    "Se muestran las <strong>{otras} partículas más "
                    "grandes</strong>. En este lote no se detectó ninguna "
                    "fibra, así que la muestra no incluye ninguna"
                ).format(otras=n_others_sample)
            records_html = (
                tr(
                    "<p>Cada partícula con el número que lleva en la imagen anotada, su "
                    "recorte y la medida dibujada <strong>sobre ella</strong>: en "
                    "amarillo la recta de Feret, en magenta el camino geodésico, según "
                    "cuál de las dos haya decidido su talla. El contorno verde es la "
                    "máscara que se midió. A la izquierda va la partícula sin marcas, "
                    "para poder juzgar si el contorno la sigue.</p>"
                )
                + tr(
                    "<p>En una partícula <strong>cóncava</strong> la recta amarilla "
                    "cruza por fuera del contorno, y eso es correcto: Feret es la "
                    "separación de dos mordazas de calibre, no un camino por dentro "
                    "de la partícula. Para que se vea, el tramo que cae fuera va "
                    "<strong>a trazos</strong>. El camino que sí va por dentro es el "
                    "geodésico, en magenta, y se usa cuando la partícula es delgada "
                    "y está doblada.</p>"
                )
                + "<p>"
                + criterion
                + tr(
                    ". El reparto es deliberado: las fibras son el caso donde actúa el "
                    "método geodésico —y son minoría, de modo que una muestra tomada al "
                    "azar podría no enseñar ninguna—, y la mayor de cada tipo es la que "
                    "sostiene cualquier afirmación sobre talla máxima. <strong>No es "
                    "una muestra representativa del lote</strong>, sino la selección "
                    "que permite comprobar el método donde más puede fallar.</p>"
                )
                + "".join(parts)
            )
    return records_html


def _batch_metrics(state, solo_images):
    """Compute the batch-wide inputs every section builder consumes.

    Verbatim hoist (S03/T01) of the precomputation block that used to live
    inline in ``_generate_report_impl``: same empty-set filter semantics,
    same computation order, same ``Counter`` loops. Returns a plain tuple —
    no :class:`~typing.NamedTuple` and no new imports, so the lazy-import
    gate (``test_importes_diferidos_informe.py``) sees nothing new and no
    heavy module (``ultralytics``/``cv2``/``morfologia``/``calibracion``)
    gets touched at module level.
    """
    if solo_images is None:
        results_2 = dict(state.results)
    else:
        chosen = {Path(p) for p in solo_images}
        results_2 = {
            mi: [r for r in rs if Path(r.image_path) in chosen] for mi, rs in state.results.items()
        }

    all_results = [r for rs in results_2.values() for r in rs]
    total_imgs = len({r.image_path for r in all_results})
    total_dets = sum(len(r.predictions) for r in all_results)
    confs = [p.conf for r in all_results for p in r.predictions]
    sizes = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
    avg_conf = sum(confs) / len(confs) if confs else 0
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    any_gt = any(r.has_gt for r in all_results)

    classes_batch = _classes_views(all_results)

    # ── Distribution by class ──
    per_class = Counter()
    for r in all_results:
        for p in r.predictions:
            per_class[p.class_name] += 1

    return (
        results_2,
        all_results,
        total_imgs,
        total_dets,
        confs,
        sizes,
        avg_conf,
        avg_size,
        any_gt,
        classes_batch,
        per_class,
    )


# Legacy anchor names, to avoid breaking links to reports already
# distributed. A new section does NOT have to appear here: its own id already
# serves as an anchor. This dictionary was a second list to keep in
# sync with SECCIONES, and adding `profundidad` crashed with
# KeyError -- exactly what SECCIONES avoids by being a single list.
_ANCHORS = {
    "resumen": "abstract",
    "metodos": "methods",
    "calibracion": "calib",
    "resultados": "results",
    "modelos": "models",
    "errores": "errors",
    "comparacion": "compare",
    "galeria": "gallery",
    "referencias": "refs",
}


def anchor(sid: str) -> str:
    """Anchor name for a section id: legacy alias if any, the id itself otherwise."""
    return _ANCHORS.get(sid, sid)


def _document_html(alive, active) -> str:
    """Assemble the final document: cover header, TOC, numbered sections, footer.

    ``vivas`` arrives as (id, translated title, body) tuples in SECCIONES
    order, already filtered to the requested sections that have content.
    The enumeration (SECCIONES order, start=1), the ``<h2>`` titles and the
    ``{NUM}`` placeholder substitution all live here because this helper is
    the only place that knows the final order; moving or reordering them is
    byte-visible in the frozen render (MEM090).
    """
    toc_html = "".join(f"<li><a href='#{anchor(sid)}'>{title}</a></li>" for sid, title, _ in alive)
    sections_html = "".join(
        f"<h2 id='{anchor(sid)}'>{i}. {title}</h2>\n{body.replace(NUM, str(i))}\n"
        for i, (sid, title, body) in enumerate(alive, start=1)
    )

    return f"""<!doctype html>
<html lang='{language()}'><head><meta charset='utf-8'>
<title>{tr("Informe de detección")} · Poly-X</title>
<style>{REPORT_CSS}</style></head><body>
<div class='container'>

<header class='cover'>
  <div class='kicker'>Poly-X · {tr("Informe de detección")}</div>
  <h1>{tr("Informe de detección de microplásticos")}<br>{tr("por fluorescencia Nile Red")}</h1>
  <p class='meta'><strong>{tr("Autor:")}</strong> Cristofher Ferrada &middot;
    <strong>{tr("Modelos:")}</strong> {_esc(", ".join(s.alias for s in active)) or "—"}</p>
</header>

<div class='toc'>
  <h3>📑 {tr("Índice")}</h3>
  <ol>{toc_html}</ol>
</div>

{sections_html}
<footer>
  © Cristofher Ferrada · {tr("Generado por Poly-X")}<br>
  {tr("Suite de detección de microplásticos por fluorescencia Nile Red (254 nm) e IA (YOLO v8/v11).")}
</footer>
</div></body></html>
"""


def _generate_report_impl(
    state,
    output_path: Path,
    include_refs: bool = True,
    include_gallery: bool = True,
    max_gallery: int = 60,
    max_records: int = 12,
    solo_images=None,
    sections=None,
) -> Path:
    """Generates the HTML report. `state` is a ``DetectorState`` with results.

    ``max_gallery`` limits how many images are embedded in the gallery.
    Images go into the HTML itself in base64, so without a cap a large
    batch produced an unmanageable file (14.5 MB with just 4 images).

    ``solo_imagenes`` narrows the report down to a set of paths. With
    ``None`` the whole batch is in. The filter is applied before anything
    is computed, so that the totals, the charts and the confusion matrix
    describe exactly the same subset being shown; if only the gallery
    were filtered, the figures above would talk about one batch and the
    images about another.

    ``secciones`` is the set of :data:`SECCIONES` identifiers to include;
    ``None`` includes all of them. A requested section that has no
    content — errors without ground truth, for example — is skipped
    anyway: ticking it does not invent data that does not exist.
    """
    requested = set(IDS_SECTIONS if sections is None else sections)
    # The two old checkboxes still take precedence if passed as False, so as not to
    # change the meaning of calls that already existed.
    if not include_refs:
        requested.discard("referencias")
    if not include_gallery:
        requested.discard("galeria")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")  # noqa: F841
    active = state.active_models()

    # ── Batch metrics ──
    # The precomputation block now lives in _metricas_de_lote; the impl only
    # unpacks. The tuple order is the order in which the assembler
    # consumes them, so the unpacking names the same variables as
    # always.
    (
        results_2,
        all_results,
        total_imgs,
        total_dets,
        confs,
        sizes,
        avg_conf,
        avg_size,
        any_gt,
        classes_batch,
        per_class,
    ) = _batch_metrics(state, solo_images)

    # ── Error analysis: matrix + per-class metrics (main model) ──
    err_section = _section_errors(state, results_2, active, any_gt)

    # ── Verdict and confidence sweep ──
    verdict_html = _section_verdict(results_2, state)

    # ── Models side by side, photo by photo ──
    compare_html = _section_comparison(state, results_2)

    # ── Side-by-side gallery: Prediction vs Ground Truth ──
    gallery_html = _section_gallery(
        results_2, state, classes_batch, max_gallery=max_gallery, include_gallery=include_gallery
    )

    # ── Aggregate figures ──
    figures_html = _section_results(per_class, confs, sizes)

    # ── Methods ──
    methods_html, team_html, methods_for = _section_methods(
        state, active, all_results, total_imgs, total_dets, any_gt
    )

    count_html = _section_count(results_2, state, active, classes_batch)

    refs_html = _section_references(include_refs)

    # ── Scale calibration ──
    calib_html = _section_calibration(state)

    # ── Shape and size ──
    # It is separated from `Resultados` because it answers another question: not how many
    # particles there are, but what shape they have. Morphotype and curvature are
    # variables that the microplastics literature reports and that the detector
    # box cannot give. `formas` is computed once because the depth profile
    # and the cards also consume it.
    shapes, n_en_el_aro, n_formless = _lot_forms(state, all_results)
    # The depth profile only exists if the names carry the stretch.
    depth_html = section_depth(shapes, all_results) if shapes else ""
    shape_html, by_html_folder = _section_shape(state, shapes, n_en_el_aro, n_formless)
    # ── Card for each particle ──
    records_html = _section_records(state, shapes, max_records)

    # ── Assemble HTML ──
    # Each block is (id, body). The body does NOT carry its <h2>: the title and
    # number are set by the assembler, which is the only one that knows which sections
    # survived the filter and therefore which number belongs to each one.
    bodies = {
        "resumen": _section_summary(total_imgs, total_dets, avg_conf, avg_size, active, any_gt),
        "metodos": f"""
<table class='data'><tr>{tr("<th>Parámetro</th><th>Valor</th>")}</tr>{methods_html}</table>
{team_html}
{methods_for}""",
        "calibracion": calib_html,
        "forma": shape_html,
        "talla_carpetas": by_html_folder,
        "profundidad": depth_html,
        "fichas": records_html,
        "resultados": figures_html,
        "modelos": _section_models(state, results_2, all_results, any_gt),
        "errores": err_section,
        "comparacion": f"{verdict_html}\n{compare_html}",
        "galeria": gallery_html,
        "conteo": count_html,
        "referencias": refs_html,
    }

    # An id in ``pedidas`` with an empty body is dropped here: the checkbox was ticked but
    # there was nothing to fill it with (errors without ground truth, count without samples).
    # tr() on the title: SECCIONES stores the Spanish version, which is the
    # dictionary key. The interface already translated it for its checkboxes; the
    # document did not, and came out with the titles in Spanish even though the rest did not.
    alive = [
        (sid, tr(title), bodies.get(sid, ""))
        for sid, title in SECTIONS
        if sid in requested and (bodies.get(sid) or "").strip()
    ]

    # Cover, table of contents, section numbering and footer live in _documento_html:
    # the impl remains the ordered list of calls that feeds cuerpos.
    html = _document_html(alive, active)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_report(
    state,
    output_path: Path,
    include_refs: bool = True,
    include_gallery: bool = True,
    max_gallery: int = 60,
    max_records: int = 12,
    solo_images=None,
    sections=None,
) -> Path:
    """Generates the HTML report with an observability wrapper.

    Generates the HTML report, measuring it and recording report.generate /
    report.finish / report.error in the detector JSONL. Re-throws failures:
    the caller decides what to do, the log keeps the diagnosis.
    """
    from polyx.core.observability import logger_for, timer

    n_results = sum(len(rs) for rs in (state.results or {}).values())
    with timer(
        logger_for("detector"),
        "report",
        results_2=n_results,
        output=str(output_path),
        sections=sorted(sections) if sections else None,
        solo_images=(len(solo_images) if solo_images is not None else None),
    ):
        return _generate_report_impl(
            state,
            output_path,
            include_refs=include_refs,
            include_gallery=include_gallery,
            max_gallery=max_gallery,
            max_records=max_records,
            solo_images=solo_images,
            sections=sections,
        )
