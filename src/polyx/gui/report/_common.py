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
import platform
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# matplotlib in non-interactive mode
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from polyx.core.i18n import language, tr
from polyx.core.quality.assignment import NO_ASSIGNABLE, class_id_reported, resume
from polyx.core.quality.metrics import aggregate_per_class, confusion_matrix

from polyx.gui import theme as T

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


_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


_MAX_GROUPS_BOXPLOT = 12
_MAX_FOLDER_PER_PHOTO = 6


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


_NAME_MORPHOTYPE = {"fibra": "Fibra", "fragmento": "Partícula"}


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


# Late imports: figures and sections use helpers defined above.
from .figures import *  # noqa: E402
from .sections import *  # noqa: E402

# Late imports: submodules use helpers defined above.
from .figures import *  # noqa: E402
from .sections import *  # noqa: E402
from .stats import *  # noqa: E402
