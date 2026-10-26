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

from polyx.core import report_metrics as _core_report_metrics
from polyx.core import stats as _core_stats
from polyx.core import report_inputs as _core_inputs
from polyx.gui import theme as T

globals().update({k: v for k, v in vars(_core_inputs).items() if not k.startswith("__")})

globals().update({k: v for k, v in vars(_core_report_metrics).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_core_stats).items() if not k.startswith("__")})

# ────────────────────────────────────────────────────────────────────
# Report styles (identical to the manual)
# ────────────────────────────────────────────────────────────────────
REPORT_CSS = Path(__file__).with_name("report.css").read_text(encoding="utf-8")



# The plates from the Loa river are named `<tramo>.<testigo>` with an `x`/`xx` suffix for
# the second and third plate of the same stretch. Crops add `__f<row>c<col>`.


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


_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


_MAX_GROUPS_BOXPLOT = 12
_MAX_FOLDER_PER_PHOTO = 6


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


# Late imports: submodules use helpers defined above.
from .figures import *  # noqa: E402
from .sections import *  # noqa: E402
from .stats import *  # noqa: E402
