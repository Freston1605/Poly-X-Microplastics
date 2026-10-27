"""HTML primitives of the report: escaping, stylesheet, embedded images, tables.

Everything here turns report DATA into markup and nothing else: ``_esc`` for
interpolated values, ``_report_css`` for the stylesheet, ``_img_b64`` /
``_uri_gallery`` for the base64 payloads that make the document
self-contained, and the two table builders the sections compose.

Re-exported from ``polyx.gui.report._common`` so the package surface
(76-name contract, ``tests/test_report_surface.py``) and the section
builders' ``from ._common import ...`` keep working unchanged.
"""

from __future__ import annotations

import base64
import html
from collections import Counter
from pathlib import Path

from polyx.core.i18n import tr


# ────────────────────────────────────────────────────────────────────
# Report styles (identical to the manual)
# ────────────────────────────────────────────────────────────────────
def _report_css() -> str:
    """The stylesheet in ``report.css``, read when a report is rendered.

    Read on demand and not at import time: importing the report module must
    not touch the disk, so a stylesheet missing in an install fails here,
    naming the file, instead of breaking every import of
    ``polyx.gui.report_html``.
    """
    return Path(__file__).with_name("report.css").read_text(encoding="utf-8")


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


def _img_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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

        from polyx.core.image_io import decode_image_bytes

        arr = decode_image_bytes(data)
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
