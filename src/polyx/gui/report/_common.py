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
import numpy as np

matplotlib.use("Agg")

from polyx.core.i18n import language, tr
from polyx.gui import theme as T

# Machine provenance (equipment table, checkpoint training info) lives in
# machine.py. It is re-exported HERE, not imported by its readers, because
# the section builders read it through this module at call time and that is
# what the byte-identity fixtures patch.
from .machine import _team as _team
from .machine import _training_of as _training_of

# HTML primitives (escape, stylesheet, base64 embeds, tables) live in
# markup.py; re-exported here because that is where the section builders
# and the package surface import them from.
from .markup import (
    _WIDTH_GALLERY as _WIDTH_GALLERY,
)
from .markup import (
    _esc as _esc,
)
from .markup import (
    _img_b64 as _img_b64,
)
from .markup import (
    _report_css as _report_css,
)
from .markup import (
    _table_count as _table_count,
)
from .markup import (
    _table_width as _table_width,
)
from .markup import (
    _uri_gallery as _uri_gallery,
)

# Samples/ordering/sampling/batch aggregation live in samples.py;
# re-exported here to keep the package surface and call sites stable.
from .samples import (
    _MAX_FOLDER_PER_PHOTO as _MAX_FOLDER_PER_PHOTO,
)
from .samples import (
    _ORDER_STATION as _ORDER_STATION,
)
from .samples import (
    _PLATE as _PLATE,
)
from .samples import (
    _RE_PLATE as _RE_PLATE,
)
from .samples import (
    _STATION as _STATION,
)
from .samples import (
    _batch_metrics as _batch_metrics,
)
from .samples import (
    _clue_order as _clue_order,
)
from .samples import (
    _lot_forms as _lot_forms,
)
from .samples import (
    _sample_of as _sample_of,
)
from .samples import (
    choose_records as choose_records,
)

# Three presentation colours of the report (fixed palette: a report is
# printed and archived, so it must not follow the UI theme).
_PALETTE = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


_MAX_GROUPS_BOXPLOT = 12


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
<style>{_report_css()}</style></head><body>
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
