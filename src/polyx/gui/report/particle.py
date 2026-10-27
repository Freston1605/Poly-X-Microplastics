"""Particle card with mask overlay for the fichas section.

``_record_particle`` crops the particle, draws its measured contour in
green on a side-by-side view, and appends the pixel→µm accounting with
Feret/geodesic for comparison. It uses the SAME ``box_margin`` as
``segment`` (must not drift: see the comment in the body and
``tests/test_mascara_alineada.py``), keeps heavy imports lazy (cv2,
``polyx.core.measure.*``) and returns ``""`` on any failure so the
report never crashes over a bad crop.

Re-exported from ``polyx.gui.report._common`` (removed in refactor) for the package surface and
call sites.
"""

from __future__ import annotations

import numpy as np

from polyx.core.i18n import tr

# Marker that is replaced by the final section number during assembly.
NUM = "@@N@@"


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
        from .markup import _img_b64

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
