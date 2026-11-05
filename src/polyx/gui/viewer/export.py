"""The Viewer's exported sizes — the measured particle, formatted.

Every export reads the fields the measurement layer wrote
(``diam_um``/``area_um2``, with ``size_source`` saying ``mask`` or
``box``); nothing is recomputed from the box sides here. The box's
equal-area diameter in px appears only as a declared descriptor of the
BOX — and as the fallback label when there is no calibration, because
without um/px there are no micrometers to show.
"""

from __future__ import annotations

from polyx.core.detect.protocol import equal_area_diameter
from polyx.core.detect.yolo import Detection

HEADER = [
    "clase",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "diam_px",
    "diam_um",
    "area_um2",
    "talla_origen",
]


def _box_diam_px(det: Detection) -> float:
    """Equal-area diameter of the BOX in px.

    Through the one equal-area definition in the core: a descriptor of the
    box, never presented as the particle's size.
    """
    return equal_area_diameter((det.x2 - det.x1) * (det.y2 - det.y1))


def size_label(det: Detection, um_per_px: float | None) -> str:
    """The on-canvas label: measured um when calibration allows, box px otherwise.

    ``≈`` marks an approximation: a size sourced from the box fallback
    (``size_source == "box"``) is shown with it, on the live canvas and in
    the exported PNG, so the approximate number is never mistaken for a
    measured one. Without calibration there are no µm at all and the
    box's px is shown instead — clearly not a µm figure.
    """
    if um_per_px and um_per_px > 0 and det.diam_um is not None:
        mark = "≈" if det.size_source == "box" else ""
        return f"{det.class_name} {mark}{det.diam_um:.1f} μm"
    return f"{det.class_name} {_box_diam_px(det):.0f} px"


def csv_rows(dets: list[Detection], um_per_px: float | None) -> list[list[str]]:
    """The rows of ``detecciones.csv``: measured sizes with declared origin.

    ``diam_um``/``area_um2`` are whatever the measurement layer wrote —
    mask-measured, or the box fallback when segmentation failed — and
    ``talla_origen`` says which. Without calibration there are no um, and
    none are invented.
    """
    origin_in_csv = {"mask": "mascara", "box": "caja"}  # data is English; the CSV is UI
    rows = [HEADER]
    for det in dets:
        sized = um_per_px is not None and um_per_px > 0 and det.diam_um is not None
        rows.append(
            [
                det.class_name,
                f"{det.conf:.4f}",
                f"{det.x1:.1f}",
                f"{det.y1:.1f}",
                f"{det.x2:.1f}",
                f"{det.y2:.1f}",
                f"{_box_diam_px(det):.2f}",
                f"{det.diam_um:.2f}" if sized else "",
                f"{det.area_um2:.2f}" if sized and det.area_um2 is not None else "",
                origin_in_csv.get(det.size_source, "") if sized else "",
            ]
        )
    return rows
