"""Tests for the Viewer's exported sizes — the measured particle.

``detecciones.csv`` used to fabricate every size from the box sides
(√(w·h), scaled) even when the particle had been measured over its mask.
The exported row now reads the measured fields and declares their origin
(``talla_origen``: mascara | caja). ``diam_px`` stays as a BOX descriptor
— the equal-area diameter of the box, the same definition the fallback
uses — never as the particle's size.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.protocol import compute_box_size_um  # noqa: E402
from polyx.core.detect.yolo import Detection  # noqa: E402
from polyx.gui.viewer.export import csv_rows, size_label  # noqa: E402

UM = 2.0
_CAJA_PX = 2.0 * math.sqrt(40 * 40 / math.pi)  # equal-area diameter of a 40x40 box


def _medido() -> Detection:
    """A particle measured over its mask: known values, declared origin."""
    det = Detection(class_id=0, class_name="PET", conf=0.9, x1=10, y1=10, x2=50, y2=50)
    det.diam_um = 50.0
    det.area_um2 = 100.0
    det.size_source = "mask"
    return det


def _en_caja() -> Detection:
    """A particle whose segmentation failed: the box fallback stands, declared."""
    det = Detection(class_id=1, class_name="PP", conf=0.8, x1=10, y1=10, x2=50, y2=50)
    compute_box_size_um(det, UM)
    return det


def test_csv_row_reports_the_measured_size():
    row = csv_rows([_medido()], UM)[1]
    assert row[0] == "PET"
    # The box descriptor column stays a box descriptor...
    assert row[6] == f"{_CAJA_PX:.2f}"
    # ...but the sizes are the MEASURED ones — sqrt(w*h)*um would say 80.00.
    assert row[7] == "50.00"
    assert row[8] == "100.00"
    assert row[9] == "mascara"


def test_csv_row_declares_the_box_fallback():
    det = _en_caja()
    row = csv_rows([det], UM)[1]
    assert row[7] == f"{det.diam_um:.2f}"
    assert row[9] == "caja"


def test_csv_header_declares_the_origin_column():
    assert csv_rows([], UM)[0][-1] == "talla_origen"


def test_csv_row_without_calibration_leaves_sizes_empty():
    det = Detection(class_id=0, class_name="PET", conf=1.0, x1=10, y1=10, x2=50, y2=50)
    row = csv_rows([det], 0.0)[1]
    assert row[6] == f"{_CAJA_PX:.2f}"  # the box descriptor needs no calibration
    assert row[7] == "" and row[8] == "" and row[9] == ""


def test_size_label_prefers_the_measured_um():
    assert size_label(_medido(), UM) == "PET 50.0 μm"


def test_size_label_falls_back_to_the_box_px():
    det = Detection(class_id=0, class_name="PET", conf=1.0, x1=10, y1=10, x2=50, y2=50)
    assert size_label(det, 0.0) == f"PET {_CAJA_PX:.0f} px"


def test_size_label_uncalibrated_with_measured_size_still_falls_back():
    # Without um/px there is no um to show; the measured number must not
    # be presented as if it were.
    det = _medido()
    det.diam_um = None
    assert size_label(det, None) == f"PET {_CAJA_PX:.0f} px"
