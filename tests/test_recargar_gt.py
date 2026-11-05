"""Regression tests for the GT reload flow (``gui.detector.recargar_gt``).

The reload path used to size every reloaded GT box with the calibration of
whatever ``ImageResult`` happened to be left over from the grouping loop,
not the image actually being processed. With per-photo calibration -- this
study spans 31-49 um/px across photos -- reloaded GT sizes came out measured
against the wrong photo's scale, and those sizes land in the report.

The seam under test is ``reload_gt(state)``: the observable outcomes are the
reloaded ``ImageResult.gt`` boxes with their sizes, plus the redrawn files.
Nothing here touches ultralytics or the real ``runs/`` folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_RAIZ = Path(__file__).resolve().parents[1]
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.yolo import Detection  # noqa: E402
from polyx.gui.detector.recargar_gt import reload_gt  # noqa: E402
from polyx.gui.detector.state import DetectorState, ImageResult  # noqa: E402

# One box, 0.2 of the side, centered: 20x20 px on a 100x100 plate.
_TXT_BOX = "0 0.5 0.5 0.2 0.2\n"
# Independent source of truth: 20x20 px = 400 px^2.
# 400 * 31^2 = 384400 -> diam = 2*sqrt(384400/pi) = 699.6 um
# 400 * 49^2 = 960400 -> diam = 2*sqrt(960400/pi) = 1105.9 um
_EXPECTED_31 = 699.6
_EXPECTED_49 = 1105.9


class _Calibration:
    """Duck-typed stand-in for the runner's per-photo calibration record."""

    def __init__(self, um_por_px: float):
        self.um_por_px = um_por_px
        self.valid = True


def _plate(path: Path) -> Path:
    """A 100x100 dark plate with a bright square, so the redraw path can decode it."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path


def _state(
    tmp_path: Path, scales: dict[str, float]
) -> tuple[DetectorState, dict[str, ImageResult]]:
    st = DetectorState()
    slot = st.model_slots[0]
    slot.alias = "m0"
    slot.path = tmp_path / "modelo.pt"
    slot.loaded = None  # no model object: class names fall back to the numeric id

    st.gt_folder = tmp_path / "gt"
    st.gt_folder.mkdir(parents=True)
    st.run_dir = tmp_path / "run"
    st.run_dir.mkdir(parents=True)
    st.params.um_per_px = 0.0  # only the per-photo calibration may provide a scale

    results: dict[str, ImageResult] = {}
    images: list[Path] = []
    for name in scales:
        path = _plate(tmp_path / "fotos" / name)
        images.append(path)
        (st.gt_folder / f"{path.stem}.txt").write_text(_TXT_BOX, encoding="utf-8")
        # A GT different from the .txt, so the reload actually re-pairs and redraws.
        old = Detection(class_id=0, class_name="PET", conf=1.0, x1=5, y1=5, x2=25, y2=25)
        results[name] = ImageResult(
            model_idx=0, image_path=path, predictions=[], gt=[old], has_gt=True
        )
    st.images = images
    st.results = {0: [results[n] for n in scales]}
    st.calibrations = {n: _Calibration(s) for n, s in scales.items()}
    return st, results


def test_reload_sizes_each_gt_with_its_own_photo_scale(qapp, tmp_path):
    st, results = _state(tmp_path, {"placa_a.png": 31.0, "placa_b.png": 49.0})

    reload_gt(st)

    for name, expected in (("placa_a.png", _EXPECTED_31), ("placa_b.png", _EXPECTED_49)):
        r = results[name]
        assert r.has_gt
        assert len(r.gt) == 1
        assert r.gt[0].diam_um == pytest.approx(expected, rel=1e-3), (
            f"{name}: reloaded GT must be sized with its own photo's calibration"
        )
