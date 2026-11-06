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

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _reload_state(tmp_path: Path, gt_text: str | None) -> tuple[DetectorState, ImageResult]:
    """One-image state with a loaded model, ready for an end-to-end reload.

    The model is a duck-typed stand-in: ``reload_gt`` reads only ``.names``
    from the first loaded slot (to label the boxes the ``.txt`` stores as an
    index). A stale GT box is left on the result so the reload has something
    to replace and therefore actually redraws.
    """
    plate = _plate(tmp_path / "photos" / "plate.png")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    if gt_text is not None:
        (gt_dir / "plate.txt").write_text(gt_text, encoding="utf-8")

    st = DetectorState()
    slot = st.model_slots[0]
    slot.alias = "m0"
    slot.path = tmp_path / "best.pt"
    slot.loaded = SimpleNamespace(names={0: "PET"})
    st.gt_folder = gt_dir
    st.run_dir = tmp_path / "run"
    st.run_dir.mkdir()
    st.images = [plate]
    # The scale is per photo; {} falls back to params.um_per_px (0 -> None),
    # which leaves the reloaded sizes uncalibrated but does not stop the redraw.
    st.calibrations = {}

    prediction = Detection(class_id=0, class_name="PET", conf=0.9, x1=20, y1=20, x2=80, y2=80)
    prediction.number = 1
    stale_gt = Detection(class_id=0, class_name="PET", conf=1.0, x1=5, y1=5, x2=25, y2=25)
    result = ImageResult(
        model_idx=0,
        image_path=plate,
        predictions=[prediction],
        gt=[stale_gt],
        has_gt=True,
    )
    st.results = {0: [result]}
    return st, result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reload_redraw_golden_combined_and_gt_views(qapp, tmp_path):
    """Characterization golden of the GT reload's redrawn views.

    Reloading a corrected ground truth redraws the combined (``_annot``) and
    GT-only (``_gt``) views of the run. Pins the run-view file names and the
    SHA-256 of both PNGs, captured before the triptych migration (#38): the
    test fails if either the drawing or the naming drifts. The predictions
    view is not rewritten -- the predictions did not change -- so
    ``pred_path`` stays ``None``. No production code is changed here.
    """
    st, result = _reload_state(tmp_path, _TXT_BOX)

    summary = reload_gt(st)

    assert summary["cambiadas"] == ["plate.png"]
    assert summary["recalculadas"] == 1
    assert result.has_gt is True

    # Naming convention: {index:04d}_{stem}{suffix}.png, under the run's model alias.
    assert result.annotated_path is not None
    assert result.annotated_path.name == "0000_plate_annot.png"
    assert result.gt_path is not None
    assert result.gt_path.name == "0000_plate_gt.png"
    assert result.pred_path is None
    model_dir = st.run_dir / "m0"
    assert result.annotated_path.parent == result.gt_path.parent == model_dir

    # Golden hashes of the reload's current drawing path
    # (_draw_annotated / _color_bgr_for_class). Captured before #38.
    assert (
        _sha(result.annotated_path)
        == "419d7f0ed6cdf164f29588bb7c3fff398af4c2bf85f5d29dbaeddad8e68f17a7"
    )
    assert (
        _sha(result.gt_path) == "ca4332b0858cb674a2ca43d6c0f0d966f9ba9e93f58ff359a0f67a8bbcf91505"
    )


def test_reload_with_an_empty_gt_file_keeps_the_gt_view(qapp, tmp_path):
    """A reviewed plate with zero boxes still writes its ``_gt`` view.

    The GT file exists (the plate was reviewed and nothing was counted),
    which is not the same as no GT file: the GT view is the untouched plate
    and the combined view is the predictions alone.
    """
    st, result = _reload_state(tmp_path, gt_text="")

    summary = reload_gt(st)

    assert summary["cambiadas"] == ["plate.png"]
    assert result.has_gt is True
    assert result.gt == []
    assert result.annotated_path is not None
    assert result.annotated_path.name == "0000_plate_annot.png"
    assert result.gt_path is not None
    assert result.gt_path.name == "0000_plate_gt.png"
    assert result.pred_path is None
    # The combined view carries no GT boxes, so it is the predictions render.
    assert (
        _sha(result.annotated_path)
        == "f683e1d1268a40eae62a6d67de6f6ca608bdc1b9c3edea06108601329af42866"
    )


def test_reload_without_gt_redraws_only_combined_view(qapp, tmp_path):
    """A stale GT that no longer has a ``.txt`` drops the ``_gt`` view.

    With no ground-truth file the reload draws the predictions alone into
    ``_annot`` and leaves ``gt_path`` (and ``pred_path``) ``None``.
    """
    st, result = _reload_state(tmp_path, gt_text=None)

    summary = reload_gt(st)

    assert summary["sin_gt"] == ["plate.png"]
    assert summary["cambiadas"] == ["plate.png"]
    assert result.has_gt is False
    assert result.gt == []

    assert result.annotated_path is not None
    assert result.annotated_path.name == "0000_plate_annot.png"
    assert result.gt_path is None
    assert result.pred_path is None
    assert result.annotated_path.parent == st.run_dir / "m0"
    assert (
        _sha(result.annotated_path)
        == "f683e1d1268a40eae62a6d67de6f6ca608bdc1b9c3edea06108601329af42866"
    )
