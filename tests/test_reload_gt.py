"""The GT reload is the run engine deciding again over the corrected labels.

A run freezes the ground truth it read: correcting an annotation afterwards
does not change its counts by itself, and the report would keep quoting the
old labels. :func:`polyx.core.detection_run.reload_ground_truth` re-reads the
``.txt`` files and recomputes measurement, matching, drawing and the run
record through the engine's own rules, without running inference again.

The seam under test is ``reload_ground_truth(run)``: the observable outcomes
are the reloaded ``ImageResult`` rows (GT boxes with their per-photo sizes,
counts, match and rule), the redrawn files and the run folder itself. Nothing
here touches ultralytics or a real ``runs/`` folder.
"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.protocol import Instance  # noqa: E402
from polyx.core.detection_run import (  # noqa: E402
    CalibrationPolicy,
    ImageResult,
    ModelInfo,
    ModelSpec,
    RunConfig,
    RunParams,
    RunResult,
    RunStatus,
    load_run,
    run_detection,
)
from polyx.core.detection_run.reload import (  # noqa: E402
    ReloadSummary,
    reload_ground_truth,
)
from polyx.core.detection_run.schema import aggregate_metrics  # noqa: E402
from polyx.core.measure.calibration import ORIGIN_MANUAL, Calibration  # noqa: E402

# One box, 0.2 of the side, centered: 20x20 px on a 100x100 plate.
_TXT_BOX = "0 0.5 0.5 0.2 0.2\n"
# Independent source of truth: 20x20 px = 400 px^2.
# 400 * 31^2 = 384400 -> diam = 2*sqrt(384400/pi) = 699.6 um
# 400 * 49^2 = 960400 -> diam = 2*sqrt(960400/pi) = 1105.9 um
_EXPECTED_31 = 699.6
_EXPECTED_49 = 1105.9


def _plate(path: Path) -> Path:
    """A 100x100 dark plate with a bright square, so the redraw path can decode it."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path


def _stale_gt() -> Instance:
    """A GT box different from ``_TXT_BOX``, so the reload actually re-pairs."""
    return Instance(class_id=0, class_name="PET", conf=1.0, x1=5, y1=5, x2=25, y2=25)


def _prediction() -> Instance:
    detection = Instance(class_id=0, class_name="PET", conf=0.9, x1=20, y1=20, x2=80, y2=80)
    detection.number = 1
    return detection


def _single_image_run(
    tmp_path: Path,
    gt_text: str | None,
    *,
    predictions: list[Instance] | None = None,
    params: RunParams | None = None,
) -> tuple[RunResult, ImageResult]:
    """A one-image run whose folder exists, ready for an end-to-end reload.

    The stale GT box is left on the result so the reload has something to
    replace and therefore actually redraws. No calibration is recorded, which
    leaves the reloaded sizes unmeasured without stopping the redraw.
    """
    plate = _plate(tmp_path / "photos" / "plate.png")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    if gt_text is not None:
        (gt_dir / "plate.txt").write_text(gt_text, encoding="utf-8")

    result = ImageResult(
        model_idx=0,
        image_path=plate,
        predictions=[_prediction()] if predictions is None else predictions,
        gt=[_stale_gt()],
        has_gt=True,
    )
    run = RunResult(
        run_dir=tmp_path / "run",
        images=[plate],
        models=[ModelInfo(alias="m0", names={0: "PET"})],
        params=params or RunParams(measure_shape=False),
        results={0: [result]},
        status=RunStatus.COMPLETED,
    )
    return run, result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reload_sizes_each_gt_with_its_own_photo_scale(tmp_path):
    """Per-photo scales (31-49 um/px) must not cross between images.

    Each reloaded GT box is measured with the calibration the run recorded for
    *that* photo. A shared or leftover scale corrupts the sizes that land in
    the report.
    """
    run_dir = tmp_path / "run"
    images: list[Path] = []
    rows: list[ImageResult] = []
    calibrations = {}
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for name, scale in (("placa_a.png", 31.0), ("placa_b.png", 49.0)):
        path = _plate(tmp_path / "fotos" / name)
        images.append(path)
        (gt_dir / f"{path.stem}.txt").write_text(_TXT_BOX, encoding="utf-8")
        rows.append(
            ImageResult(model_idx=0, image_path=path, predictions=[], gt=[_stale_gt()], has_gt=True)
        )
        calibrations[path] = Calibration(origin=ORIGIN_MANUAL, um_por_px=scale)
    run = RunResult(
        run_dir=run_dir,
        images=images,
        models=[ModelInfo(alias="m0", names={0: "PET"})],
        params=RunParams(measure_shape=False),
        calibrations=calibrations,
        results={0: rows},
        status=RunStatus.COMPLETED,
    )

    summary = reload_ground_truth(run, gt_folder=gt_dir)

    assert summary == ReloadSummary(revisited=2, changed=images, missing_gt=[], recalculated=2)
    for image, expected in zip(images, (_EXPECTED_31, _EXPECTED_49), strict=True):
        item = next(row for row in rows if row.image_path == image)
        assert item.has_gt
        assert len(item.gt) == 1
        assert item.gt[0].diam_um == pytest.approx(expected, rel=1e-3), (
            f"{image.name}: reloaded GT must be sized with its own photo's calibration"
        )


def test_reload_redraw_golden_combined_and_gt_views(tmp_path):
    """Characterization golden of the reload's redrawn views.

    Reloading a corrected ground truth redraws the combined (``_annot``) and
    GT-only (``_gt``) views of the run. Pins the run-view file names and the
    SHA-256 of both PNGs, captured before the triptych migration (#38): the
    test fails if either the drawing or the naming drifts. The predictions
    view is not rewritten -- the predictions did not change -- so
    ``pred_path`` stays ``None``.
    """
    run, result = _single_image_run(tmp_path, _TXT_BOX)

    summary = reload_ground_truth(run, gt_folder=tmp_path / "gt")

    assert summary.changed == [result.image_path]
    assert summary.recalculated == 1
    assert result.has_gt is True

    # Naming convention: {index:04d}_{stem}{suffix}.png, under the run's model alias.
    assert result.annotated_path is not None
    assert result.annotated_path.name == "0000_plate_annot.png"
    assert result.gt_path is not None
    assert result.gt_path.name == "0000_plate_gt.png"
    assert result.pred_path is None
    model_dir = run.run_dir / "m0"
    assert result.annotated_path.parent == result.gt_path.parent == model_dir

    # Golden hashes of the reload's pre-#38 drawing path: the composer
    # migration must not move them.
    assert (
        _sha(result.annotated_path)
        == "419d7f0ed6cdf164f29588bb7c3fff398af4c2bf85f5d29dbaeddad8e68f17a7"
    )
    assert (
        _sha(result.gt_path) == "ca4332b0858cb674a2ca43d6c0f0d966f9ba9e93f58ff359a0f67a8bbcf91505"
    )


def test_reload_with_an_empty_gt_file_keeps_the_gt_view(tmp_path):
    """A reviewed plate with zero boxes still writes its ``_gt`` view.

    The GT file exists (the plate was reviewed and nothing was counted),
    which is not the same as no GT file: the GT view is the untouched plate
    and the combined view is the predictions alone.
    """
    run, result = _single_image_run(tmp_path, gt_text="")

    summary = reload_ground_truth(run, gt_folder=tmp_path / "gt")

    assert summary.changed == [result.image_path]
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


def test_reload_without_gt_redraws_only_combined_view(tmp_path):
    """A stale GT that no longer has a ``.txt`` drops the ``_gt`` view.

    With no ground-truth file the reload draws the predictions alone into
    ``_annot`` and leaves ``gt_path`` (and ``pred_path``) ``None``.
    """
    run, result = _single_image_run(tmp_path, gt_text=None)

    summary = reload_ground_truth(run, gt_folder=tmp_path / "gt")

    assert summary.missing_gt == [result.image_path]
    assert summary.changed == [result.image_path]
    assert result.has_gt is False
    assert result.gt == []

    assert result.annotated_path is not None
    assert result.annotated_path.name == "0000_plate_annot.png"
    assert result.gt_path is None
    assert result.pred_path is None
    assert result.annotated_path.parent == run.run_dir / "m0"
    assert (
        _sha(result.annotated_path)
        == "f683e1d1268a40eae62a6d67de6f6ca608bdc1b9c3edea06108601329af42866"
    )


def test_reload_replaces_the_match_it_recomputes(tmp_path):
    """The reload decides again, so the identity must move with the counters.

    Leaving the old ``match`` next to new ``tp/fp/fn`` would make the Errors
    page and the report describe the labels the reload just threw away.
    """
    run, result = _single_image_run(tmp_path, _TXT_BOX)
    assert result.match is None

    reload_ground_truth(run, gt_folder=tmp_path / "gt")

    # The label is 20x20 px inside a 60x60 px prediction: box IoU 0.111, so no
    # true positive, one false positive and one missed label -- the numbers
    # the reloaded GT produces, not the ones the result carried before.
    assert result.match is not None
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert (result.match.tp, result.match.fp, result.match.fn) == (0, 1, 1)
    assert result.match.tp_pairs == []
    assert result.match.fp_idx == [0]
    assert result.match.fn_idx == [0]


def test_reload_finds_split_layout_gt_without_an_explicit_folder(tmp_path):
    """Ticket #92: the reload resolves labels/<split> exactly like the run.

    The image lives under ``images/train`` and the label under
    ``labels/train``; no GT folder is selected, so the conventional layout is
    the only lookup that can find it. A legacy same-directory decoy proves the
    conventional label wins, and the recomputed counts prove the boxes came
    from that file.
    """
    dataset = tmp_path / "dataset"
    image = _plate(dataset / "images" / "train" / "plate.png")
    label = dataset / "labels" / "train" / "plate.txt"
    label.parent.mkdir(parents=True)
    label.write_text(_TXT_BOX, encoding="utf-8")
    (image.parent / "plate.txt").write_text("0 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    result = ImageResult(
        model_idx=0,
        image_path=image,
        predictions=[_prediction()],
        gt=[_stale_gt()],
        has_gt=True,
    )
    run = RunResult(
        run_dir=tmp_path / "run",
        images=[image],
        models=[ModelInfo(alias="m0", names={0: "PET"})],
        params=RunParams(measure_shape=False),
        results={0: [result]},
        status=RunStatus.COMPLETED,
    )

    summary = reload_ground_truth(run)

    assert summary.missing_gt == []
    assert summary.changed == [image]
    assert summary.recalculated == 1
    assert result.has_gt is True
    assert len(result.gt) == 1
    assert result.gt[0].class_name == "PET"
    # The conventional label's box is centred at (50, 50), not the decoy's.
    assert result.gt[0].cx == pytest.approx(50.0)
    assert result.gt[0].cy == pytest.approx(50.0)
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.gt_path is not None and result.gt_path.name == "0000_plate_gt.png"
    assert result.gt_path.is_file()
    assert result.annotated_path is not None and result.annotated_path.is_file()


class _FixedDetector:
    """A loaded detection source: no weights, fixed instances, fresh copies."""

    def __init__(self, detections: list[Instance]):
        self._detections = detections
        self.last_imgsz = 0
        self.last_fallback = False

    @property
    def names(self) -> dict[int, str]:
        return {0: "PET", 1: "PP"}

    def predict_auto(self, image_path, **kwargs) -> list[Instance]:
        self.last_imgsz = int(kwargs.get("imgsz", 0) or 0)
        return [deepcopy(det) for det in self._detections]


def _engine_run(
    tmp_path: Path,
    image: Path,
    detector: _FixedDetector,
    *,
    gt_folder: Path | None = None,
    um_per_px: float = 0.0,
    run_name: str = "run",
) -> RunResult:
    """One model, one image, through the canonical engine (the reference run)."""
    mode = "manual" if um_per_px > 0 else "none"
    return run_detection(
        RunConfig(
            images=[image],
            models=[ModelSpec(detector=detector, alias="m0")],
            output_dir=tmp_path / run_name,
            params=RunParams(measure_shape=False),
            gt_folder=gt_folder,
            calibration=CalibrationPolicy(mode=mode, um_per_px=um_per_px),
        )
    )


def _box_detection() -> Instance:
    """The 60x60 box the stale label below also covers."""
    return Instance(class_id=0, class_name="PET", conf=0.9, x1=20, y1=20, x2=80, y2=80)


def test_reload_rewrites_the_run_record_and_load_run_agrees(tmp_path):
    """The folder must describe the reloaded images, not the frozen GT.

    A corrected ``.txt`` changes the GT boxes and the counts; both live in
    ``run.json``. ``load_run`` reads the folder alone, so it is the honest
    observer of whether the record was republished. ``detections.csv`` holds
    predictions and a reload never runs inference, so it stays byte-identical.
    """
    plate = _plate(tmp_path / "photos" / "plate.png")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    label = gt_dir / "plate.txt"
    # The stale label covers the prediction box exactly: tp=1 before the fix.
    label.write_text("0 0.5 0.5 0.6 0.6\n", encoding="utf-8")
    run = _engine_run(
        tmp_path, plate, _FixedDetector([_box_detection()]), gt_folder=gt_dir, um_per_px=2.0
    )
    item = run.results[0][0]
    assert (item.tp, item.fp, item.fn) == (1, 0, 0)
    csv_before = _sha(run.run_dir / "detections.csv")

    # The corrected label is a 20x20 box: no true positive anymore.
    label.write_text(_TXT_BOX, encoding="utf-8")
    summary = reload_ground_truth(run, gt_folder=gt_dir)

    assert summary.changed == [plate]
    assert summary.recalculated == 1
    assert (item.tp, item.fp, item.fn) == (0, 1, 1)
    # 20x20 px at 2 um/px -> 1600 um^2 -> 2*sqrt(1600/pi) = 45.1 um
    assert item.gt[0].diam_um == pytest.approx(45.14, rel=1e-3)

    loaded = load_run(run.run_dir)
    rebuilt = loaded.results[0][0]
    assert (rebuilt.tp, rebuilt.fp, rebuilt.fn, rebuilt.miscls) == (
        item.tp,
        item.fp,
        item.fn,
        item.miscls,
    )
    assert (rebuilt.tp, rebuilt.fp, rebuilt.fn) == (0, 1, 1)
    assert rebuilt.has_gt is True
    assert rebuilt.match_use_mask_iou is False
    assert rebuilt.match_iou_thr == pytest.approx(run.params.iou_tp)
    assert len(rebuilt.gt) == 1
    assert (rebuilt.gt[0].class_id, rebuilt.gt[0].x1, rebuilt.gt[0].y1) == (
        item.gt[0].class_id,
        item.gt[0].x1,
        item.gt[0].y1,
    )
    assert rebuilt.gt[0].diam_um == pytest.approx(item.gt[0].diam_um)
    assert rebuilt.annotated_path == item.annotated_path
    assert rebuilt.gt_path == item.gt_path
    assert rebuilt.pred_path == item.pred_path

    # The published aggregates and the folder's own image rows moved together.
    assert run.metrics == aggregate_metrics(run)
    assert loaded.metrics == run.metrics
    assert loaded.metrics[0]["tp"] == 0
    payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
    model_row = payload["images"][0]["models"][0]
    assert (model_row["tp"], model_row["fp"], model_row["fn"]) == (0, 1, 1)
    assert payload["metrics"]["0"]["tp"] == 0
    assert payload["images"][0]["gt"][0]["x1"] == pytest.approx(40.0)
    assert payload["images"][0]["gt"][0]["diam_um"] == pytest.approx(45.14, rel=1e-3)

    # Predictions did not change, so their table did not either.
    assert _sha(run.run_dir / "detections.csv") == csv_before


def test_reload_uses_each_photo_of_a_shared_basename_its_own_scale(tmp_path):
    """Two ``plate.png`` in different folders are two photographs, not one.

    The run keys calibrations by path; a name-keyed lookup would hand one
    photo the other's scale. Each reloaded GT is measured against its own.
    """
    images: list[Path] = []
    rows: list[ImageResult] = []
    calibrations = {}
    for folder, scale in (("set_a", 31.0), ("set_b", 49.0)):
        image = _plate(tmp_path / folder / "images" / "plate.png")
        label = tmp_path / folder / "labels" / "plate.txt"
        label.parent.mkdir(parents=True)
        label.write_text(_TXT_BOX, encoding="utf-8")
        images.append(image)
        rows.append(
            ImageResult(
                model_idx=0, image_path=image, predictions=[], gt=[_stale_gt()], has_gt=True
            )
        )
        calibrations[image] = Calibration(origin=ORIGIN_MANUAL, um_por_px=scale)

    run = RunResult(
        run_dir=tmp_path / "run",
        images=images,
        models=[ModelInfo(alias="m0", names={0: "PET"})],
        params=RunParams(measure_shape=False),
        calibrations=calibrations,
        results={0: rows},
        status=RunStatus.COMPLETED,
    )

    reload_ground_truth(run)

    for row, expected in zip(rows, (_EXPECTED_31, _EXPECTED_49), strict=True):
        assert row.gt[0].diam_um == pytest.approx(expected, rel=1e-3)


def test_reload_aborts_keeping_what_was_already_recomputed(tmp_path):
    """Cancelling the pass mid-way must leave a coherent folder behind.

    The photos already redrawn keep their new GT and counts; the rest keep the
    ones the run froze. ``run.json`` is written as each changed photo finishes,
    so the record describes exactly that mix.
    """
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    images: list[Path] = []
    rows: list[ImageResult] = []
    for name in ("a.png", "b.png", "c.png"):
        image = _plate(tmp_path / "photos" / name)
        (gt_dir / f"{image.stem}.txt").write_text(_TXT_BOX, encoding="utf-8")
        images.append(image)
        rows.append(
            ImageResult(
                model_idx=0,
                image_path=image,
                predictions=[_prediction()],
                gt=[_stale_gt()],
                has_gt=True,
            )
        )
    run = RunResult(
        run_dir=tmp_path / "run",
        images=images,
        models=[ModelInfo(alias="m0", names={0: "PET"})],
        params=RunParams(measure_shape=False),
        results={0: rows},
        status=RunStatus.COMPLETED,
    )

    calls: list[tuple[int, int, str]] = []

    def progress(done: int, total: int, name: str) -> bool:
        calls.append((done, total, name))
        return done != 2

    summary = reload_ground_truth(run, gt_folder=gt_dir, progress=progress)

    assert calls == [(1, 3, "a.png"), (2, 3, "b.png")]
    assert summary.revisited == 1
    assert summary.changed == [images[0]]
    assert summary.recalculated == 1
    assert rows[0].gt[0].x1 == pytest.approx(40.0)
    assert rows[1].gt[0].x1 == pytest.approx(5.0)
    assert rows[2].gt[0].x1 == pytest.approx(5.0)

    loaded = load_run(run.run_dir)
    assert loaded.results[0][0].gt[0].x1 == pytest.approx(40.0)
    assert loaded.results[0][1].gt[0].x1 == pytest.approx(5.0)
    assert loaded.results[0][2].gt[0].x1 == pytest.approx(5.0)


def _polygon_line(box: tuple[float, float, float, float], size: int = 100) -> str:
    """One YOLO-seg label line (normalized polygon) for a square plate."""
    x1, y1, x2, y2 = box
    coords = " ".join(
        f"{x / size:.6f} {y / size:.6f}" for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    )
    return f"0 {coords}\n"


def _label_mask(box: tuple[int, int, int, int]) -> np.ndarray:
    """A full-frame boolean mask covering ``box``."""
    mask = np.zeros((100, 100), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def test_a_masked_run_reloaded_equals_a_fresh_run(tmp_path):
    """The reload decides with the run's rule, not with a box-only fallback.

    The prediction's box is the whole plate but its mask covers the particle:
    mask IoU finds the true positive (1.0) where box IoU finds none (0.04).
    A reload that forgot ``use_mask_iou`` would report fp=1, fn=1 for the same
    corrected labels a fresh run scores as tp=1.
    """
    plate = _plate(tmp_path / "photos" / "plate.png")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    label = gt_dir / "plate.txt"
    # Before the correction the label is a different, unmasked box: mask
    # matching falls back to box IoU pair by pair and finds nothing.
    label.write_text("0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    detector = _FixedDetector(
        [
            Instance(
                class_id=0,
                class_name="PET",
                conf=0.9,
                x1=0.0,
                y1=0.0,
                x2=100.0,
                y2=100.0,
                mask=_label_mask((40, 40, 60, 60)),
                mask_source="yolo-seg",
            )
        ]
    )
    run = _engine_run(tmp_path, plate, detector, gt_folder=gt_dir)
    item = run.results[0][0]
    assert (item.tp, item.fp, item.fn) == (0, 1, 1)

    # The corrected label is the polygon of the particle: mask IoU is 1.0.
    label.write_text(_polygon_line((40, 40, 60, 60)), encoding="utf-8")
    summary = reload_ground_truth(run, gt_folder=gt_dir)

    assert summary.recalculated == 1
    assert (item.tp, item.fp, item.fn) == (1, 0, 0)
    assert item.match_use_mask_iou is True
    assert item.match_iou_thr == pytest.approx(run.params.iou_tp)
    # What the report's Methods section reads: the run matches by mask IoU.
    assert run.match_use_mask_iou is True

    payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
    model_row = payload["images"][0]["models"][0]
    assert model_row["match_use_mask_iou"] is True
    assert (model_row["tp"], model_row["fp"], model_row["fn"]) == (1, 0, 0)

    # Independent source of truth: a fresh engine run over the same labels.
    fresh = _engine_run(
        tmp_path,
        plate,
        _FixedDetector([deepcopy(item.predictions[0])]),
        gt_folder=gt_dir,
        run_name="fresh",
    )
    fresh_item = fresh.results[0][0]
    assert (fresh_item.tp, fresh_item.fp, fresh_item.fn) == (item.tp, item.fp, item.fn)
    assert fresh_item.match_use_mask_iou == item.match_use_mask_iou
    assert fresh_item.match is not None and item.match is not None
    assert fresh_item.match.tp_pairs == item.match.tp_pairs


def test_a_run_reopened_from_disk_reloads_with_the_rule_it_can_apply(tmp_path):
    """A folder holds no prediction masks, and the record must say so.

    Reloading from the GUI keeps the masks in memory, so mask IoU still
    decides. Reopening the run from its folder leaves box-only predictions:
    the reload cannot mask-match, decides with box IoU and records that rule,
    instead of repeating the flag of a decision it could not take.
    """
    plate = _plate(tmp_path / "photos" / "plate.png")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    label = gt_dir / "plate.txt"
    # Box label covering the particle; the prediction's box is the whole plate.
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    detector = _FixedDetector(
        [
            Instance(
                class_id=0,
                class_name="PET",
                conf=0.9,
                x1=0.0,
                y1=0.0,
                x2=100.0,
                y2=100.0,
                mask=_label_mask((40, 40, 60, 60)),
                mask_source="yolo-seg",
            )
        ]
    )
    run = _engine_run(tmp_path, plate, detector, gt_folder=gt_dir)
    assert run.results[0][0].match_use_mask_iou is True

    loaded = load_run(run.run_dir)
    assert loaded.results[0][0].predictions[0].mask is None

    # A different label forces the reload to decide again.
    label.write_text("0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    summary = reload_ground_truth(loaded, gt_folder=gt_dir)

    assert summary.recalculated == 1
    row = loaded.results[0][0]
    assert row.match_use_mask_iou is False
    assert row.match_iou_thr == pytest.approx(loaded.params.iou_tp)
    payload = json.loads((loaded.run_dir / "run.json").read_text(encoding="utf-8"))
    model_row = payload["images"][0]["models"][0]
    assert model_row["match_use_mask_iou"] is False
