"""Adapter tests for ``DetectorRunner.run``.

The runner is now a thin Qt adapter over the canonical core engine. These tests
pin the observable GUI contract -- signals, result bookkeeping, files on disk,
abort and failure behaviour -- against a fake model. The engine's own
semantics are covered by ``tests/test_detection_engine.py``; here the extra
cases are the GUI-only seams: dense-to-slot index mapping and review verdicts
stored in ``DetectorState``.

Nothing here touches ultralytics or the real ``runs/`` folder: the model is a
stub and the run folder is redirected to ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.yolo import Detection  # noqa: E402
from polyx.gui.detector.runner import DetectorRunner  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402

# Two bright squares on a black plate: high contrast, so the Otsu/ring
# segmentation inside each box succeeds and the shape measurement runs.
_BOX_TOP = (20, 20, 60, 60)
_BOX_BOTTOM = (90, 90, 130, 130)


def _plate(path: Path) -> Path:
    """Write a 160x160 plate with the two white squares the boxes point at."""
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in (_BOX_TOP, _BOX_BOTTOM):
        image[y1 + 5 : y2 - 5, x1 + 5 : x2 - 5] = 255
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buffer))
    return path


class _FakeModel:
    """Stands in for ``YoloModel``: loads instantly, returns two boxes."""

    names = {0: "PET", 1: "PP"}

    def __init__(self):
        self.is_loaded = False

    def load(self):
        self.is_loaded = True
        return self

    def predict_auto(self, _path, **_kw) -> list[Detection]:
        return [
            Detection(
                class_id=0,
                class_name="PET",
                conf=0.9,
                x1=_BOX_TOP[0],
                y1=_BOX_TOP[1],
                x2=_BOX_TOP[2],
                y2=_BOX_TOP[3],
            ),
            Detection(
                class_id=1,
                class_name="PP",
                conf=0.4,
                x1=_BOX_BOTTOM[0],
                y1=_BOX_BOTTOM[1],
                x2=_BOX_BOTTOM[2],
                y2=_BOX_BOTTOM[3],
            ),
        ]


class _FailingModel(_FakeModel):
    """Fails on one named image, so the engine can leave a partial run."""

    def __init__(self, failing_name: str):
        super().__init__()
        self.failing_name = failing_name

    def predict_auto(self, path, **kw) -> list[Detection]:
        if Path(path).name == self.failing_name:
            raise RuntimeError("simulated failure")
        return super().predict_auto(path, **kw)


def _state(tmp_path: Path, images: list[Path]) -> DetectorState:
    state = DetectorState()
    slot = state.model_slots[0]
    slot.alias = "model_a"
    slot.path = tmp_path / "best.pt"
    slot.loaded = _FakeModel()
    state.images = images
    state.calibration.measure_plate = False
    state.calibration.um_per_px = 2.0  # manual scale, so the sizes come out in um
    state.params.tiling = "nunca"
    return state


def _runner_with_folder(state: DetectorState, tmp_path: Path) -> DetectorRunner:
    """Runner whose run folder lands in tmp_path instead of the user data dir."""
    return DetectorRunner(state, runs_root=tmp_path / "runs")


def _execute(runner: DetectorRunner) -> dict:
    """Call ``run()`` synchronously and record every signal it emits."""
    events: dict[str, list] = {"progress": [], "image": [], "ok": [], "failed": [], "aborted": []}
    runner.progress.connect(lambda d, t, n: events["progress"].append((d, t, n)))
    runner.image_done.connect(lambda i, r: events["image"].append((i, r)))
    runner.finished_ok.connect(lambda: events["ok"].append(True))
    runner.failed.connect(events["failed"].append)
    runner.aborted.connect(lambda: events["aborted"].append(True))
    runner.run()  # QThread.run() called directly: synchronous, no event loop
    return events


def test_run_one_image_produces_result_files_and_progress(qapp, tmp_path):
    path = _plate(tmp_path / "photos" / "plate1.png")
    state = _state(tmp_path, [path])
    runner = _runner_with_folder(state, tmp_path)

    events = _execute(runner)

    assert events["failed"] == [], "the run must not fail"
    assert events["ok"] == [True], "finished_ok must be emitted once"

    # One result for the one active model, with both detections.
    assert set(state.results) == {0, 1, 2}
    assert len(state.results[0]) == 1
    result = state.results[0][0]
    assert result.model_idx == 0
    assert [det.number for det in result.predictions] == [1, 2], "numbering runs top to bottom"
    assert result.tiling == "", "without tiling the descriptor stays empty"

    # Files: annotated PNGs plus the YOLO labels with the confidence column.
    assert result.annotated_path and result.annotated_path.exists()
    assert result.pred_path and result.pred_path.exists()
    assert result.gt_path is None, "there is no GT, so its PNG must not be drawn"
    labels = state.run_dir / "model_a" / "labels" / "0000_plate1.txt"
    lines = labels.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(len(line.split()) == 6 for line in lines)

    # Progress counts models, and the batch closes on the last one.
    assert events["progress"] == [(1, 1, "plate1.png")]
    assert events["image"][-1][0] == 0

    # The scale of the photo is registered for the report.
    assert state.calibrations["plate1.png"].um_por_px == 2.0


def test_run_with_ground_truth_counts_tp_and_fp(qapp, tmp_path):
    path = _plate(tmp_path / "photos" / "plate2.png")
    # GT only for the upper box: the lower prediction must be a false positive.
    x1, y1, x2, y2 = _BOX_TOP
    cx, cy = (x1 + x2) / 2 / 160, (y1 + y2) / 2 / 160
    (path.parent / "plate2.txt").write_text(
        f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / 160:.6f} {(y2 - y1) / 160:.6f}\n",
        encoding="utf-8",
    )
    state = _state(tmp_path, [path])
    runner = _runner_with_folder(state, tmp_path)

    events = _execute(runner)

    assert events["failed"] == []
    result = state.results[0][0]
    assert result.has_gt is True
    assert (result.tp, result.fp, result.fn, result.miscls) == (1, 1, 0, 0)
    assert result.gt_path and result.gt_path.exists(), "with GT its PNG must be saved"
    assert result.gt_path != result.pred_path


def test_run_aborts_at_image_boundary(qapp, tmp_path):
    paths = [
        _plate(tmp_path / "photos" / "a.png"),
        _plate(tmp_path / "photos" / "b.png"),
    ]
    state = _state(tmp_path, paths)
    runner = _runner_with_folder(state, tmp_path)

    # Ask to stop as soon as the first image is done: the abort is honoured at
    # the image boundary, never mid-image, so no half-written result remains.
    def _request_stop(_index, _result):
        state.request_abort()

    runner.image_done.connect(_request_stop)
    events = _execute(runner)

    assert events["aborted"] == [True], "aborted must be emitted"
    assert events["ok"] == [], "an aborted run cannot emit finished_ok"
    assert [result.image_path.name for result in state.results[0]] == ["a.png"]
    assert state.run_result is not None
    assert state.run_result.status == "aborted"
    assert state.run_dir == state.run_result.run_dir


def test_run_failed_emits_failed_and_keeps_partial_result(qapp, tmp_path):
    paths = [
        _plate(tmp_path / "photos" / "a.png"),
        _plate(tmp_path / "photos" / "b.png"),
    ]
    state = _state(tmp_path, paths)
    state.model_slots[0].loaded = _FailingModel("b.png")
    runner = _runner_with_folder(state, tmp_path)

    events = _execute(runner)

    assert events["failed"] == ["RuntimeError: simulated failure"]
    assert events["ok"] == []
    assert events["aborted"] == []
    assert state.run_result is not None
    assert state.run_result.status == "failed"
    assert "simulated failure" in (state.run_result.error or "")
    assert state.run_dir == state.run_result.run_dir
    assert [result.image_path.name for result in state.results[0]] == ["a.png"]
    assert (state.run_dir / "run.json").is_file()


def test_dense_engine_indices_map_to_real_slots(qapp, tmp_path):
    path = _plate(tmp_path / "photos" / "plate.png")
    state = DetectorState()
    state.model_slots[0].alias = "model_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[0].loaded = _FakeModel()
    state.model_slots[2].alias = "model_c"
    state.model_slots[2].path = tmp_path / "c.pt"
    state.model_slots[2].loaded = _FakeModel()
    state.images = [path]
    state.calibration.measure_plate = False
    state.calibration.um_per_px = 2.0
    state.params.tiling = "nunca"
    runner = _runner_with_folder(state, tmp_path)

    events = _execute(runner)

    assert events["failed"] == []
    assert events["ok"] == [True]
    # The engine numbers its active models 0..N-1; the GUI keeps its real
    # model_slots indices, including the empty slot in between.
    assert set(state.results) == {0, 1, 2}
    assert len(state.results[0]) == 1
    assert state.results[1] == []
    assert len(state.results[2]) == 1
    assert [index for index, _result in events["image"]] == [0, 2]
    assert state.run_result is not None
    assert set(state.run_result.results) == {0, 1}
    assert [model.alias for model in state.run_result.models] == ["model_a", "model_c"]
    assert (state.run_dir / "model_a" / "labels" / "0000_plate.txt").is_file()
    assert (state.run_dir / "model_c" / "labels" / "0000_plate.txt").is_file()


def test_verdicts_live_in_state_and_not_in_result(qapp, tmp_path):
    path = _plate(tmp_path / "photos" / "plate.png")
    state = _state(tmp_path, [path])

    events = _execute(_runner_with_folder(state, tmp_path))

    assert events["failed"] == []
    result = state.results[0][0]
    assert not hasattr(result, "verdict")
    state.set_verdict(0, path, "buena")
    assert state.verdicts[(0, path)] == "buena"
    assert state.verdict_for(0, path) == "buena"
    assert state.verdict_for(1, path) is None
    state.set_verdict(0, path, None)
    assert state.verdict_for(0, path) is None


def test_a_new_run_clears_previous_verdicts(qapp, tmp_path):
    path = _plate(tmp_path / "photos" / "plate.png")
    state = _state(tmp_path, [path])
    _execute(_runner_with_folder(state, tmp_path))
    state.set_verdict(0, path, "buena")

    second = DetectorRunner(state, runs_root=tmp_path / "runs_second")
    events = _execute(second)

    assert events["failed"] == []
    assert state.verdicts == {}, "a new run starts unreviewed"


def test_review_dialog_reads_and_writes_state_verdicts(qapp, tmp_path):
    from polyx.gui.detector.review_dialog import ReviewDialog

    path = _plate(tmp_path / "photos" / "plate.png")
    state = _state(tmp_path, [path])
    _execute(_runner_with_folder(state, tmp_path))
    state.set_verdict(0, path, "buena")

    dialog = ReviewDialog(state)

    assert "✓" in dialog.list.item(0).text()
    dialog._set_verdict("mala")
    assert state.verdicts[(0, path)] == "mala"
    assert "✗" in dialog.list.item(0).text()
    dialog.close()


def test_run_without_models_fails_before_creating_folder(qapp, tmp_path):
    path = _plate(tmp_path / "photos" / "plate3.png")
    state = _state(tmp_path, [path])
    state.model_slots[0].path = None  # nothing active
    runner = _runner_with_folder(state, tmp_path)

    events = _execute(runner)

    assert events["failed"] == ["No hay modelos cargados."]
    assert events["ok"] == []
    assert not (tmp_path / "runs").exists(), "without models the run folder must not be created"


def test_characterization_detector_drawing_golden(qapp, tmp_path):
    """Characterization golden of the Detector runner's drawing path.

    Pins the three views (predictions, ground truth, combined) and the naming
    convention before the triptych migration (ticket #31). The engine writes
    them now; the composer must not move them.
    """
    plate_path = _plate(tmp_path / "photos" / "plate_gt.png")
    # GT only for the upper box (same as the other GT test).
    x1, y1, x2, y2 = _BOX_TOP
    cx, cy = (x1 + x2) / 2 / 160, (y1 + y2) / 2 / 160
    (plate_path.parent / "plate_gt.txt").write_text(
        f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / 160:.6f} {(y2 - y1) / 160:.6f}\n",
        encoding="utf-8",
    )
    state = _state(tmp_path, [plate_path])
    runner = _runner_with_folder(state, tmp_path)
    events = _execute(runner)

    assert events["failed"] == []
    result = state.results[0][0]
    assert result.has_gt is True

    # Naming convention: {index:04d}_{stem}{suffix}.png
    assert result.pred_path is not None and result.pred_path.name == "0000_plate_gt_pred.png"
    assert result.gt_path is not None and result.gt_path.name == "0000_plate_gt_gt.png"
    assert (
        result.annotated_path is not None
        and result.annotated_path.name == "0000_plate_gt_annot.png"
    )
    assert result.pred_path.parent == result.gt_path.parent == result.annotated_path.parent

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # Golden pixel-hashes of the three views produced by the runner before the
    # triptych migration: the composer must not move them.
    assert (
        _sha(result.pred_path) == "cd2d41b0ada491fa3da76995c438cf615edf80d3222eb9036aa3c27c5d84cba9"
    )
    assert (
        _sha(result.gt_path) == "9260e807bdcf18b92057ace6a6ec8f79655fba31bf782e2cd3c2cf72fd48bcf8"
    )
    assert (
        _sha(result.annotated_path)
        == "3be19e6adb0800e204a3e5856569e60aa7124758c7a1117b82bc3372cd39e3f7"
    )
