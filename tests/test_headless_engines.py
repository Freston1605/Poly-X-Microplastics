"""Fixture tests for the headless engines (no Ultralytics, no Qt).

Covers the new core engines: polygon GT parsing, mask IoU and mask-aware
matching, ``run_batch`` on a stub detector, the compact report generator,
and the pure metric extraction of the evaluation module. These modules
previously had only translation-walker coverage, which let a silent
zero-mask bug through — see the fillPoly note in
``core/detect/protocol.py``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import pytest

from polyx.core.detect.protocol import Instance, find_gt_for_image, read_yolo_txt
from polyx.core.detection_run import run_batch
from polyx.core.quality.metrics import mask_iou, match_image
from polyx.core.report_simple import generate as generate_report

# A cross-shaped polygon: its bounding box is the whole (200x150) frame,
# while the mask itself is the two bars. Any box-vs-box comparison of a
# small box against it scores IoU ~0.13, so box IoU and mask IoU can be
# told apart unambiguously in the tests below.
CROSS_POINTS_PX = [
    [90, 0], [110, 0], [110, 90], [200, 90], [200, 110], [110, 110],
    [110, 150], [90, 150], [90, 110], [0, 110], [0, 90], [90, 90],
]


def _cross_mask(w: int = 200, h: int = 150) -> np.ndarray:
    u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(u8, [np.array(CROSS_POINTS_PX, dtype=np.int32)], 1)
    return u8.astype(bool)


def _cross_polygon_txt() -> str:
    pts = " ".join(f"{x / 200:.4f} {y / 150:.4f}" for x, y in CROSS_POINTS_PX)
    return f"0 {pts}\n"


def _write_image(path, w: int = 200, h: int = 150) -> None:
    """A white canvas with one dark square so segmentation has something."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (90, 80), (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


class _FakeDetector:
    """Minimal DetectionSource: returns fixed boxes, owns names."""

    def __init__(self, boxes, masks: Optional[List[np.ndarray]] = None):
        self._boxes = boxes
        self._masks = masks or [None] * len(boxes)
        self._names = {0: "particle", 1: "fiber"}

    def load(self):
        return self

    @property
    def names(self):
        return self._names

    @property
    def last_imgsz(self) -> int:
        return 1280

    def predict(self, image_path, conf=0.25, iou=0.45, imgsz=1280,
                device=None, **kwargs) -> List[Instance]:
        out = []
        for i, (cid, x1, y1, x2, y2) in enumerate(self._boxes):
            out.append(Instance(
                class_id=cid,
                class_name=self._names.get(cid, str(cid)),
                conf=0.9,
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                mask=self._masks[i],
                mask_source="stub" if self._masks[i] is not None else None,
            ))
        return out


def _run_dir(tmp_path):
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    _write_image(image_dir / "img1.png")
    out = tmp_path / "run"
    return image_dir, out


# ── Polygon GT parsing ──────────────────────────────────────────────────

def test_read_yolo_txt_polygon_mask_is_filled(tmp_path):
    txt = tmp_path / "img1.txt"
    # class + four (x, y) pairs: a square from 20% to 80% of the frame
    txt.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8")
    inst = read_yolo_txt(txt, 100, 100, {0: "particle"})
    assert len(inst) == 1
    d = inst[0]
    assert d.mask_source == "gt-polygon"
    # The regression: the mask used to come out all zeros because fillPoly
    # filled a throwaway copy.
    assert d.mask is not None and d.mask.any()
    area = d.mask.sum() / (100 * 100)
    assert 0.25 < area < 0.8
    assert d.x1 == pytest.approx(20.0)
    assert d.x2 == pytest.approx(80.0)


def test_read_yolo_txt_detection_branch_unchanged(tmp_path):
    txt = tmp_path / "img1.txt"
    txt.write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    inst = read_yolo_txt(txt, 100, 100, {0: "particle"})
    assert len(inst) == 1
    d = inst[0]
    assert d.x1 == pytest.approx(40.0) and d.x2 == pytest.approx(60.0)
    assert d.mask is None          # detection format carries no mask
    assert d.mask_source is None


def test_read_yolo_txt_cross_polygon_bbox_covers_shape(tmp_path):
    txt = tmp_path / "img1.txt"
    txt.write_text(_cross_polygon_txt(), encoding="utf-8")
    inst = read_yolo_txt(txt, 200, 150, {0: "particle"})
    assert len(inst) == 1
    d = inst[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (0.0, 0.0, 200.0, 150.0)
    # The polygon survives a float round-trip through normalized
    # coordinates, so compare near-equality instead of exact bytes
    # (a rounding diff of one pixel row is expected).
    m_pred, m_exp = d.mask, _cross_mask()
    inter = np.logical_and(m_pred, m_exp).sum()
    union = np.logical_or(m_pred, m_exp).sum()
    assert inter / union > 0.97


# ── Mask-aware matching ─────────────────────────────────────────────────

def test_mask_iou_falls_back_to_box_without_masks():
    a = Instance(class_id=0, class_name="p", conf=1.0, x1=0, y1=0, x2=10, y2=10)
    b = Instance(class_id=0, class_name="p", conf=1.0, x1=0, y1=0, x2=10, y2=10)
    assert mask_iou(a, b) == pytest.approx(1.0)   # identical boxes


def test_mask_iou_computes_mask_ratio():
    m1 = np.zeros((20, 20), dtype=bool)
    m1[0:10, 0:10] = True
    m2 = np.zeros((20, 20), dtype=bool)
    m2[5:15, 5:15] = True          # inter 25 px, union 175 px
    assert mask_iou(SimpleNamespace(mask=m1), SimpleNamespace(mask=m2)) \
        == pytest.approx(25 / 175)


def test_matching_switches_tp_only_through_masks():
    same_mask = _cross_mask()
    # pred: small box + cross mask; gt: whole-frame bbox + same cross mask
    pred = Instance(class_id=0, class_name="p", conf=1.0,
                    x1=0, y1=90, x2=200, y2=110, mask=same_mask)
    gt = Instance(class_id=0, class_name="p", conf=1.0,
                  x1=0, y1=0, x2=200, y2=150, mask=same_mask.copy())
    box_only = match_image([pred], [gt], iou_thr=0.5, use_mask_iou=False)
    assert box_only.fp == 1        # box IoU ~0.13 < 0.5
    masked = match_image([pred], [gt], iou_thr=0.5, use_mask_iou=True)
    assert masked.tp == 1          # identical masks -> IoU 1.0


# ── run_batch on a stub detector ────────────────────────────────────────

def test_run_batch_counts_and_outputs(tmp_path):
    image_dir, out = _run_dir(tmp_path)
    # prediction matches the dark square; GT declares the same box
    model = _FakeDetector([(0, 50, 40, 90, 80)])
    gt = image_dir / "img1.txt"
    cx, cy, w, h = (70 / 200), (60 / 150), (40 / 200), (40 / 150)
    gt.write_text(f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n", encoding="utf-8")

    summary = run_batch(model, image_dir, out, device="cpu", measure_shape=False,
                        gt_folder=image_dir)
    assert summary["images"] == 1 and summary["detections"] == 1
    assert summary["has_gt"] is True
    assert summary["metrics"]["tp"] == 1 and summary["metrics"]["fp"] == 0
    assert summary["metrics"]["fn"] == 0

    csv_text = (out / "centroids.csv").read_text(encoding="utf-8")
    assert "class_id" in csv_text and "img1.png" in csv_text
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["tp"] == 1
    assert (out / "images" / "img1.png").is_file()


def test_run_batch_uses_mask_iou_when_both_sides_have_masks(tmp_path):
    image_dir, out = _run_dir(tmp_path)
    gt = image_dir / "img1.txt"
    gt.write_text(_cross_polygon_txt(), encoding="utf-8")
    # Prediction: small box + the same cross mask. Box IoU vs the
    # whole-frame GT bbox is ~0.13, so box matching would call it FP;
    # identical masks must make it a TP through the mask path.
    model = _FakeDetector([(0, 0, 90, 200, 110)], masks=[_cross_mask()])
    summary = run_batch(model, image_dir, out, device="cpu", measure_shape=False,
                        gt_folder=image_dir)
    assert summary["metrics"]["tp"] == 1, summary


def test_run_batch_without_masks_keeps_box_matching(tmp_path):
    image_dir, out = _run_dir(tmp_path)
    gt = image_dir / "img1.txt"
    gt.write_text(_cross_polygon_txt(), encoding="utf-8")
    # Neither side carries masks: matching must degrade to box IoU silently.
    model = _FakeDetector([(0, 0, 90, 200, 110)])
    summary = run_batch(model, image_dir, out, device="cpu", measure_shape=False,
                        gt_folder=image_dir)
    assert summary["metrics"]["fp"] == 1


def test_run_batch_empty_inputs_still_write_csv(tmp_path):
    image_dir, out = _run_dir(tmp_path)
    model = _FakeDetector([])
    summary = run_batch(model, image_dir, out, device="cpu", measure_shape=False)
    assert summary["detections"] == 0
    assert (out / "centroids.csv").read_text(encoding="utf-8").startswith("image")


# ── Mask-aware measurement ─────────────────────────────────────────────

def test_measurement_prefers_attached_mask_over_otsu():
    """The provided mask wins; the det.mask fallback works too."""
    from polyx.core.detect.protocol import Instance
    from polyx.core.measure.morphology import apply_to_detection

    img = np.full((150, 200, 3), 255, dtype=np.uint8)
    cv2.circle(img, (70, 60), 12, (0, 0, 0), -1)
    full = np.zeros((150, 200), dtype=bool)
    full[40:100, 50:110] = True        # 60 x 60 square, mask area = 3600 px

    box_only = Instance(class_id=0, class_name="p", conf=0.9,
                        x1=50, y1=40, x2=90, y2=80)
    assert apply_to_detection(box_only, img, 1.0) is False   # Otsu fails here
    assert box_only.area_um2 is None

    # explicit mask argument
    det = Instance(class_id=0, class_name="p", conf=0.9,
                   x1=50, y1=40, x2=90, y2=80)
    assert apply_to_detection(det, img, 1.0, mask=full) is True
    assert det.area_um2 == pytest.approx(3600.0, rel=0.01)

    # fallback through det.mask (what a two-stage model attaches)
    det2 = Instance(class_id=0, class_name="p", conf=0.9,
                    x1=50, y1=40, x2=90, y2=80, mask=full)
    assert apply_to_detection(det2, img, 1.0) is True
    assert det2.area_um2 == pytest.approx(3600.0, rel=0.01)


def test_run_batch_records_per_image_metrics_and_params(tmp_path):
    """Per-image counts must sum to the aggregate, never repeat it."""
    image_dir, out = _run_dir(tmp_path)
    _write_image(image_dir / "img2.png")
    # img1: prediction matches GT -> TP. img2: GT sits elsewhere -> FP + FN.
    (image_dir / "img1.txt").write_text("0 0.35 0.4 0.2 0.2667\n", encoding="utf-8")
    (image_dir / "img2.txt").write_text("0 0.95 0.95 0.05 0.05\n", encoding="utf-8")
    model = _FakeDetector([(0, 50, 40, 90, 80)])

    summary = run_batch(model, image_dir, out, device="cpu", measure_shape=False,
                        gt_folder=image_dir)
    assert summary["per_image"]["img1.png"] == {"tp": 1, "fp": 0, "fn": 0, "miscls": 0}
    assert summary["per_image"]["img2.png"] == {"tp": 0, "fp": 1, "fn": 1, "miscls": 0}
    assert sum(v["tp"] for v in summary["per_image"].values()) == summary["metrics"]["tp"]

    assert summary["params"]["imgsz"] == 1280
    assert summary["params"]["conf"] == 0.25
    assert summary["params"]["measure_shape"] is False
    assert summary["params"]["device"] == "cpu"


def test_run_state_rebuild_does_not_multiply_metrics(tmp_path):
    """The report sums per image: rebuilt state must carry per-image counts."""
    image_dir, out = _run_dir(tmp_path)
    _write_image(image_dir / "img2.png")
    (image_dir / "img1.txt").write_text("0 0.35 0.4 0.2 0.2667\n", encoding="utf-8")
    (image_dir / "img2.txt").write_text("0 0.95 0.95 0.05 0.05\n", encoding="utf-8")
    model = _FakeDetector([(0, 50, 40, 90, 80)])
    summary = run_batch(model, image_dir, out, device="cpu", measure_shape=False,
                        gt_folder=image_dir)

    from polyx.gui.detector.run_state import build_state_from_run
    state = build_state_from_run(out, image_dir)
    rs = state.results[0]
    by_name = {r.image_path.name: r for r in rs}
    assert by_name["img1.png"].tp == 1
    assert by_name["img2.png"].tp == 0 and by_name["img2.png"].fn == 1
    # The invariant the report relies on: the per-image sum is the aggregate.
    assert sum(r.tp for r in rs) == summary["metrics"]["tp"]
    assert all(r.has_gt for r in rs)
    assert all(r.annotated_path.is_file() for r in rs)
    # Run parameters travel with the run instead of defaulting to 1280/0.25.
    assert state.params.imgsz == summary["params"]["imgsz"]


# ── Compact report ──────────────────────────────────────────────────────

def test_report_simple_generates_html(tmp_path):
    image_dir, out = _run_dir(tmp_path)
    run_batch(_FakeDetector([(0, 50, 40, 90, 80)]), image_dir, out,
              device="cpu", measure_shape=False)
    report = generate_report(out)
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Poly-X run report" in text
    assert "particle" in text


# ── Evaluation metric extraction (no Ultralytics needed) ────────────────

def test_eval_metrics_extraction_maps_sparse_classes():
    from polyx.core.evaluation import _metrics_from_results
    box = SimpleNamespace(
        map50=0.75, map=0.6,
        p=[0.8, 0.7], r=[0.9, 0.6], ap50=[0.85, 0.65],
        # Real Ultralytics attribute name; class 1 never appears in the
        # val set, so p[1] belongs to class 2 and dense indexing would
        # mislabel it.
        ap_class_index=[0, 2],
    )
    results = SimpleNamespace(box=box)
    metrics = _metrics_from_results(results, {0: "particle", 1: "fiber", 2: "pellet"})
    assert metrics["mAP50"] == pytest.approx(0.75)
    assert metrics["mAP50-95"] == pytest.approx(0.6)
    # class 1 is absent from ap_class_index; only present classes are reported
    assert set(metrics["per_class"]) == {"particle", "pellet"}
    assert metrics["per_class"]["pellet"]["precision"] == pytest.approx(0.7)
    assert metrics["per_class"]["particle"]["AP50"] == pytest.approx(0.85)


def test_eval_write_report_creates_parents(tmp_path):
    from polyx.core.evaluation import write_report
    deep = tmp_path / "a" / "b" / "metrics.json"
    write_report({"mAP50": 0.5}, deep)
    assert deep.is_file()
    assert json.loads(deep.read_text(encoding="utf-8"))["mAP50"] == 0.5


# ── GT lookup helper ────────────────────────────────────────────────────

def test_find_gt_for_image_matches_stem(tmp_path):
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    _write_image(image_dir / "img1.png")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert find_gt_for_image(image_dir / "img1.png", labels) == labels / "img1.txt"
