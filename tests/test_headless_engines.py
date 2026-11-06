"""Fixture tests for the Qt-free headless helpers (no Ultralytics, no Qt).

Covers polygon GT parsing, mask IoU and mask-aware matching, mask-aware
measurement, the pure metric extraction of the evaluation module, and the GT
lookup helper. These modules previously had only translation-walker coverage,
which let a silent zero-mask bug through — see the fillPoly note in
``core/detect/protocol.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from polyx.core.detect.protocol import Instance, find_gt_for_image, read_yolo_txt
from polyx.core.quality.metrics import mask_iou, match_image

# A cross-shaped polygon: its bounding box is the whole (200x150) frame,
# while the mask itself is the two bars. Any box-vs-box comparison of a
# small box against it scores IoU ~0.13, so box IoU and mask IoU can be
# told apart unambiguously in the tests below.
CROSS_POINTS_PX = [
    [90, 0],
    [110, 0],
    [110, 90],
    [200, 90],
    [200, 110],
    [110, 110],
    [110, 150],
    [90, 150],
    [90, 110],
    [0, 110],
    [0, 90],
    [90, 90],
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
    assert d.mask is None  # detection format carries no mask
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
    assert mask_iou(a, b) == pytest.approx(1.0)  # identical boxes


def test_mask_iou_computes_mask_ratio():
    m1 = np.zeros((20, 20), dtype=bool)
    m1[0:10, 0:10] = True
    m2 = np.zeros((20, 20), dtype=bool)
    m2[5:15, 5:15] = True  # inter 25 px, union 175 px
    assert mask_iou(SimpleNamespace(mask=m1), SimpleNamespace(mask=m2)) == pytest.approx(25 / 175)


def test_matching_switches_tp_only_through_masks():
    same_mask = _cross_mask()
    # pred: small box + cross mask; gt: whole-frame bbox + same cross mask
    pred = Instance(
        class_id=0, class_name="p", conf=1.0, x1=0, y1=90, x2=200, y2=110, mask=same_mask
    )
    gt = Instance(
        class_id=0, class_name="p", conf=1.0, x1=0, y1=0, x2=200, y2=150, mask=same_mask.copy()
    )
    box_only = match_image([pred], [gt], iou_thr=0.5, use_mask_iou=False)
    assert box_only.fp == 1  # box IoU ~0.13 < 0.5
    masked = match_image([pred], [gt], iou_thr=0.5, use_mask_iou=True)
    assert masked.tp == 1  # identical masks -> IoU 1.0


# ── Mask-aware measurement ─────────────────────────────────────────────


def test_measurement_prefers_attached_mask_over_otsu():
    """The provided mask wins; the det.mask fallback works too."""
    from polyx.core.detect.protocol import Instance
    from polyx.core.measure.shape import apply_to_detection

    img = np.full((150, 200, 3), 255, dtype=np.uint8)
    cv2.circle(img, (70, 60), 12, (0, 0, 0), -1)
    full = np.zeros((150, 200), dtype=bool)
    full[40:100, 50:110] = True  # 60 x 60 square, mask area = 3600 px

    box_only = Instance(class_id=0, class_name="p", conf=0.9, x1=50, y1=40, x2=90, y2=80)
    assert apply_to_detection(box_only, img, 1.0) is False  # Otsu fails here
    assert box_only.area_um2 is None

    # explicit mask argument
    det = Instance(class_id=0, class_name="p", conf=0.9, x1=50, y1=40, x2=90, y2=80)
    assert apply_to_detection(det, img, 1.0, mask=full) is True
    assert det.area_um2 == pytest.approx(3600.0, rel=0.01)

    # fallback through det.mask (what a two-stage model attaches)
    det2 = Instance(class_id=0, class_name="p", conf=0.9, x1=50, y1=40, x2=90, y2=80, mask=full)
    assert apply_to_detection(det2, img, 1.0) is True
    assert det2.area_um2 == pytest.approx(3600.0, rel=0.01)


# ── Evaluation metric extraction (no Ultralytics needed) ────────────────


def test_eval_metrics_extraction_maps_sparse_classes():
    from polyx.core.evaluation import _metrics_from_results

    box = SimpleNamespace(
        map50=0.75,
        map=0.6,
        p=[0.8, 0.7],
        r=[0.9, 0.6],
        ap50=[0.85, 0.65],
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
