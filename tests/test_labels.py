"""Tests for the YOLO-label codec (``polyx.core.labels``).

The codec owns the YOLO ``.txt`` format and its folder conventions for the
whole program: parse, serialize, locate, classes.txt and -- at the file
level -- the EXIF-safe image size. One contract, one owner: every module
that reads or writes a label file goes through here, so the conventions
stop being re-derived per module and the dialects stay compatible.
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

from polyx.core import labels as L  # noqa: E402
from polyx.core.detect.yolo import Detection  # noqa: E402


def _png_bytes(width: int = 100, height: int = 100) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


def _png(path: Path, width: int = 100, height: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(width, height))
    return path


# ── Lines (pure) ──────────────────────────────────────────────────────────


def test_format_line_five_columns_and_round_trip():
    b = L.BBox(class_id=1, cx=0.5, cy=0.25, w=0.2, h=0.1)
    line = L.format_line(b)
    assert line == "1 0.500000 0.250000 0.200000 0.100000"
    assert L.parse_box_line(line) == b


def test_conf_column_round_trips_when_requested():
    b = L.BBox(class_id=0, cx=0.1, cy=0.2, w=0.3, h=0.4, conf=0.9123)
    line = L.format_line(b, include_conf=True)
    assert line.endswith(" 0.9123")
    assert L.parse_box_line(line).conf == pytest.approx(0.9123)


def test_conf_column_omitted_without_a_value_even_when_requested():
    b = L.BBox(class_id=0, cx=0.1, cy=0.2, w=0.3, h=0.4)
    assert L.format_line(b, include_conf=True) == L.format_line(b)


def test_parse_box_line_rejects_garbage():
    for line in ("", "   ", "hello world", "0 1 2", "x 0.1 0.1 0.1 0.1"):
        assert L.parse_box_line(line) is None


def test_bbox_from_and_to_instance_round_trip():
    det = Detection(class_id=2, class_name="LDPE", conf=0.75, x1=10, y1=20, x2=50, y2=90)
    b = L.bbox_from(det, 100, 200)
    assert (b.cx, b.cy, b.w, b.h) == (0.3, 0.275, 0.4, 0.35)

    back = L.to_instance(b, {2: "LDPE"}, 100, 200)
    assert (back.x1, back.y1, back.x2, back.y2) == pytest.approx((10, 20, 50, 90), abs=1e-3)
    assert back.conf == pytest.approx(0.75)
    assert back.class_name == "LDPE"


def test_to_instance_defaults_conf_to_full_certainty():
    inst = L.to_instance(L.BBox(0, 0.5, 0.5, 0.2, 0.2), {0: "PET"}, 100, 100)
    assert inst.conf == 1.0  # manual GT: total certainty
    assert inst.class_name == "PET"


def test_to_instance_falls_back_to_numeric_name():
    inst = L.to_instance(L.BBox(7, 0.5, 0.5, 0.1, 0.1), {0: "PET"}, 100, 100)
    assert inst.class_name == "7"


# ── Files at explicit paths ───────────────────────────────────────────────


def test_write_then_read_boxes_round_trip_creates_parents(tmp_path):
    txt = tmp_path / "a" / "b" / "labels" / "pic.txt"
    boxes = [L.BBox(0, 0.5, 0.5, 0.2, 0.2), L.BBox(1, 0.1, 0.1, 0.1, 0.1, conf=0.42)]
    L.write_lines(txt, boxes, include_conf=True)
    read, malformed = L.read_boxes(txt)
    assert malformed == 0
    assert read == boxes


def test_empty_write_creates_empty_file_and_reads_back_empty(tmp_path):
    txt = tmp_path / "labels" / "empty.txt"
    L.write_lines(txt, [])
    assert txt.exists()
    assert txt.read_text(encoding="utf-8") == ""
    assert L.read_boxes(txt) == ([], 0)


def test_read_boxes_counts_malformed_and_polygon_lines_without_boxing_them(tmp_path):
    txt = tmp_path / "pic.txt"
    txt.write_text(
        "0 0.5 0.5 0.2 0.2\n"  # good
        "not a box\n"  # malformed
        "1 0.1 0.1 0.05 0.05 0.9\n"  # good, with confidence column
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"  # polygon: valid YOLO-seg, not a box
        "9 0.5\n",  # malformed
        encoding="utf-8",
    )
    boxes, malformed = L.read_boxes(txt)
    assert len(boxes) == 2
    assert boxes[1].conf == pytest.approx(0.9)
    assert malformed == 3  # the two garbage lines + the polygon, counted and never boxed


# ── The folder contract, both directions ──────────────────────────────────


def test_find_label_file_precedence(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    img = images / "pic.png"
    img.write_bytes(b"png")
    gt_folder = tmp_path / "gt"
    gt_folder.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    assert L.find_label_file(img, gt_folder) is None  # nothing written yet
    (labels_dir / "pic.txt").write_text("", encoding="utf-8")
    assert L.find_label_file(img, None) == labels_dir / "pic.txt"  # sibling labels/
    (images / "pic.txt").write_text("", encoding="utf-8")
    assert L.find_label_file(img, None) == images / "pic.txt"  # same dir beats labels/
    (gt_folder / "pic.txt").write_text("", encoding="utf-8")
    assert L.find_label_file(img, gt_folder) == gt_folder / "pic.txt"  # explicit wins


def test_label_path_for_mirrors_the_convention(tmp_path):
    plain = tmp_path / "sueltas"
    plain.mkdir()
    assert L.label_path_for(plain / "pic.png") == plain / "pic.txt"

    images = tmp_path / "ds" / "images"
    images.mkdir(parents=True)
    assert L.label_path_for(images / "pic.png") == tmp_path / "ds" / "labels" / "pic.txt"
    assert (tmp_path / "ds" / "labels").is_dir()  # created on demand

    gt = tmp_path / "gt"
    assert L.label_path_for(images / "pic.png", gt) == gt / "pic.txt"


def test_classes_dir_for_places_the_file_beside_labels():
    assert L.classes_dir_for(Path("datos/labels")) == Path("datos")
    assert L.classes_dir_for(Path("datos")) == Path("datos")


# ── Full reader (polygon-aware, Instance currency) ────────────────────────


def test_read_label_file_detection_and_polygon(tmp_path):
    txt = tmp_path / "pic.txt"
    txt.write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",  # polygon (8 coords)
        encoding="utf-8",
    )
    instances, malformed = L.read_label_file(txt, 100, 100, {0: "PET", 1: "PP"})
    assert malformed == 0
    assert instances[0].x1 == pytest.approx(40)
    assert instances[0].x2 == pytest.approx(60)
    assert instances[0].class_name == "PET"
    assert instances[0].conf == 1.0  # GT certainty

    poly = instances[1]
    assert poly.mask is not None and poly.mask.shape == (100, 100)
    assert poly.mask_source == "gt-polygon"
    assert (poly.x1, poly.y1, poly.x2, poly.y2) == pytest.approx((10, 10, 20, 20))


def test_legacy_read_yolo_txt_keeps_its_signature(tmp_path):
    txt = tmp_path / "pic.txt"
    txt.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dets = L.read_yolo_txt(txt, 100, 100, {0: "PET"})
    assert len(dets) == 1
    assert dets[0].class_name == "PET"


# ── High level: locate + EXIF-safe size in one call ───────────────────────


def test_write_labels_and_read_labels_round_trip(tmp_path):
    img = _png(tmp_path / "images" / "pic.png")
    dets = [Detection(class_id=0, class_name="PET", conf=0.8, x1=40, y1=40, x2=60, y2=60)]
    out = L.write_labels(img, dets)
    assert out == tmp_path / "labels" / "pic.txt"

    instances, malformed = L.read_labels(img)
    assert malformed == 0
    assert instances[0].class_name == "PET"
    assert (instances[0].x1, instances[0].y1, instances[0].x2, instances[0].y2) == pytest.approx(
        (40, 40, 60, 60)
    )


def test_read_labels_finds_labeller_layout_and_study_classes(tmp_path):
    _png(tmp_path / "images" / "pic.png")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "pic.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    instances, _ = L.read_labels(tmp_path / "images" / "pic.png")
    assert instances[0].class_name == "LDPE"  # the study's classes, as fallback


def test_read_labels_without_label_file_is_empty(tmp_path):
    _png(tmp_path / "images" / "alone.png")
    assert L.read_labels(tmp_path / "images" / "alone.png") == ([], 0)


def test_write_labels_raises_when_size_unreadable(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        L.write_labels(broken, [])


# ── classes.txt ───────────────────────────────────────────────────────────


def test_read_classes_falls_back_to_the_study_classes(tmp_path):
    assert L.read_classes(tmp_path) == ["PET", "PP", "LDPE"]


def test_classes_txt_round_trip_preserves_order(tmp_path):
    L.write_classes(tmp_path, ["LDPE", "OTHER"])
    assert (tmp_path / "classes.txt").read_text(encoding="utf-8") == "LDPE\nOTHER\n"
    assert L.read_classes(tmp_path) == ["LDPE", "OTHER"]


def test_read_classes_treats_a_blank_file_as_absent(tmp_path):
    (tmp_path / "classes.txt").write_text("\n   \n", encoding="utf-8")
    assert L.read_classes(tmp_path) == ["PET", "PP", "LDPE"]
