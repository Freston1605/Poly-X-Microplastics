"""Tests for the plate annotation module — colour, thickness, naming.

The seams (spec #29's testing decisions): the public functions of the
annotation module. Goldens of the drawn views live at the bottom of this
file once the behavioural tests above them are green.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core import annotate as A  # noqa: E402

# ── Colour mapping ────────────────────────────────────────────────────────


def test_bgr_for_known_class_is_the_documented_colour():
    # PET is drawn red (#e3342f) — BGR, not RGB, because the plates are BGR.
    assert A.bgr_for_class("PET") == (0x2F, 0x34, 0xE3)
    assert A.bgr_for_class("PP") == (0x00, 0x8C, 0xFF)
    assert A.bgr_for_class("LDPE") == (0x00, 0xD7, 0xFF)


def test_bgr_for_unknown_class_falls_back_to_one_colour_everywhere():
    assert A.bgr_for_class("NAILS") == (0xFF, 0xAA, 0x33)


def test_palette_keeps_the_pe_alias():
    assert A.bgr_for_class("PE") == A.bgr_for_class("LDPE")


def test_theme_reexports_the_palette_so_consumers_keep_their_import_path():
    from polyx.gui import theme as T

    assert T.CLASS_COLOR_HEX is A.CLASS_COLOR_HEX


# ── Thickness policy ──────────────────────────────────────────────────────


def test_line_width_scales_with_the_plate_and_stays_visible():
    # The docstring's contract: a full plate draws ~6 px, a crop keeps ~2,
    # the width never drops below 2.
    assert A.line_width((1630, 1630)) == 2
    assert A.line_width((3260, 2448)) == 5
    assert A.line_width((4096, 3072)) == 6
    assert A.line_width((200, 200)) == 2


# ── Run-file naming: write and parse share one owner ──────────────────────


def test_run_view_path_writes_the_documented_convention(tmp_path):
    assert A.run_view_path(tmp_path, 3, "placa1", "_annot") == tmp_path / "0003_placa1_annot.png"
    assert A.run_view_path(tmp_path, 12, "placa_2", "") == tmp_path / "0012_placa_2.png"


def test_parse_run_view_name_round_trips(tmp_path):
    # All four: the three named views and the bare prediction label file.
    for suffix in A.VIEW_SUFFIXES:
        name = A.run_view_path(tmp_path, 3, "placa1", suffix).name
        assert A.parse_run_view_name(name) == (3, "placa1", suffix), suffix


def test_parse_run_view_name_rejects_strangers():
    assert A.parse_run_view_name("image.png") is None
    assert A.parse_run_view_name("0003_placa1_annot.jpg") is None
    assert A.parse_run_view_name("placa1_annot.png") is None


def test_run_label_path_writes_the_documented_convention(tmp_path):
    assert A.run_label_path(tmp_path, 3, "placa1") == tmp_path / "0003_placa1.txt"
    assert A.run_label_path(tmp_path, 12, "placa_2") == tmp_path / "0012_placa_2.txt"


def test_parse_run_label_name_round_trips(tmp_path):
    # The label file is the prediction beside the run views: same index and
    # stem contract, .txt instead of .png, no view suffix.
    for stem in ("placa1", "placa_2", "placa_pred"):
        name = A.run_label_path(tmp_path, 3, stem).name
        assert A.parse_run_label_name(name) == (3, stem), stem


def test_parse_run_label_name_rejects_strangers():
    assert A.parse_run_label_name("classes.txt") is None
    assert A.parse_run_label_name("0003_placa1.png") is None
    assert A.parse_run_label_name("placa1.txt") is None


# ── The primitive: one annotated view ─────────────────────────────────────

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from polyx.core.detect.yolo import Detection  # noqa: E402


def _plate() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)


def _pet() -> Detection:
    return Detection(class_id=0, class_name="PET", conf=0.9, x1=20, y1=20, x2=80, y2=80)


def _gt_below() -> Detection:
    return Detection(class_id=0, class_name="PET", conf=1.0, x1=100, y1=100, x2=170, y2=170)


def test_primitive_draws_the_box_in_the_class_colour():
    out = A.draw_detections(_plate(), [_pet()])
    assert tuple(int(v) for v in out[20, 20]) == A.bgr_for_class("PET")


def test_primitive_prefixes_the_particles_number():
    numbered = _pet()
    numbered.number = 7
    with_number = A.draw_detections(_plate(), [numbered])
    explicit = A.draw_detections(_plate(), [_pet()], label_for=lambda d: "7. PET")
    assert np.array_equal(with_number, explicit)


def test_primitive_draws_the_label_when_there_is_one():
    with_label = A.draw_detections(_plate(), [_pet()])
    without_label = A.draw_detections(_plate(), [_pet()], label_for=lambda d: "")
    assert np.any(with_label != without_label), "the label tag changes the plate"


# ── The triptych: the three views of a run ────────────────────────────────


def test_triptych_builds_the_three_views():
    views = A.triptych(_plate(), [_pet()], [_gt_below()])
    assert views._fields == ("pred", "gt", "combined")
    assert tuple(int(v) for v in views.pred[20, 20]) == A.bgr_for_class("PET")
    # The GT view keeps the class colours; the combined one overlays the GT style.
    assert tuple(int(v) for v in views.gt[100, 100]) == A.bgr_for_class("PET")
    assert tuple(int(v) for v in views.combined[100, 100]) == A.GT_COLOR_BGR
    assert np.any(views.combined != views.pred), "the combined view adds the GT"


def test_triptych_without_gt_merges_the_views():
    views = A.triptych(_plate(), [_pet()], [])
    assert views.gt is None
    assert views.combined is views.pred


# ── Goldens: the style pinned, so any change is deliberate ────────────────

import hashlib  # noqa: E402

# Synthetic plate: two numbered, differently-confident predictions and one
# GT box. The hashes pin the rendered style — a change here is a review,
# never an accident.
_GOLDEN_PRED = "f05ca5e22a5b2696886bb8b31971b2f91ff8d6749072268e159497ba56db20c0"
_GOLDEN_GT = "8570f32c72e8923fb7bd3538b62a9648cc1997dd818e557bed741dc8e1d8865b"
_GOLDEN_COMBINED = "e95f9e2d39ae15248c633adf0794f122cc9164c3392c1c3d1cc5947386788524"
_GOLDEN_SINGLE = "b4e2ecbf2952e8d06f3e353072ef11a922b8e64cebe24ac3212f6b47bae8adfe"


def _golden_plate() -> tuple[np.ndarray, list, list]:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    d1 = Detection(class_id=0, class_name="PET", conf=0.9, x1=20, y1=20, x2=80, y2=80)
    d1.number = 5
    d2 = Detection(class_id=1, class_name="PP", conf=0.35, x1=110, y1=40, x2=185, y2=130)
    d2.number = 6
    return (
        img,
        [d1, d2],
        [Detection(class_id=0, class_name="PET", conf=1.0, x1=90, y1=150, x2=200, y2=200)],
    )


def _sha(view: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", view)
    assert ok
    return hashlib.sha256(bytes(buf)).hexdigest()


def test_golden_triptych_matches_the_pinned_style():
    img, dets, gt = _golden_plate()
    views = A.triptych(img, dets, gt)
    assert _sha(views.pred) == _GOLDEN_PRED
    assert _sha(views.gt) == _GOLDEN_GT
    assert _sha(views.combined) == _GOLDEN_COMBINED


def test_golden_single_view_matches_the_pinned_style():
    img, dets, _gt = _golden_plate()
    assert _sha(A.draw_detections(img, dets)) == _GOLDEN_SINGLE


def test_golden_unknown_class_renders_the_one_fallback():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    stranger = Detection(class_id=7, class_name="ZZZ", conf=0.5, x1=30, y1=30, x2=90, y2=90)
    stranger.number = 3
    assert _sha(A.draw_detections(img, [stranger])) == (
        "8d8847f0e585c0e506672b0717362a1ce0594d8df20a46c6983a53e378967412"
    )


def test_golden_full_plate_renders_the_scaled_thickness():
    # A 3260x2448 plate: line_width says 5 — the pinned render proves the
    # policy survives the scaling, not just the arithmetic.
    img = np.zeros((3260, 2448, 3), dtype=np.uint8)
    det = Detection(class_id=0, class_name="PET", conf=0.8, x1=100, y1=100, x2=600, y2=500)
    det.number = 1
    assert _sha(A.draw_detections(img, [det])) == (
        "9e38ae8756f9344cbf533ec10920f75160d5c328ea396bfb9a127407d5dc3da6"
    )


def test_parse_result_is_only_invertible_for_non_suffix_stems():
    # A stem that itself ends in a view suffix parses as that suffix —
    # inherent to the format the runner has always written. Pinned so the
    # limit is documented, not discovered.
    assert A.parse_run_view_name(A.run_view_path(Path("."), 1, "placa_pred", "").name) == (
        1,
        "placa",
        "_pred",
    )


# ── Persistence: the view saved under its own naming ──────────────────────


def test_save_view_persists_a_decodable_png_at_the_documented_name(tmp_path):
    out = A.save_view(_plate(), tmp_path, 3, "placa1", "_annot")
    assert out == tmp_path / "0003_placa1_annot.png"
    assert out.is_file()
    decoded = cv2.imdecode(np.fromfile(out, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "accented-path-safe decode round-trips"
    assert decoded.shape[:2] == (200, 200)
