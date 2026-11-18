"""Forcing tiling must not change the TILE SIZE.

Whether to tile and at which size are two different questions, and mixing them
up caused a failure with strange symptoms: choosing "always" in the interface
left detection extremely slow and with ZERO results.

The cause: ``predict_auto`` forced tiling by passing ``threshold_px=0``, its way
of saying "do not decide, tile". But the tile side came from that same
threshold, ``min(threshold_px, imgsz)``, that is ``min(0, 2080) = 0``, which the
clamp raised to 256. On a 4096x3072 photo that is 336 tiles of 256 px, and each
one is rescaled 8x up to imgsz: the model sees nothing resembling what it was
trained on and detects nothing.

Checked on 10.3x.jpg with the yolov11m weights: with automatic tiling it gives
64 healthy detections, and with forced tiling it gave 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyx.core.detect.protocol import policy_tiling as protocol_policy_tiling
from polyx.core.detect.tiling import (
    TilingDecision,
    TilingMode,
    TilingPlan,
    decide_tiling,
    describe_tiling,
    policy_tiling,
)
from polyx.core.detect.yolo import policy_tiling as yolo_policy_tiling

# The study's typical photo and the imgsz it was trained at.
WIDTH, HEIGHT, IMGSZ = 4096, 3072, 2080


# ── The geometry the policy always had ─────────────────────────────────


def test_forcing_tiling_does_not_shrink_the_tile():
    """The failure, in one line: the tile must be in the order of imgsz."""
    plan = policy_tiling(WIDTH, HEIGHT, IMGSZ, threshold_px=0, tile=0)
    assert plan is not None
    assert plan.tile >= IMGSZ // 2, (
        f"tile of {plan.tile} px: it would be rescaled "
        f"{IMGSZ / plan.tile:.0f}x up to imgsz and the model would recognise nothing"
    )


def test_forcing_tiling_does_not_explode_the_tile_count():
    """The other symptom. 336 tiles was a wait of minutes per photo."""
    plan = policy_tiling(WIDTH, HEIGHT, IMGSZ, threshold_px=0, tile=0)
    assert plan is not None
    assert plan.n_tiles <= 20, f"{plan.n_tiles} tiles is unworkable per photo"


def test_forced_and_automatic_keep_a_similar_geometry():
    """Forcing must only skip the DECISION, not change the crop.

    If both routes gave different geometries, the particle count would depend
    on an interface checkbox instead of the photo.
    """
    forced = policy_tiling(WIDTH, HEIGHT, IMGSZ, threshold_px=0, tile=0)
    automatic = policy_tiling(WIDTH, HEIGHT, IMGSZ, threshold_px=2000, tile=0)
    assert automatic is not None
    assert forced is not None
    assert forced.n_tiles == automatic.n_tiles
    assert abs(forced.tile - automatic.tile) <= 0.1 * automatic.tile


def test_a_manual_tile_is_respected():
    plan = policy_tiling(WIDTH, HEIGHT, IMGSZ, threshold_px=0, tile=1280)
    assert plan is not None
    assert plan.tile == 1280


@pytest.mark.parametrize("threshold_px", [0, 2000])
def test_a_small_photo_is_only_tiled_when_forced(threshold_px):
    """Forcing has to force: with threshold 0 no photo is spared."""
    plan = policy_tiling(1200, 900, IMGSZ, threshold_px=threshold_px, tile=0)
    if threshold_px == 0:
        assert plan is not None
    else:
        assert plan is None, "below the threshold it enters whole, untiled"


def test_policy_tiling_is_re_exported_for_existing_importers():
    """Two modules moved it; the old import paths keep resolving to one object."""
    assert protocol_policy_tiling is policy_tiling
    assert yolo_policy_tiling is policy_tiling


# ── One mode, normalised at every boundary ─────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", TilingMode.AUTO),
        ("always", TilingMode.ALWAYS),
        ("never", TilingMode.NEVER),
        (TilingMode.ALWAYS, TilingMode.ALWAYS),
        (" ALWAYS ", TilingMode.ALWAYS),
        ("", TilingMode.AUTO),
        (None, TilingMode.AUTO),
    ],
)
def test_tiling_mode_normalises_boundary_values(raw, expected):
    assert TilingMode.parse(raw) is expected


def test_legacy_tiling_spellings_are_not_modes():
    """`siempre`/`nunca` are gone: parsing them fails instead of translating them."""
    from polyx.core.detect.tiling import TILING_VALUES

    assert TILING_VALUES == ("auto", "always", "never")
    for legacy in ("siempre", "nunca"):
        with pytest.raises(ValueError, match="auto, always, never"):
            TilingMode.parse(legacy)


def test_unknown_mode_warning_names_the_canonical_values():
    """An invalid boundary value is rejected with the canonical values in the error."""
    with pytest.raises(ValueError) as exc:
        TilingMode.parse("sideways")
    assert all(word in str(exc.value) for word in ("auto", "always", "never"))


def test_never_forces_single_pass():
    decision = decide_tiling(WIDTH, HEIGHT, IMGSZ, mode=TilingMode.NEVER, threshold_px=2000, tile=0)
    assert not decision.tiled
    assert decision.plan is None
    assert decision.describe() == ""


def test_always_forces_tiling():
    """Even a small photo, and still at the forced geometry's tile size."""
    decision = decide_tiling(
        WIDTH, HEIGHT, IMGSZ, mode=TilingMode.ALWAYS, threshold_px=2000, tile=0
    )
    assert decision.tiled
    assert decision.plan is not None
    assert decision.plan.tile >= IMGSZ // 2
    small = decide_tiling(1200, 900, IMGSZ, mode=TilingMode.ALWAYS, threshold_px=2000, tile=0)
    assert small.tiled


def test_auto_tiles_only_above_the_threshold():
    above = decide_tiling(WIDTH, HEIGHT, IMGSZ, mode=TilingMode.AUTO, threshold_px=2000)
    below = decide_tiling(1200, 900, IMGSZ, mode=TilingMode.AUTO, threshold_px=2000)
    assert above.tiled and below.tiled is False


# ── One sentence for the plan ──────────────────────────────────────────


def test_describe_tiling_has_one_sentence(selected_language):
    """The engine and both previews share this exact sentence, numbers included."""
    selected_language("en")
    decision = decide_tiling(WIDTH, HEIGHT, IMGSZ, mode=TilingMode.AUTO, threshold_px=2000)
    assert decision.plan == TilingPlan(
        tile=2000, overlap=0.25, n_tiles=6, side=4096, threshold_px=2000
    )
    assert describe_tiling(decision) == "4096×3072 px → 6 tiles of 2000 px, 25% overlap"
    assert decision.describe() == describe_tiling(decision)


def test_a_persisted_plan_describes_itself_in_the_active_language(selected_language):
    """#125: the record keeps numbers, not prose, so the reader phrases them."""
    plan = TilingPlan(tile=2000, overlap=0.25, n_tiles=6, side=4096, threshold_px=2000)

    selected_language("en")
    assert plan.describe() == "4096 px longest side → 6 tiles of 2000 px, 25% overlap"

    selected_language("es")
    assert plan.describe() == "4096 px de lado mayor → 6 mosaicos de 2000 px, 25% de solape"


def test_describe_tiling_is_resolved_when_called(monkeypatch):
    """The description is translated at call time, never frozen at import."""
    from polyx.core.detect import tiling as tiling_module

    decision = decide_tiling(WIDTH, HEIGHT, IMGSZ, mode=TilingMode.AUTO, threshold_px=2000)
    monkeypatch.setattr(tiling_module, "ngettext", lambda singular, plural, n: "TRANSLATED")
    assert describe_tiling(decision) == "TRANSLATED"


def test_the_decision_is_rebuilt_from_the_backend_record():
    plan = TilingPlan(tile=2000, overlap=0.25, n_tiles=6, side=4096, threshold_px=2000)
    record = {"width": 4096, "height": 3072, "tiled": True, "plan": plan}
    decision = TilingDecision.from_record(record, TilingMode.AUTO)
    assert decision == TilingDecision(mode=TilingMode.AUTO, width=4096, height=3072, plan=plan)
    empty = TilingDecision.from_record({}, "never")
    assert empty == TilingDecision(mode=TilingMode.NEVER, width=0, height=0, plan=None)


# ── The previews share the description ─────────────────────────────────


def _wide_photo(path: Path) -> Path:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(bytes(buffer))
    return path


def test_parameters_and_execute_previews_use_the_shared_sentence(qapp, tmp_path):
    """Both pages show the plan the decision module phrased, not their own copy."""
    pytest.importorskip("PySide6")

    from polyx.gui.detector.pages.execute import RunPage
    from polyx.gui.detector.pages.parameters import ParametersPage
    from polyx.gui.detector.state import DetectorState

    photo = _wide_photo(tmp_path / "plate.png")
    state = DetectorState()
    state.images = [photo]
    state.params.imgsz = IMGSZ
    state.params.tiling = TilingMode.AUTO
    state.params.tiling_threshold_px = 2000

    expected = describe_tiling(
        decide_tiling(WIDTH, HEIGHT, IMGSZ, mode=TilingMode.AUTO, threshold_px=2000)
    )
    parameters_page = ParametersPage(state)
    execute_page = RunPage(state)
    execute_page._refresh_warning_tiling()

    assert expected in parameters_page.lbl_plan.text()
    assert expected in execute_page.lbl_tiling.text()


def test_the_combo_stores_english_values_and_translates_only_labels(qapp, tmp_path):
    """The displayed label is translated; the value is the English mode."""
    pytest.importorskip("PySide6")

    from polyx.gui.detector.pages.parameters import ParametersPage
    from polyx.gui.detector.state import DetectorState

    state = DetectorState()
    state.params.tiling = TilingMode.NEVER
    page = ParametersPage(state)

    assert [page.combo_tiling.itemData(index) for index in range(page.combo_tiling.count())] == [
        "auto",
        "always",
        "never",
    ]
    assert page.combo_tiling.currentData() == "never"
