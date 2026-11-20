"""Packed masks are exact, and the evaluator scores them like plain arrays."""

from __future__ import annotations

import numpy as np
import pytest

from polyx.pipeline.masks import PackedMask, as_mask

NAMES = {0: "PET"}


def _ground_truth() -> dict:
    return {
        "images": [{"id": 0, "width": 64, "height": 48}],
        "categories": [{"id": 0, "name": "PET"}],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "bbox": [8, 6, 24, 18],
                "area": 432,
                "iscrowd": 0,
                "segmentation": [[8, 6, 32, 6, 32, 24, 8, 24]],
            }
        ],
    }


def _prediction(mask: np.ndarray) -> list[dict]:
    return [
        {
            "image_id": 0,
            "category_id": 0,
            "bbox": [8, 6, 24, 18],
            "score": 0.9,
            "mask": mask,
        }
    ]


def _packed_prediction(mask: np.ndarray) -> list[dict]:
    prediction = _prediction(mask)
    prediction[0].pop("mask")
    prediction[0]["packed_mask"] = PackedMask.pack(mask)
    return prediction


def test_packing_round_trips_every_pixel_and_shrinks_the_mask() -> None:
    rng = np.random.default_rng(20261119)
    mask = rng.random((48, 64)) >= 0.5

    packed = PackedMask.pack(mask)

    assert packed.shape == (48, 64)
    assert np.array_equal(packed.unpack(), mask)
    assert packed.bits.nbytes * 8 <= mask.nbytes + 7, "a packed mask is not bit-dense"
    assert packed.any() == mask.any()


def test_packing_accepts_probabilities_through_the_same_threshold() -> None:
    soft = np.zeros((4, 4), dtype=float)
    soft[1, 2] = 0.75
    soft[3, 3] = 0.25

    assert PackedMask.pack(soft).unpack().sum() == 1


def test_as_mask_accepts_both_representations() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True

    assert np.array_equal(as_mask(mask), mask)
    assert np.array_equal(as_mask(PackedMask.pack(mask)), mask)
    assert as_mask(None) is None


def test_the_evaluator_scores_a_packed_mask_exactly_like_an_array() -> None:
    """Compression must not move a single figure, or the benchmark changes."""
    from polyx.pipeline.evaluation import SharedEvaluator, rasterize_segmentation

    width, height = 64, 48
    perfect = rasterize_segmentation([[8, 6, 32, 6, 32, 24, 8, 24]], width, height)
    assert perfect is not None
    shifted = np.roll(perfect, 2, axis=0)

    array_evaluator = SharedEvaluator(_ground_truth(), split="val", task="segment")
    packed_evaluator = SharedEvaluator(_ground_truth(), split="val", task="segment")

    for mask in (perfect, shifted):
        as_array = array_evaluator.evaluate(_prediction(mask))
        as_packed = packed_evaluator.evaluate(_packed_prediction(mask))
        assert as_packed == as_array, "the packed path changed a scored figure"

    assert array_evaluator.evaluate(_prediction(perfect))["selection"]["value"] == pytest.approx(
        1.0
    )
