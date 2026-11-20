"""Mask R-CNN validation must not materialize a split of 4K masks at once."""

from __future__ import annotations

import types

import numpy as np
import torch

from polyx.pipeline.adapters import MASK_PASTE_CHUNK, _install_chunked_mask_postprocess
from polyx.pipeline.masks import PackedMask


class _PastingTransform:
    """torchvision's transform, reduced to its paste step and its call sizes."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.sizes: list[int] = []

    def postprocess(
        self,
        detections: list[dict],
        image_sizes: list[tuple[int, int]],
        original_sizes: list[tuple[int, int]],
    ) -> list[dict]:
        detection = detections[0]
        count = len(detection["boxes"])
        self.sizes.append(count)
        masks = torch.zeros((count, 1, self.height, self.width))
        masks[:, 0, 2:10, 3:11] = 1.0
        return [
            {
                "boxes": detection["boxes"],
                "labels": detection["labels"],
                "scores": detection["scores"],
                "masks": masks,
            }
        ]


def _detection(count: int) -> dict:
    return {
        "boxes": torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
        "labels": torch.arange(1, count + 1),
        "scores": torch.linspace(0.9, 0.1, count),
        "masks": torch.zeros((count, 1, 28, 28)),
    }


def test_the_paste_is_bounded_to_the_chunk_and_keeps_the_order() -> None:
    height, width = 30, 40
    transform = _PastingTransform(height, width)
    model = types.SimpleNamespace(transform=transform)
    _install_chunked_mask_postprocess(model)

    count = MASK_PASTE_CHUNK * 2 + 3
    detection = _detection(count)
    pasted = model.transform.postprocess([detection], [(height, width)], [(height, width)])[0]

    # No single call asked for more masks than the bound; every chunk was used.
    assert transform.sizes == [MASK_PASTE_CHUNK, MASK_PASTE_CHUNK, count % MASK_PASTE_CHUNK]
    assert max(transform.sizes) <= MASK_PASTE_CHUNK

    masks = pasted["masks"]
    assert all(isinstance(mask, PackedMask) for mask in masks), "a full-resolution tensor escaped"
    assert len(masks) == count
    assert masks[0].shape == (height, width)
    expected = np.zeros((height, width), dtype=bool)
    expected[2:10, 3:11] = True
    assert np.array_equal(masks[0].unpack(), expected)

    # Boxes, labels and scores keep their rank order across the chunks.
    assert torch.equal(pasted["boxes"], detection["boxes"])
    assert torch.equal(pasted["labels"], detection["labels"])
    assert torch.allclose(pasted["scores"], detection["scores"])


def test_an_image_with_no_detections_pastes_nothing() -> None:
    height, width = 12, 16
    transform = _PastingTransform(height, width)
    model = types.SimpleNamespace(transform=transform)
    _install_chunked_mask_postprocess(model)

    pasted = model.transform.postprocess([_detection(0)], [(height, width)], [(height, width)])[0]

    assert transform.sizes == []
    assert pasted["masks"] == []


def test_a_stand_in_without_a_transform_is_left_alone() -> None:
    model = types.SimpleNamespace()

    _install_chunked_mask_postprocess(model)

    assert not hasattr(model, "transform")
