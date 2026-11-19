"""Ultralytics results are converted to COCO without touching CUDA in NumPy.

A GPU validation pass must not depend on where the framework keeps its
tensors. A tensor double that refuses direct NumPy conversion proves the
converters detach and move every array to CPU first, including empty boxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyx.pipeline.coco import (
    coco_predictions_from_results,
    coco_segmentation_predictions_from_results,
)


class _CudaTensor:
    """A tensor that can only reach NumPy after ``detach().cpu()``."""

    def __init__(self, values: object, dtype: type = float) -> None:
        self.values = np.asarray(values, dtype=dtype)

    def detach(self) -> _CudaTensor:
        """Return a detached view."""
        return self

    def cpu(self) -> _CudaTensor:
        """Return the host copy."""
        return self

    def numpy(self) -> np.ndarray:
        """The only supported route to NumPy."""
        return self.values

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("a CUDA tensor was handed directly to NumPy")


class _Boxes:
    def __init__(self, xyxy: object, conf: object, cls: object) -> None:
        self.xyxy = _CudaTensor(xyxy)
        self.conf = _CudaTensor(conf)
        self.cls = _CudaTensor(cls, dtype=int)


class _Masks:
    def __init__(self, polygons: list[object]) -> None:
        self.xy = [_CudaTensor(polygon) for polygon in polygons]


class _Result:
    def __init__(self, boxes: _Boxes, masks: _Masks | None = None) -> None:
        self.boxes = boxes
        self.masks = masks


def test_detection_conversion_detaches_cuda_tensors() -> None:
    """Boxes, confidences and class ids all go through the CPU converter."""
    result = _Result(
        _Boxes([[10, 10, 30, 30]], [0.9], [1]),
    )
    predictions = coco_predictions_from_results([result])
    assert predictions == [
        {"image_id": 0, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9}
    ]


def test_empty_cuda_boxes_convert_without_an_array_error() -> None:
    """Even an empty detection result holds CUDA tensors in the pinned stack."""
    result = _Result(_Boxes(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)))
    assert coco_predictions_from_results([result]) == []


def test_segmentation_conversion_detaches_boxes_and_polygons() -> None:
    """Mask polygons can be CUDA tensors too."""
    polygon = [[10, 10], [30, 10], [30, 30], [10, 30]]
    result = _Result(_Boxes([[10, 10, 30, 30]], [0.8], [2]), _Masks([polygon]))
    predictions = coco_segmentation_predictions_from_results([result])
    assert predictions[0]["category_id"] == 2
    assert predictions[0]["segmentation"] == [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]]


@pytest.mark.parametrize("converter", [coco_predictions_from_results])
def test_converter_accepts_no_results(converter: object) -> None:
    """A batch with no images produces no predictions."""
    assert converter([]) == []
