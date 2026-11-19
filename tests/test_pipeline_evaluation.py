"""The shared evaluator separates ranking (AP) from operating-point F1."""

from __future__ import annotations

import pytest

from polyx.pipeline.evaluation import SharedEvaluator, rasterize_segmentation


def _ground_truth(count: int = 2) -> dict:
    return {
        "images": [{"id": 0, "width": 100, "height": 100}],
        "categories": [{"id": 0, "name": "PET"}],
        "annotations": [
            {
                "id": index,
                "image_id": 0,
                "category_id": 0,
                "bbox": [index * 40, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
            for index in range(count)
        ],
    }


def _prediction(box: list[float], score: float, category: int = 0) -> dict:
    return {"image_id": 0, "category_id": category, "bbox": box, "score": score}


def test_below_threshold_predictions_do_not_change_f1() -> None:
    """AP keeps the ranked tail; the operating point applies its threshold."""
    evaluator = SharedEvaluator(_ground_truth(), split="val")
    clean = [_prediction([0, 10, 20, 20], 0.9), _prediction([40, 10, 20, 20], 0.9)]
    good = evaluator.evaluate(clean)
    noisy = evaluator.evaluate(clean + [_prediction([5, 50, 10, 10], 0.1) for _ in range(10)])
    assert good["selection"]["value"] == 1.0
    assert noisy["selection"]["value"] == 1.0
    assert noisy["selection"]["criterion_name"] == "with_class"
    assert noisy["selection"]["confidence_threshold"] == 0.25
    assert noisy["selection"]["iou_threshold"] == 0.5


def test_above_threshold_false_positives_lower_f1() -> None:
    """The threshold is applied, not ignored: confident errors count."""
    evaluator = SharedEvaluator(_ground_truth(), split="val")
    noisy = evaluator.evaluate(
        [_prediction([0, 10, 20, 20], 0.9), _prediction([40, 10, 20, 20], 0.9)]
        + [_prediction([5, 50, 10, 10], 0.9) for _ in range(10)]
    )
    assert noisy["selection"]["value"] < 0.3


def test_ap_and_f1_can_rank_models_differently() -> None:
    """A confident model wins AP; a precise-at-0.5 model wins operating F1."""
    evaluator = SharedEvaluator(_ground_truth(), split="val")
    precise_at_05 = evaluator.evaluate(
        [_prediction([4, 10, 20, 20], 0.9), _prediction([44, 10, 20, 20], 0.9)]
    )
    confident_ap = evaluator.evaluate(
        [_prediction([0, 10, 20, 20], 0.9), _prediction([40, 10, 20, 20], 0.9)]
        + [_prediction([5, 50, 10, 10], 0.9) for _ in range(5)]
    )
    assert precise_at_05["selection"]["value"] > confident_ap["selection"]["value"]
    assert (
        precise_at_05["figures"]["mAP50-95"]["value"] < confident_ap["figures"]["mAP50-95"]["value"]
    )


def test_coco_ap_interpolates_precision_across_recall() -> None:
    """TP, FP, TP interpolates to ~0.835, not the old sum-at-hits 0.583."""
    evaluator = SharedEvaluator(_ground_truth(), split="val")
    result = evaluator.evaluate(
        [
            _prediction([0, 10, 20, 20], 0.9),
            _prediction([20, 10, 20, 20], 0.8),
            _prediction([40, 10, 20, 20], 0.7),
        ]
    )
    ap = result["figures"]["mAP50"]["value"]
    assert 0.83 < ap < 0.84
    assert ap > (1.0 + 2 / 3) / 3  # the old sum-at-hits arithmetic
    assert result["selection"]["value"] < 1.0


def test_coco_ap_caps_the_first_true_positive_at_rank_101() -> None:
    """One hundred higher-scored false positives push the TP out of COCO AP."""
    evaluator = SharedEvaluator(_ground_truth(1), split="val")
    result = evaluator.evaluate(
        [_prediction([5, 50, 10, 10], 0.9) for _ in range(100)]
        + [_prediction([0, 10, 20, 20], 0.1)]
    )
    assert result["figures"]["mAP50"]["value"] == 0.0


def test_coco_ap_caps_per_image_not_across_the_batch() -> None:
    """A crowded plate cannot hide the true positive on another image."""
    ground_truth = _ground_truth(1)
    ground_truth["images"].append({"id": 1, "width": 100, "height": 100})
    ground_truth["annotations"].append(
        {
            "id": 99,
            "image_id": 1,
            "category_id": 0,
            "bbox": [0, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        }
    )
    evaluator = SharedEvaluator(ground_truth, split="val")
    predictions = [_prediction([5, 50, 10, 10], 0.9) for _ in range(100)] + [
        {"image_id": 1, "category_id": 0, "bbox": [0, 10, 20, 20], "score": 0.9}
    ]
    result = evaluator.evaluate(predictions)
    assert result["figures"]["mAP50"]["value"] > 0.0


def test_wrong_polymer_is_not_a_true_positive() -> None:
    """The with-class criterion reuses the core misclassification rule.

    A well-placed box of the wrong polymer is a misclassification; under the
    core definition it stays out of both the with-class and the localisation
    numerators, so neither criterion can claim it as a true positive.
    """
    ground_truth = _ground_truth(1)
    ground_truth["categories"].append({"id": 1, "name": "PP"})
    evaluator = SharedEvaluator(ground_truth, split="val")
    result = evaluator.evaluate([_prediction([0, 10, 20, 20], 0.9, category=1)])
    assert result["selection"]["value"] == 0.0
    assert result["figures"]["localisation_score"]["value"] == 0.0


def test_the_detection_cap_is_applied_per_image_and_category() -> None:
    """A crowded polymer cannot push another class out of the ranking."""
    ground_truth = _ground_truth(1)
    ground_truth["categories"].append({"id": 1, "name": "PP"})
    evaluator = SharedEvaluator(ground_truth, split="val")
    crowded = [_prediction([0, 10, 20, 20], 0.99, category=1) for _ in range(150)]
    result = evaluator.evaluate([*crowded, _prediction([0, 10, 20, 20], 0.5)])
    assert result["figures"]["mAP50"]["value"] == pytest.approx(1.0), (
        "the PET true positive is ranked behind 150 PP detections; a per-image "
        "cap would drop it, a per-image-and-category cap keeps it"
    )


def test_a_subpixel_shift_does_not_empty_a_coco_polygon() -> None:
    """COCO maps both squares to the same pixel; a rounded fill does not."""
    square = [[2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0]]
    shifted = [[2.4, 2.4, 3.4, 2.4, 3.4, 3.4, 2.4, 3.4]]
    first = rasterize_segmentation(square, 16, 16)
    second = rasterize_segmentation(shifted, 16, 16)
    assert first is not None and second is not None
    assert first[2, 2] and second[2, 2]
    assert (first == second).all()


def test_mask_iou_uses_cocos_polygon_rasterization() -> None:
    """A shifted polygon is the same mask, so the predicted hit is a true one."""
    ground_truth = _ground_truth(1)
    ground_truth["annotations"][0]["segmentation"] = [
        [0.0, 10.0, 20.0, 10.0, 20.0, 30.0, 0.0, 30.0]
    ]
    evaluator = SharedEvaluator(ground_truth, split="val", task="segment")
    prediction = _prediction([0, 10, 20, 20], 0.9)
    prediction["segmentation"] = [[0.4, 10.4, 20.4, 10.4, 20.4, 30.4, 0.4, 30.4]]
    result = evaluator.evaluate([prediction])
    assert result["selection"]["value"] == pytest.approx(1.0)
