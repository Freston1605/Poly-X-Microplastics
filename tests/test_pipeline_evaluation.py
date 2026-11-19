"""The shared evaluator separates ranking (AP) from operating-point F1."""

from __future__ import annotations

from polyx.pipeline.evaluation import SharedEvaluator


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


def test_with_class_selection_is_operating_point_f1_not_map() -> None:
    """Late low-score false positives barely move AP but wreck fixed F1."""
    evaluator = SharedEvaluator(_ground_truth(), split="val")
    good = evaluator.evaluate(
        [_prediction([0, 10, 20, 20], 0.9), _prediction([40, 10, 20, 20], 0.9)]
    )
    noisy = evaluator.evaluate(
        [_prediction([0, 10, 20, 20], 0.9), _prediction([40, 10, 20, 20], 0.9)]
        + [_prediction([5, 50, 10, 10], 0.1) for _ in range(10)]
    )
    # The ranking metric stays high; the fixed-threshold score collapses.
    assert noisy["figures"]["mAP50-95"]["value"] > 0.9
    assert good["selection"]["value"] == 1.0
    assert noisy["selection"]["value"] < 0.3
    assert noisy["selection"]["criterion_name"] == "with_class"
    assert noisy["selection"]["confidence_threshold"] == 0.25
    assert noisy["selection"]["iou_threshold"] == 0.5


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
