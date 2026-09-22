"""Tests for `datagen.simulate.inference`'s pure accuracy-scoring pieces."""

from datagen.simulate.ground_truth import Box
from datagen.simulate.inference import score_accuracy

_CAT_BOX = Box(category_name="cat", x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
_CAT_BOX_HIGH_IOU = Box(category_name="cat", x_min=1.0, y_min=1.0, x_max=11.0, y_max=11.0)
_CAR_BOX_SAME_POSITION = Box(category_name="car", x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
_DOG_BOX = Box(category_name="dog", x_min=100.0, y_min=100.0, x_max=110.0, y_max=110.0)
_DOG_BOX_HIGH_IOU = Box(
    category_name="dog", x_min=101.0, y_min=101.0, x_max=111.0, y_max=111.0
)


def test_score_accuracy_perfect_match_scores_one() -> None:
    ground_truth_boxes = [_CAT_BOX, _DOG_BOX]
    predicted_boxes = [_CAT_BOX_HIGH_IOU, _DOG_BOX_HIGH_IOU]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 1.0


def test_score_accuracy_no_matching_predictions_scores_zero() -> None:
    ground_truth_boxes = [_CAT_BOX, _DOG_BOX]
    predicted_boxes: list[Box] = []

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 0.0


def test_score_accuracy_partial_match_scores_correct_fraction() -> None:
    ground_truth_boxes = [_CAT_BOX, _DOG_BOX]
    predicted_boxes = [_CAT_BOX_HIGH_IOU]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 0.5


def test_score_accuracy_same_position_different_category_does_not_match() -> None:
    ground_truth_boxes = [_CAT_BOX]
    predicted_boxes = [_CAR_BOX_SAME_POSITION]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 0.0


def test_score_accuracy_zero_ground_truth_boxes_scores_one() -> None:
    ground_truth_boxes: list[Box] = []
    predicted_boxes = [_CAT_BOX]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 1.0


def test_score_accuracy_below_iou_threshold_does_not_match() -> None:
    # IoU here is 25 / 175 ≈ 0.14 — below the default 0.5 threshold.
    ground_truth_boxes = [_CAT_BOX]
    slightly_overlapping = Box(category_name="cat", x_min=5.0, y_min=5.0, x_max=15.0, y_max=15.0)

    assert score_accuracy([slightly_overlapping], ground_truth_boxes) == 0.0


def test_score_accuracy_respects_custom_iou_threshold() -> None:
    # Same ~0.14 IoU box as above, but now the threshold is low enough to match.
    ground_truth_boxes = [_CAT_BOX]
    slightly_overlapping = Box(category_name="cat", x_min=5.0, y_min=5.0, x_max=15.0, y_max=15.0)

    assert (
        score_accuracy([slightly_overlapping], ground_truth_boxes, iou_threshold=0.1) == 1.0
    )
