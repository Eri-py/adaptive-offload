"""Real-model-independent pieces of the real-inference module.

`DetectionResult` is the shape a real model's inference on a single frame
produces (real measured latency, real measured accuracy). `score_accuracy`
is the pure IoU-based scoring function that turns a model's predicted boxes
plus a frame's ground-truth boxes into an accuracy value — independent of
which model produced the predictions. A later task adds the real
YOLO-model-running pieces on top of this.
"""

from __future__ import annotations

from typing import NamedTuple

from datagen.simulate import ground_truth


class DetectionResult(NamedTuple):
    """One model's real measured inference result for a single frame."""

    latency_ms: float
    accuracy: float


def score_accuracy(
    predicted_boxes: list[ground_truth.Box],
    ground_truth_boxes: list[ground_truth.Box],
    *,
    iou_threshold: float = 0.5,
) -> float:
    """Score detection accuracy as the fraction of ground-truth boxes matched.

    A ground-truth box counts as matched if `predicted_boxes` contains a
    same-`category_name` box with IoU >= `iou_threshold` against it. Returns
    `1.0` when `ground_truth_boxes` is empty — vacuously nothing was missed,
    and it avoids a division by zero.
    """
    if not ground_truth_boxes:
        return 1.0

    matched = 0
    for gt_box in ground_truth_boxes:
        if any(
            pred_box.category_name == gt_box.category_name
            and _iou(pred_box, gt_box) >= iou_threshold
            for pred_box in predicted_boxes
        ):
            matched += 1

    return matched / len(ground_truth_boxes)


def _iou(a: ground_truth.Box, b: ground_truth.Box) -> float:
    """Intersection-over-union of two axis-aligned boxes; `0.0` if disjoint."""
    intersection_x_min = max(a.x_min, b.x_min)
    intersection_y_min = max(a.y_min, b.y_min)
    intersection_x_max = min(a.x_max, b.x_max)
    intersection_y_max = min(a.y_max, b.y_max)

    intersection_width = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    intersection_area = intersection_width * intersection_height

    if intersection_area == 0.0:
        return 0.0

    a_area = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    b_area = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union_area = a_area + b_area - intersection_area

    return intersection_area / union_area
