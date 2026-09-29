"""Tests for the detection-confidence router features.

Avoids `import pytest` (per `training/tests/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv). Plain
`def test_...()` functions are enough for pytest to collect and run these.
"""

from __future__ import annotations

import numpy as np
import torch
from ultralytics.engine.results import Results

from coco.router.frame_features.confidence import confidence_features, from_results


def test_known_confidences_give_known_max_mean_min() -> None:
    features = confidence_features(confidences=[0.9, 0.3, 0.6], box_areas=[0.1, 0.2, 0.3])
    assert features.detection_count == 3
    assert features.max_confidence == 0.9
    assert features.mean_confidence == 0.6
    assert features.min_confidence == 0.3


def test_known_box_areas_give_known_mean_and_min() -> None:
    features = confidence_features(confidences=[0.5, 0.5], box_areas=[0.02, 0.25])
    assert features.mean_box_area == 0.135
    assert features.min_box_area == 0.02


def test_zero_detections_gives_all_zeros() -> None:
    features = confidence_features(confidences=[], box_areas=[])
    assert features.detection_count == 0
    assert features.max_confidence == 0.0
    assert features.mean_confidence == 0.0
    assert features.min_confidence == 0.0
    assert features.mean_box_area == 0.0
    assert features.min_box_area == 0.0


def test_mismatched_lengths_raise() -> None:
    try:
        confidence_features(confidences=[0.5], box_areas=[])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched sequence lengths")


def _results_with_boxes(boxes: torch.Tensor, size: int = 100) -> Results:
    """Build a `Results` with a blank `size`x`size` image and the given `xyxy` boxes.

    `boxes` rows are `[x1, y1, x2, y2, conf, cls]` in pixel coordinates, the
    same shape `ultralytics.engine.results.Boxes` expects (per its docstring).
    """
    orig_img = np.zeros((size, size, 3), dtype=np.uint8)
    return Results(orig_img=orig_img, path="synthetic.jpg", names={0: "object"}, boxes=boxes)


def test_from_results_extracts_known_confidences_and_normalized_box_areas() -> None:
    # Box A: 20x10 px -> normalised 0.2 x 0.1 = area 0.02. Box B: 50x50 px -> area 0.25.
    boxes = torch.tensor(
        [
            [0.0, 0.0, 20.0, 10.0, 0.9, 0.0],
            [0.0, 0.0, 50.0, 50.0, 0.3, 0.0],
        ]
    )
    features = from_results(_results_with_boxes(boxes))
    assert features.detection_count == 2
    # float32's ~1e-7 rounding, not this adapter, would make an exact `==` flaky here.
    assert round(features.max_confidence, 3) == 0.9
    assert round(features.min_confidence, 3) == 0.3
    assert round(features.mean_box_area, 3) == 0.135
    assert round(features.min_box_area, 3) == 0.02


def test_from_results_with_no_detections_gives_all_zeros() -> None:
    features = from_results(_results_with_boxes(torch.zeros((0, 6))))
    assert features.detection_count == 0
    assert features.max_confidence == 0.0
    assert features.mean_confidence == 0.0
    assert features.min_confidence == 0.0
    assert features.mean_box_area == 0.0
    assert features.min_box_area == 0.0
