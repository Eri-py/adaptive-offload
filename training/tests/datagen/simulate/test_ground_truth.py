"""Tests for `datagen.simulate.ground_truth`.

Uses small fake annotation fixtures under `tmp_path` throughout — never
touches the real `training/data/coco/` directory or the network.
"""

import json
from pathlib import Path

import pytest

from datagen.simulate.ground_truth import Box, load_ground_truth

_FAKE_CATEGORIES = [
    {"supercategory": "animal", "id": 1, "name": "cat"},
    {"supercategory": "vehicle", "id": 2, "name": "car"},
]

_FAKE_ANNOTATIONS = [
    # image 1: one "cat" box.
    {
        "id": 100,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "area": 1200.0,
        "iscrowd": 0,
        "segmentation": [],
    },
    # image 2: one "cat" box and one "car" box.
    {
        "id": 101,
        "image_id": 2,
        "category_id": 1,
        "bbox": [0.0, 0.0, 5.0, 5.0],
        "area": 25.0,
        "iscrowd": 0,
        "segmentation": [],
    },
    {
        "id": 102,
        "image_id": 2,
        "category_id": 2,
        "bbox": [50.0, 60.0, 10.0, 5.0],
        "area": 50.0,
        "iscrowd": 0,
        "segmentation": [],
    },
    # image 3 has no annotations entries at all.
]


def _write_fake_annotations(tmp_path: Path) -> Path:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(
        json.dumps({"categories": _FAKE_CATEGORIES, "annotations": _FAKE_ANNOTATIONS})
    )
    return annotations_path


def test_load_ground_truth_round_trips_multi_image_multi_annotation_fixture(
    tmp_path: Path,
) -> None:
    ground_truth = load_ground_truth(_write_fake_annotations(tmp_path))

    assert ground_truth[1] == [
        Box(category_name="cat", x_min=10.0, y_min=20.0, x_max=40.0, y_max=60.0)
    ]
    assert ground_truth[2] == [
        Box(category_name="cat", x_min=0.0, y_min=0.0, x_max=5.0, y_max=5.0),
        Box(category_name="car", x_min=50.0, y_min=60.0, x_max=60.0, y_max=65.0),
    ]


def test_load_ground_truth_excludes_crowd_annotations(tmp_path: Path) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(
        json.dumps(
            {
                "categories": _FAKE_CATEGORIES,
                "annotations": [
                    # image 4: one crowd "cat" box (excluded) and one regular
                    # "car" box (kept).
                    {
                        "id": 200,
                        "image_id": 4,
                        "category_id": 1,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "area": 12.0,
                        "iscrowd": 1,
                        "segmentation": [],
                    },
                    {
                        "id": 201,
                        "image_id": 4,
                        "category_id": 2,
                        "bbox": [5.0, 6.0, 7.0, 8.0],
                        "area": 56.0,
                        "iscrowd": 0,
                        "segmentation": [],
                    },
                ],
            }
        )
    )

    ground_truth = load_ground_truth(annotations_path)

    assert ground_truth[4] == [
        Box(category_name="car", x_min=5.0, y_min=6.0, x_max=12.0, y_max=14.0)
    ]


def test_load_ground_truth_omits_image_id_with_no_annotations(tmp_path: Path) -> None:
    ground_truth = load_ground_truth(_write_fake_annotations(tmp_path))

    assert 3 not in ground_truth


def test_load_ground_truth_raises_file_not_found_for_missing_annotations(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError, match="does_not_exist.json"):
        load_ground_truth(annotations_path)


def test_load_ground_truth_raises_value_error_for_invalid_json(tmp_path: Path) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text("not valid json {")

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_ground_truth(annotations_path)


def test_load_ground_truth_raises_value_error_when_annotations_key_missing(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(json.dumps({"categories": _FAKE_CATEGORIES}))

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_ground_truth(annotations_path)


def test_load_ground_truth_raises_value_error_when_categories_key_missing(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(json.dumps({"annotations": _FAKE_ANNOTATIONS}))

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_ground_truth(annotations_path)


def test_load_ground_truth_raises_value_error_when_top_level_is_not_object(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(json.dumps(42))

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_ground_truth(annotations_path)


def test_load_ground_truth_raises_value_error_when_annotations_value_is_not_list(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(
        json.dumps({"categories": _FAKE_CATEGORIES, "annotations": "nope"})
    )

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_ground_truth(annotations_path)


def test_load_ground_truth_raises_value_error_when_categories_value_is_not_list(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(
        json.dumps({"categories": "nope", "annotations": _FAKE_ANNOTATIONS})
    )

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_ground_truth(annotations_path)
