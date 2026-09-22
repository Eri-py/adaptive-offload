"""COCO-format ground-truth box parsing, for accuracy scoring.

Reads a COCO-format annotations file's `annotations`/`categories` arrays
(distinct from `image_source.load_image_index`'s `images` array, though
it's the same underlying file) and resolves them into per-image ground-truth
boxes, keyed by `image_id`. This is a pure parsing concern: no network
access, no inference — just turning the annotations file's raw shape into
the `Box` shape the rest of `simulate/` scores predictions against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple


class Box(NamedTuple):
    """A labeled axis-aligned box: a category name plus corner coordinates.

    Used for both ground-truth boxes (this module) and predicted boxes (a
    later task's real-model output) — structurally identical, both are just
    a labeled box.
    """

    category_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def load_ground_truth(annotations_path: Path) -> dict[int, list[Box]]:
    """Read the COCO-format `annotations`/`categories` arrays into per-image ground truth.

    Returns a `dict` mapping `image_id` to that image's list of ground-truth
    `Box`es. An `image_id` with no matching `annotations` entries is simply
    absent from the returned dict — the caller treats a missing key as "zero
    ground-truth boxes for this frame".

    Raises `FileNotFoundError` if `annotations_path` doesn't exist, or
    `ValueError` if it exists but isn't valid JSON or doesn't have
    `"annotations"`/`"categories"` keys holding lists.
    """
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations file not found at {annotations_path}. "
            "Check the path or generate/download the annotations file first."
        )

    with annotations_path.open("r", encoding="utf-8") as f:
        try:
            data: Any = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Annotations file at {annotations_path} is not valid JSON: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Annotations file at {annotations_path} does not contain a JSON object at "
            f"the top level — expected a COCO-format annotations file, got "
            f"{type(data).__name__}."
        )

    if "categories" not in data:
        raise ValueError(
            f"Annotations file at {annotations_path} has no \"categories\" key — "
            "expected a COCO-format annotations file."
        )

    categories = data["categories"]
    if not isinstance(categories, list):
        raise ValueError(
            f"Annotations file at {annotations_path} has a \"categories\" value that is not "
            f"a list — expected a COCO-format annotations file, got {type(categories).__name__}."
        )

    if "annotations" not in data:
        raise ValueError(
            f"Annotations file at {annotations_path} has no \"annotations\" key — "
            "expected a COCO-format annotations file."
        )

    annotations = data["annotations"]
    if not isinstance(annotations, list):
        raise ValueError(
            f"Annotations file at {annotations_path} has an \"annotations\" value that is "
            f"not a list — expected a COCO-format annotations file, "
            f"got {type(annotations).__name__}."
        )

    category_names: dict[int, str] = {entry["id"]: entry["name"] for entry in categories}

    ground_truth: dict[int, list[Box]] = {}
    for entry in annotations:
        image_id = entry["image_id"]
        x, y, width, height = entry["bbox"]
        box = Box(
            category_name=category_names[entry["category_id"]],
            x_min=x,
            y_min=y,
            x_max=x + width,
            y_max=y + height,
        )
        ground_truth.setdefault(image_id, []).append(box)

    return ground_truth
