"""COCO-format annotation loading and local image-file resolution.

Reads a COCO-format annotations file to get the pool of (image_id,
file_name) pairs, and resolves each image's local file path — a pure local
lookup that fails clearly if the file isn't already present under the given
images directory. This works for any COCO-format dataset, not just COCO
val2017.

This module never fetches anything over the network: it assumes the caller
already has the annotations file and every image file present locally, and
fails loudly (rather than trying to "top up" missing files) if either isn't
there. The main simulator pipeline runs against exactly what it's given —
acquiring the data is the caller's responsibility, not this module's.

Every function here takes its path as an explicit, required parameter —
there is no default dataset, annotations path, or images path anymore; the
caller (the `--annotations`/`--images` CLI flags, or a test) is always the
one who supplies them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple


class ImageRecord(NamedTuple):
    """One image's id and file name, from a COCO-format annotations file."""

    image_id: int
    file_name: str


def load_image_index(annotations_path: Path) -> list[ImageRecord]:
    """Read the COCO-format `images` array and return (image_id, file_name) pairs.

    Raises `FileNotFoundError` if `annotations_path` doesn't exist, or
    `ValueError` if it exists but isn't valid JSON or doesn't have an
    `"images"` key.
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

    if "images" not in data:
        raise ValueError(
            f"Annotations file at {annotations_path} has no \"images\" key — "
            "expected a COCO-format annotations file."
        )

    images = data["images"]
    if not isinstance(images, list):
        raise ValueError(
            f"Annotations file at {annotations_path} has an \"images\" value that is not "
            f"a list — expected a COCO-format annotations file, got {type(images).__name__}."
        )

    records = []
    for index, entry in enumerate(images):
        if (
            not isinstance(entry, dict)
            or "id" not in entry
            or "file_name" not in entry
        ):
            raise ValueError(
                f"Annotations file at {annotations_path} has a malformed entry at "
                f"images[{index}] — expected an object with \"id\" and \"file_name\" keys, "
                f"got {entry!r}."
            )
        records.append(ImageRecord(image_id=entry["id"], file_name=entry["file_name"]))

    return records


def resolve_image_path(file_name: str, *, images_dir: Path) -> Path:
    """Return the local path to `file_name`.

    Pure local lookup, no network access: `file_name` is expected to already
    exist under `images_dir` (the dataset is provided ahead of time). Raises
    `FileNotFoundError` if it isn't there.
    """
    local_path = images_dir / file_name
    if not local_path.exists():
        raise FileNotFoundError(f"Image {file_name!r} not found under {images_dir}.")
    return local_path
