"""COCO val2017 annotation loading and local image-file caching.

Reads the pre-downloaded COCO val2017 annotations file to get the pool of
(image_id, file_name) pairs, and resolves each image's local file path —
treating the pre-downloaded files under `training/data/coco/val2017/` as the
common case, not the exception. The download path (for a genuinely missing
image, e.g. a fresh checkout without the pre-downloaded dataset) goes through
an injectable fetch callable so it never has to touch the network in tests.

This module caches image *bytes* only — it does not store any computed
values (e.g. scene complexity), which live in Postgres per the feature spec
(see the dedicated scene-complexity table, Task 6).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import requests

# training/data/coco/ — datagen.py lives at training/datagen/coco.py, so the
# data directory is a sibling of the datagen package, one level up.
COCO_DIR = Path(__file__).resolve().parent.parent / "data" / "coco"
ANNOTATIONS_PATH = COCO_DIR / "annotations" / "instances_val2017.json"
IMAGES_DIR = COCO_DIR / "val2017"

COCO_VAL2017_BASE_URL = "https://images.cocodataset.org/val2017"

# A fetch callable takes an image URL and returns its raw bytes. The default
# (`fetch_image_bytes`) makes a real HTTP GET; tests inject a fake instead.
FetchFn = Callable[[str], bytes]


class ImageRecord(NamedTuple):
    """One COCO val2017 image's id and file name, from the annotations file."""

    image_id: int
    file_name: str


def load_image_index(annotations_path: Path = ANNOTATIONS_PATH) -> list[ImageRecord]:
    """Read the COCO `images` array and return (image_id, file_name) pairs."""
    with annotations_path.open("r", encoding="utf-8") as f:
        data: dict[str, list[Any]] = json.load(f)
    return [
        ImageRecord(image_id=entry["id"], file_name=entry["file_name"]) for entry in data["images"]
    ]


def fetch_image_bytes(url: str) -> bytes:
    """Default fetch: a real HTTP GET. Injected/mocked away in tests."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def resolve_image_path(
    file_name: str,
    *,
    images_dir: Path = IMAGES_DIR,
    base_url: str = COCO_VAL2017_BASE_URL,
    fetch: FetchFn = fetch_image_bytes,
) -> Path:
    """Return the local path to `file_name`, downloading it only if missing.

    The common case is that `file_name` already exists under `images_dir`
    (the dataset is pre-downloaded) and this returns immediately with no
    network call at all. If it's genuinely missing, `fetch` retrieves the
    bytes from `{base_url}/{file_name}` and caches them at that same local
    path before returning it, so a later call finds it cached.
    """
    local_path = images_dir / file_name
    if local_path.exists():
        return local_path

    images_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = fetch(f"{base_url}/{file_name}")
    # Write to a sibling `.part` path first and atomically rename into place
    # only once the write has fully completed. Writing `local_path` directly
    # would leave a truncated file cached forever if the process is
    # interrupted mid-write (e.g. on a slow/unreliable connection) — a later
    # call would treat that truncated file as a valid cache hit (the
    # `local_path.exists()` check above) and never re-fetch it.
    part_path = local_path.with_suffix(local_path.suffix + ".part")
    part_path.write_bytes(image_bytes)
    part_path.replace(local_path)
    return local_path
