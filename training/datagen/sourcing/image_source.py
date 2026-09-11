"""COCO val2017 annotation loading and local image-file lookup.

Reads the pre-downloaded COCO val2017 annotations file to get the pool of
(image_id, file_name) pairs, and resolves each image's local file path — a
pure local lookup that fails clearly if the file isn't already cached under
`training/data/coco/val2017/`. Populating that cache (downloading whatever's
missing) is a separate, explicitly-run step: see `download_missing_images`
below and `python -m datagen.cli.sync_coco_cache`. Keeping acquisition out of
`resolve_image_path` means the main simulator pipeline and the
complexity-scoring entry point only ever read local files and fail loudly on
a miss, rather than silently reaching out to the network mid-run.

This module caches image *bytes* only — it does not store any computed
values (e.g. scene complexity), which live in Postgres per the feature spec
(see the dedicated scene-complexity table).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import requests

# training/data/coco/ — this file lives at training/datagen/sourcing/image_source.py,
# so the data directory is a sibling of the datagen package, two levels up.
COCO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "coco"
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


def resolve_image_path(file_name: str, *, images_dir: Path = IMAGES_DIR) -> Path:
    """Return the local path to `file_name`.

    Pure local lookup, no network access: `file_name` is expected to already
    exist under `images_dir` (the dataset is pre-downloaded). Raises
    `FileNotFoundError` if it isn't there — run
    `python -m datagen.cli.sync_coco_cache` to populate the cache first.
    """
    local_path = images_dir / file_name
    if not local_path.exists():
        raise FileNotFoundError(
            f"COCO image {file_name!r} not found under {images_dir}. "
            "Run `python -m datagen.cli.sync_coco_cache` to download missing images."
        )
    return local_path


def download_missing_images(
    image_records: list[ImageRecord],
    *,
    images_dir: Path = IMAGES_DIR,
    base_url: str = COCO_VAL2017_BASE_URL,
    fetch: FetchFn = fetch_image_bytes,
) -> list[str]:
    """Download whichever of `image_records` aren't already cached locally.

    Files already present under `images_dir` are left untouched and never
    passed to `fetch`. Returns the file names actually downloaded (a subset
    of `image_records`' file names, in the order given).
    """
    downloaded: list[str] = []
    for record in image_records:
        local_path = images_dir / record.file_name
        if local_path.exists():
            continue

        images_dir.mkdir(parents=True, exist_ok=True)
        image_bytes = fetch(f"{base_url}/{record.file_name}")
        # Write to a sibling `.part` path first and atomically rename into
        # place only once the write has fully completed. Writing
        # `local_path` directly would leave a truncated file cached forever
        # if the process is interrupted mid-write (e.g. on a slow/unreliable
        # connection) — a later call would treat that truncated file as a
        # valid cache hit (the `local_path.exists()` check above) and never
        # re-fetch it.
        part_path = local_path.with_suffix(local_path.suffix + ".part")
        part_path.write_bytes(image_bytes)
        part_path.replace(local_path)
        downloaded.append(record.file_name)
    return downloaded
