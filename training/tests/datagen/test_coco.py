"""Tests for `datagen.coco`.

Uses small fake annotation/image fixtures under `tmp_path` and an injected
fetch callable throughout — never touches the real `training/data/coco/`
directory or the network.
"""

import json
from pathlib import Path
from unittest.mock import Mock

from datagen.coco import ImageRecord, load_image_index, resolve_image_path

_FAKE_IMAGES = [
    {"id": 1, "file_name": "000000000001.jpg"},
    {"id": 2, "file_name": "000000000002.jpg"},
    {"id": 3, "file_name": "000000000003.jpg"},
]


def _write_fake_annotations(tmp_path: Path) -> Path:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(json.dumps({"images": _FAKE_IMAGES}))
    return annotations_path


def test_load_image_index_returns_id_filename_pairs(tmp_path: Path) -> None:
    records = load_image_index(_write_fake_annotations(tmp_path))

    assert records == [
        ImageRecord(image_id=1, file_name="000000000001.jpg"),
        ImageRecord(image_id=2, file_name="000000000002.jpg"),
        ImageRecord(image_id=3, file_name="000000000003.jpg"),
    ]


def test_resolve_image_path_reuses_cached_file_without_fetching(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    cached = images_dir / "000000000001.jpg"
    cached.write_bytes(b"fake-jpeg-bytes")
    fetch = Mock(side_effect=AssertionError("fetch should not be called for a cached file"))

    result = resolve_image_path("000000000001.jpg", images_dir=images_dir, fetch=fetch)

    assert result == cached
    fetch.assert_not_called()


def test_resolve_image_path_downloads_and_caches_missing_file(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"downloaded-bytes")

    result = resolve_image_path(
        "000000000002.jpg",
        images_dir=images_dir,
        base_url="https://fake.example.com/val2017",
        fetch=fetch,
    )

    assert result == images_dir / "000000000002.jpg"
    assert result.read_bytes() == b"downloaded-bytes"
    fetch.assert_called_once_with("https://fake.example.com/val2017/000000000002.jpg")


def test_resolve_image_path_second_call_reuses_cache_without_refetching(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"downloaded-bytes")

    first = resolve_image_path("000000000003.jpg", images_dir=images_dir, fetch=fetch)
    second = resolve_image_path("000000000003.jpg", images_dir=images_dir, fetch=fetch)

    assert first == second
    fetch.assert_called_once()
