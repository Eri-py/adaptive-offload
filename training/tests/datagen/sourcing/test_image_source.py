"""Tests for `datagen.sourcing.image_source`.

Uses small fake annotation/image fixtures under `tmp_path` throughout —
never touches the real `training/data/coco/` directory or the network. This
module never fetches anything over the network, so there's nothing to mock
beyond local file setup.
"""

import json
from pathlib import Path

import pytest

from datagen.sourcing.image_source import ImageRecord, load_image_index, resolve_image_path

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


def test_load_image_index_raises_file_not_found_for_missing_annotations(tmp_path: Path) -> None:
    annotations_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError, match="does_not_exist.json"):
        load_image_index(annotations_path)


def test_load_image_index_raises_value_error_for_invalid_json(tmp_path: Path) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text("not valid json {")

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_image_index(annotations_path)


def test_load_image_index_raises_value_error_when_images_key_missing(tmp_path: Path) -> None:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(json.dumps({"annotations": []}))

    with pytest.raises(ValueError, match="instances_val2017.json"):
        load_image_index(annotations_path)


def test_resolve_image_path_returns_path_for_cached_file(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    cached = images_dir / "000000000001.jpg"
    cached.write_bytes(b"fake-jpeg-bytes")

    result = resolve_image_path("000000000001.jpg", images_dir=images_dir)

    assert result == cached


def test_resolve_image_path_raises_file_not_found_for_missing_file(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="000000000099.jpg"):
        resolve_image_path("000000000099.jpg", images_dir=images_dir)


def test_resolve_image_path_raises_file_not_found_when_images_dir_missing(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "val2017"  # never created

    with pytest.raises(FileNotFoundError, match="000000000099.jpg"):
        resolve_image_path("000000000099.jpg", images_dir=images_dir)
