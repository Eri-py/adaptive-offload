"""Tests for the standalone COCO cache-population CLI.

Uses small fake annotation/image fixtures under `tmp_path` and an injected
fetch callable throughout — never touches the real `training/data/coco/`
directory or the network. No database access at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from datagen.sync_coco_cache import sync_coco_cache

_FAKE_IMAGES = [
    {"id": 1, "file_name": "000000000001.jpg"},
    {"id": 2, "file_name": "000000000002.jpg"},
    {"id": 3, "file_name": "000000000003.jpg"},
]


def _write_fake_annotations(tmp_path: Path) -> Path:
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(json.dumps({"images": _FAKE_IMAGES}))
    return annotations_path


def test_sync_coco_cache_downloads_exactly_the_missing_files(tmp_path: Path) -> None:
    annotations_path = _write_fake_annotations(tmp_path)
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    (images_dir / "000000000001.jpg").write_bytes(b"already-cached")
    fetch = Mock(return_value=b"downloaded-bytes")

    already_cached_count, newly_downloaded_count = sync_coco_cache(
        annotations_path=annotations_path,
        images_dir=images_dir,
        fetch=fetch,
    )

    assert already_cached_count == 1
    assert newly_downloaded_count == 2
    assert fetch.call_count == 2
    assert (images_dir / "000000000001.jpg").read_bytes() == b"already-cached"
    assert (images_dir / "000000000002.jpg").read_bytes() == b"downloaded-bytes"
    assert (images_dir / "000000000003.jpg").read_bytes() == b"downloaded-bytes"


def test_sync_coco_cache_leaves_already_cached_files_untouched(tmp_path: Path) -> None:
    annotations_path = _write_fake_annotations(tmp_path)
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    for record in _FAKE_IMAGES:
        (images_dir / str(record["file_name"])).write_bytes(b"already-cached")
    fetch = Mock(side_effect=AssertionError("fetch should not be called when everything's cached"))

    already_cached_count, newly_downloaded_count = sync_coco_cache(
        annotations_path=annotations_path,
        images_dir=images_dir,
        fetch=fetch,
    )

    assert already_cached_count == 3
    assert newly_downloaded_count == 0
    fetch.assert_not_called()


def test_sync_coco_cache_uses_base_url_for_downloads(tmp_path: Path) -> None:
    annotations_path = _write_fake_annotations(tmp_path)
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"downloaded-bytes")

    sync_coco_cache(
        annotations_path=annotations_path,
        images_dir=images_dir,
        base_url="https://fake.example.com/val2017",
        fetch=fetch,
    )

    fetch.assert_any_call("https://fake.example.com/val2017/000000000001.jpg")
