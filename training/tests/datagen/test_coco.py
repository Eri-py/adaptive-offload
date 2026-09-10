"""Tests for `datagen.coco`.

Uses small fake annotation/image fixtures under `tmp_path` and an injected
fetch callable throughout — never touches the real `training/data/coco/`
directory or the network.
"""

import json
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest

from datagen.coco import (
    ImageRecord,
    download_missing_images,
    load_image_index,
    resolve_image_path,
)

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

    with pytest.raises(FileNotFoundError, match="datagen.sync_coco_cache"):
        resolve_image_path("000000000099.jpg", images_dir=images_dir)


def test_resolve_image_path_raises_file_not_found_when_images_dir_missing(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "val2017"  # never created

    with pytest.raises(FileNotFoundError, match="datagen.sync_coco_cache"):
        resolve_image_path("000000000099.jpg", images_dir=images_dir)


def test_download_missing_images_skips_already_cached_files(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    cached = images_dir / "000000000001.jpg"
    cached.write_bytes(b"fake-jpeg-bytes")
    fetch = Mock(side_effect=AssertionError("fetch should not be called for a cached file"))

    downloaded = download_missing_images(
        [ImageRecord(image_id=1, file_name="000000000001.jpg")],
        images_dir=images_dir,
        fetch=fetch,
    )

    assert downloaded == []
    fetch.assert_not_called()
    assert cached.read_bytes() == b"fake-jpeg-bytes"


def test_download_missing_images_downloads_and_caches_missing_file(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"downloaded-bytes")

    downloaded = download_missing_images(
        [ImageRecord(image_id=2, file_name="000000000002.jpg")],
        images_dir=images_dir,
        base_url="https://fake.example.com/val2017",
        fetch=fetch,
    )

    result = images_dir / "000000000002.jpg"
    assert downloaded == ["000000000002.jpg"]
    assert result.read_bytes() == b"downloaded-bytes"
    fetch.assert_called_once_with("https://fake.example.com/val2017/000000000002.jpg")


def test_download_missing_images_downloads_only_the_missing_subset(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    images_dir.mkdir()
    (images_dir / "000000000001.jpg").write_bytes(b"already-cached")
    fetch = Mock(return_value=b"downloaded-bytes")

    downloaded = download_missing_images(
        [
            ImageRecord(image_id=1, file_name="000000000001.jpg"),
            ImageRecord(image_id=2, file_name="000000000002.jpg"),
            ImageRecord(image_id=3, file_name="000000000003.jpg"),
        ],
        images_dir=images_dir,
        fetch=fetch,
    )

    assert downloaded == ["000000000002.jpg", "000000000003.jpg"]
    assert fetch.call_count == 2
    assert (images_dir / "000000000001.jpg").read_bytes() == b"already-cached"
    assert (images_dir / "000000000002.jpg").read_bytes() == b"downloaded-bytes"
    assert (images_dir / "000000000003.jpg").read_bytes() == b"downloaded-bytes"


def test_download_missing_images_second_call_reuses_cache_without_refetching(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"downloaded-bytes")
    record = ImageRecord(image_id=3, file_name="000000000003.jpg")

    first = download_missing_images([record], images_dir=images_dir, fetch=fetch)
    second = download_missing_images([record], images_dir=images_dir, fetch=fetch)

    assert first == ["000000000003.jpg"]
    assert second == []
    fetch.assert_called_once()


def test_download_missing_images_leaves_no_leftover_part_file(tmp_path: Path) -> None:
    """Regression check: a successful download still ends up at the plain
    final path (not a `.part` path) with the fetched bytes, and the
    intermediate `.part` file used for the atomic write is cleaned up (renamed
    away) rather than left sitting alongside the final file.
    """
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"downloaded-bytes")

    download_missing_images(
        [ImageRecord(image_id=4, file_name="000000000004.jpg")],
        images_dir=images_dir,
        fetch=fetch,
    )

    result = images_dir / "000000000004.jpg"
    assert result.read_bytes() == b"downloaded-bytes"
    assert not (images_dir / "000000000004.jpg.part").exists()


def test_download_missing_images_interrupted_write_does_not_cache_truncated_file(
    tmp_path: Path,
) -> None:
    """If the write is interrupted partway through, the final path must never
    end up holding a truncated file that a later call would treat as a valid
    cache hit.

    Without the atomic-write fix, writing directly to the final path would
    leave a truncated file there for an interruption mid-write, and a later
    `download_missing_images` call would treat that truncated file as a
    valid cache hit (the `local_path.exists()` check) forever. With the fix,
    the write lands on a `.part` sibling first, so an interruption before the
    rename leaves no file at all at the final path.
    """
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"only-partial-bytes-before-crash")
    record = ImageRecord(image_id=5, file_name="000000000005.jpg")
    final_path = images_dir / "000000000005.jpg"
    part_path = images_dir / "000000000005.jpg.part"

    original_replace = Path.replace

    def _crash_before_replace(self: Path, target: Path) -> Path:
        if self == part_path:
            raise OSError("simulated interruption before atomic rename")
        return original_replace(self, target)

    with mock.patch.object(Path, "replace", _crash_before_replace):
        try:
            download_missing_images([record], images_dir=images_dir, fetch=fetch)
        except OSError:
            pass

    # The interrupted write must not have left a file at the final path...
    assert not final_path.exists()
    # ...and a subsequent call must therefore re-fetch rather than treating a
    # `.part` leftover (or anything else) as a cached hit.
    retry_fetch = Mock(return_value=b"full-bytes-on-retry")
    downloaded = download_missing_images([record], images_dir=images_dir, fetch=retry_fetch)

    assert downloaded == ["000000000005.jpg"]
    assert final_path.read_bytes() == b"full-bytes-on-retry"
    retry_fetch.assert_called_once()
    assert not part_path.exists()
