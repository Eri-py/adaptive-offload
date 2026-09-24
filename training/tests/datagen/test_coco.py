"""Tests for `datagen.coco`.

Uses small fake annotation/image fixtures under `tmp_path` and an injected
fetch callable throughout — never touches the real `training/data/coco/`
directory or the network.
"""

import json
from pathlib import Path
from unittest import mock
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


def test_resolve_image_path_download_leaves_no_leftover_part_file(tmp_path: Path) -> None:
    """Regression check: a successful download still ends up at the plain
    final path (not a `.part` path) with the fetched bytes, and the
    intermediate `.part` file used for the atomic write is cleaned up (renamed
    away) rather than left sitting alongside the final file.
    """
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"downloaded-bytes")

    result = resolve_image_path("000000000004.jpg", images_dir=images_dir, fetch=fetch)

    assert result == images_dir / "000000000004.jpg"
    assert result.read_bytes() == b"downloaded-bytes"
    assert not (images_dir / "000000000004.jpg.part").exists()


def test_resolve_image_path_interrupted_write_does_not_cache_truncated_file(
    tmp_path: Path,
) -> None:
    """If the write is interrupted partway through, the final path must never
    end up holding a truncated file that a later call would treat as a valid
    cache hit.

    Without the atomic-write fix, `local_path.write_bytes(...)` writes
    directly to the final path; an interruption mid-write (simulated here by
    raising after the `.part` file is written but before it can be renamed)
    would leave a truncated file at the final path for the *old* code, and
    `resolve_image_path`'s `local_path.exists()` cache check would accept it
    forever. With the fix, the write lands on a `.part` sibling first, so an
    interruption before the rename leaves no file at all at the final path.
    """
    images_dir = tmp_path / "val2017"
    fetch = Mock(return_value=b"only-partial-bytes-before-crash")
    final_path = images_dir / "000000000005.jpg"
    part_path = images_dir / "000000000005.jpg.part"

    original_replace = Path.replace

    def _crash_before_replace(self: Path, target: Path) -> Path:
        if self == part_path:
            raise OSError("simulated interruption before atomic rename")
        return original_replace(self, target)

    with mock.patch.object(Path, "replace", _crash_before_replace):
        try:
            resolve_image_path("000000000005.jpg", images_dir=images_dir, fetch=fetch)
        except OSError:
            pass

    # The interrupted write must not have left a file at the final path...
    assert not final_path.exists()
    # ...and a subsequent call must therefore re-fetch rather than treating a
    # `.part` leftover (or anything else) as a cached hit.
    retry_fetch = Mock(return_value=b"full-bytes-on-retry")
    result = resolve_image_path("000000000005.jpg", images_dir=images_dir, fetch=retry_fetch)

    assert result.read_bytes() == b"full-bytes-on-retry"
    retry_fetch.assert_called_once()
    assert not part_path.exists()
