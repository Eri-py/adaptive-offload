"""Tests for the standalone folder-of-images complexity-scoring CLI.

Avoids `import pytest` — per `test_complexity.py`'s learning, importing
pytest here makes mypy follow `_pytest`'s numpy integration into a stub
that's incompatible with this project's `python_version`/venv combination.
Plain `def test_...()` functions are enough for pytest to collect and run
these.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from datagen.cli.score_complexity import score_folder


def _blank_image(size: int = 64) -> np.ndarray:
    """A uniform-color image — no edges anywhere."""
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _checkerboard_image(size: int = 64, square_size: int = 4) -> np.ndarray:
    """A high-frequency black/white checkerboard — edges everywhere."""
    row_indices, col_indices = np.indices((size, size))
    pattern = ((row_indices // square_size) + (col_indices // square_size)) % 2
    grayscale = (pattern * 255).astype(np.uint8)
    return np.stack([grayscale] * 3, axis=-1)


def test_scores_every_image_in_the_folder(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "blank.png"), _blank_image())
    cv2.imwrite(str(tmp_path / "checkerboard.jpg"), _checkerboard_image())

    scores = score_folder(tmp_path)

    assert set(scores) == {"blank.png", "checkerboard.jpg"}
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert scores["checkerboard.jpg"] > scores["blank.png"]


def test_ignores_non_matching_files(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "blank.bmp"), _blank_image())
    (tmp_path / "notes.txt").write_text("not an image")

    scores = score_folder(tmp_path)

    assert set(scores) == {"blank.bmp"}


def test_missing_folder_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    try:
        score_folder(missing)
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError for a missing folder.")


def test_empty_folder_raises_clear_error(tmp_path: Path) -> None:
    try:
        score_folder(tmp_path)
    except ValueError as error:
        assert str(tmp_path) in str(error)
    else:
        raise AssertionError("Expected ValueError for a folder with no images.")
