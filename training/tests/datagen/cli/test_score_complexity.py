"""Tests for the folder-of-images complexity-scoring CLI.

The pure `score_folder()` tests below use plain `def test_...()` functions
with no `pytest` import, per `test_complexity.py`'s original learning about
an environment mismatch that has since been fixed (see that feature's
`learnings.md`, Task 5) — kept as-is since they don't need any pytest API.
The DB-writing tests further down need `monkeypatch`/`capsys`/the real
`postgres_engine` fixture, so they do import `pytest`, matching
`test_run_simulation.py`'s DB-backed tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import Engine

from datagen.cli import score_complexity as score_complexity_module
from datagen.cli.score_complexity import score_folder
from datagen.persistence import get_known_complexity


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


def _write_pool(tmp_path: Path) -> set[str]:
    cv2.imwrite(str(tmp_path / "blank.png"), _blank_image())
    cv2.imwrite(str(tmp_path / "checkerboard.jpg"), _checkerboard_image())
    return {"blank.png", "checkerboard.jpg"}


def _run_main(
    monkeypatch: pytest.MonkeyPatch, postgres_engine: Engine, folder: Path, dataset: str
) -> None:
    monkeypatch.setattr(score_complexity_module, "get_engine", lambda: postgres_engine)
    monkeypatch.setattr(
        sys, "argv", ["score_complexity", "--folder", str(folder), "--dataset", dataset]
    )
    score_complexity_module.main()


def test_main_persists_all_scores_for_a_fresh_dataset(
    postgres_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_names = _write_pool(tmp_path)
    dataset = "test_dataset_fresh"

    _run_main(monkeypatch, postgres_engine, tmp_path, dataset)

    known = get_known_complexity(postgres_engine, dataset)
    assert set(known) == file_names
    assert all(0.0 <= score <= 1.0 for score in known.values())


def test_main_does_not_duplicate_already_known_images_on_rerun(
    postgres_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_names = _write_pool(tmp_path)
    dataset = "test_dataset_rerun"

    _run_main(monkeypatch, postgres_engine, tmp_path, dataset)
    first_known = get_known_complexity(postgres_engine, dataset)
    _run_main(monkeypatch, postgres_engine, tmp_path, dataset)
    second_known = get_known_complexity(postgres_engine, dataset)

    assert set(second_known) == file_names
    assert second_known == first_known


def test_main_prints_every_image_including_already_known_ones(
    postgres_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    file_names = _write_pool(tmp_path)
    dataset = "test_dataset_print_all"

    _run_main(monkeypatch, postgres_engine, tmp_path, dataset)
    capsys.readouterr()  # discard first run's output
    _run_main(monkeypatch, postgres_engine, tmp_path, dataset)
    second_run_output = capsys.readouterr().out

    printed_file_names = {line.split("\t")[0] for line in second_run_output.splitlines()}
    assert printed_file_names == file_names
