"""Tests for the edge-density scene-complexity proxy.

Avoids `import pytest` — per Task 4's learning, importing pytest here makes
mypy follow `_pytest`'s numpy integration into a stub that's incompatible
with this project's `python_version`/venv combination. Plain `def
test_...()` functions are enough for pytest to collect and run these.
"""

from __future__ import annotations

import numpy as np

from datagen.complexity import scene_complexity


def _blank_image(size: int = 200) -> np.ndarray:
    """A uniform-color image — no edges anywhere."""
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _checkerboard_image(size: int = 200, square_size: int = 4) -> np.ndarray:
    """A high-frequency black/white checkerboard — edges everywhere."""
    row_indices, col_indices = np.indices((size, size))
    pattern = ((row_indices // square_size) + (col_indices // square_size)) % 2
    grayscale = (pattern * 255).astype(np.uint8)
    return np.stack([grayscale] * 3, axis=-1)


def _noise_image(size: int = 200, seed: int = 0) -> np.ndarray:
    """Random per-pixel noise — also high-frequency, no structure."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_blank_image_scores_near_zero() -> None:
    score = scene_complexity(_blank_image())
    assert score < 0.01


def test_checkerboard_scores_much_higher_than_blank() -> None:
    blank_score = scene_complexity(_blank_image())
    checkerboard_score = scene_complexity(_checkerboard_image())
    assert checkerboard_score > blank_score
    assert checkerboard_score > 0.1


def test_noise_scores_much_higher_than_blank() -> None:
    blank_score = scene_complexity(_blank_image())
    noise_score = scene_complexity(_noise_image())
    assert noise_score > blank_score
    assert noise_score > 0.1


def test_score_is_always_within_unit_interval() -> None:
    for image in (_blank_image(), _checkerboard_image(), _noise_image()):
        score = scene_complexity(image)
        assert 0.0 <= score <= 1.0


def test_grayscale_input_is_accepted() -> None:
    grayscale = _checkerboard_image()[:, :, 0]
    score = scene_complexity(grayscale)
    assert 0.0 <= score <= 1.0
