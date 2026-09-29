"""Tests for the image-statistics router features.

Avoids `import pytest` (per `training/tests/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv). Plain
`def test_...()` functions are enough for pytest to collect and run these.
"""

from __future__ import annotations

import cv2
import numpy as np
from cv2.typing import MatLike

from router.frame_features.image_stats import (
    brightness,
    colorfulness,
    contrast,
    entropy,
    image_stats,
    sharpness,
)


def _uniform_image(value: int = 128, size: int = 64) -> MatLike:
    """A flat-colour image — same value on every channel and pixel, no detail."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def _noise_image(size: int = 64, seed: int = 0) -> MatLike:
    """Random per-pixel noise — no structure, so blurring visibly softens it."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _two_color_saturated_image(size: int = 64) -> MatLike:
    """Half pure red, half pure blue — maximally saturated, unlike any grey image."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, : size // 2] = (0, 0, 255)  # BGR red
    image[:, size // 2 :] = (255, 0, 0)  # BGR blue
    return image


def test_uniform_grey_image_has_zero_contrast_entropy_and_colorfulness() -> None:
    image = _uniform_image()
    assert contrast(image) == 0.0
    assert entropy(image) == 0.0
    assert colorfulness(image) == 0.0


def test_black_image_has_zero_brightness() -> None:
    assert brightness(_uniform_image(value=0)) == 0.0


def test_white_image_has_max_brightness() -> None:
    assert brightness(_uniform_image(value=255)) == 1.0


def test_blurred_noise_has_lower_sharpness_than_original() -> None:
    noise = _noise_image()
    blurred = cv2.GaussianBlur(noise, (9, 9), sigmaX=3.0)
    assert sharpness(blurred) < sharpness(noise)


def test_saturated_two_color_image_has_higher_colorfulness_than_grey() -> None:
    assert colorfulness(_two_color_saturated_image()) > colorfulness(_uniform_image())


def test_image_stats_matches_individual_functions() -> None:
    image = _noise_image()
    stats = image_stats(image)
    assert stats.sharpness == sharpness(image)
    assert stats.brightness == brightness(image)
    assert stats.contrast == contrast(image)
    assert stats.colorfulness == colorfulness(image)
    assert stats.entropy == entropy(image)
