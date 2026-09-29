"""Cheap image-statistics router features, computed from a frame alone.

Five per-frame statistics — no model runs for these, so they're cheap enough
to eventually compute on a phone (see the feature spec's "Image features").
Exact definitions live in the implementation plan's feature-definitions
table. Mirrors `coco.datagen.sampling.complexity`'s path-or-array input handling
(`cv2.imread`) so callers can pass either a file path or an already-loaded
BGR array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike


@dataclass(frozen=True)
class ImageStats:
    """The five image-statistics features, named to match `FrameFeatures`' columns."""

    sharpness: float
    brightness: float
    contrast: float
    colorfulness: float
    entropy: float


def sharpness(image: str | Path | MatLike) -> float:
    """Variance of the Laplacian of the grayscale image — lower means blurrier."""
    return _sharpness(_to_grayscale(_load_array(image)))


def brightness(image: str | Path | MatLike) -> float:
    """Mean grayscale intensity, normalised to [0, 1]."""
    return _brightness(_to_grayscale(_load_array(image)))


def contrast(image: str | Path | MatLike) -> float:
    """Standard deviation of grayscale intensity (RMS contrast), normalised to [0, 1]."""
    return _contrast(_to_grayscale(_load_array(image)))


def colorfulness(image: str | Path | MatLike) -> float:
    """Hasler-Süsstrunk colourfulness metric, computed on the image's colour channels."""
    return _colorfulness(_load_array(image))


def entropy(image: str | Path | MatLike) -> float:
    """Shannon entropy, in bits, of the 256-bin grayscale histogram."""
    return _entropy(_to_grayscale(_load_array(image)))


def image_stats(image: str | Path | MatLike) -> ImageStats:
    """Compute all five image-statistics features for `image` in one call.

    Loads/converts `image` once and reuses it across statistics, instead of
    calling each function above separately (which would each reload/reconvert).
    """
    array = _load_array(image)
    gray = _to_grayscale(array)
    return ImageStats(
        sharpness=_sharpness(gray),
        brightness=_brightness(gray),
        contrast=_contrast(gray),
        colorfulness=_colorfulness(array),
        entropy=_entropy(gray),
    )


def _sharpness(gray: MatLike) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(gray: MatLike) -> float:
    return float(gray.mean()) / 255.0


def _contrast(gray: MatLike) -> float:
    return float(gray.std()) / 255.0


def _colorfulness(array: MatLike) -> float:
    """Hasler-Süsstrunk metric: combined spread and mean of two colour-opponent signals."""
    # `cv2.imread`/BGR order; a signed dtype avoids uint8 underflow when differencing.
    blue, green, red = (array[:, :, i].astype(np.float64) for i in range(3))
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    std_component = np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
    mean_component = np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    return float(std_component + 0.3 * mean_component)


def _entropy(gray: MatLike) -> float:
    histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
    counted = histogram[histogram > 0]
    probabilities = counted / counted.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _load_array(image: str | Path | MatLike) -> MatLike:
    """Resolve `image` to an in-memory array, reading from disk if it's a path."""
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image))
        if loaded is None:
            raise ValueError(f"Could not read image at {image!r}")
        return loaded
    return image


def _to_grayscale(array: MatLike) -> MatLike:
    """Convert a color array to grayscale; pass an already-grayscale array through."""
    if array.ndim == 2:
        return array
    channel_count = array.shape[2]
    if channel_count == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
