"""Edge-density scene-complexity proxy.

A cheap, model-free stand-in for "how visually complex is this frame" — used
throughout the feature spec as a per-frame value (persisted once to the
scene-complexity table by `training/datagen`'s DB-writing code, then reused
by the stratified frame sampler and the stub-inference accuracy formula).

Pure function of pixel data: grayscale conversion + Canny edge detection,
scored as the fraction of pixels classified as edges. No I/O, no database
access — accepting a file path is only a convenience (opening the file with
cv2 is the one necessary I/O step), not part of the computation itself.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

# Canny's two thresholds bound the hysteresis edge-linking step. These are
# OpenCV's commonly-used "reasonable default" values for 8-bit images (see
# the OpenCV Canny tutorial) — not tuned against this project's data, since
# this proxy only needs to rank frames by relative complexity, not match a
# calibrated absolute edge count.
CANNY_LOWER_THRESHOLD = 100.0
CANNY_UPPER_THRESHOLD = 200.0


def scene_complexity(image: str | Path | MatLike) -> float:
    """Return the fraction of edge pixels in `image`, as a float in [0, 1].

    `image` is either a path to an image file (str/Path) or an in-memory
    image array (grayscale or BGR/BGRA, as returned by `cv2.imread`).
    """
    array = _load_array(image)
    gray = _to_grayscale(array)
    edges = cv2.Canny(gray, CANNY_LOWER_THRESHOLD, CANNY_UPPER_THRESHOLD)
    edge_pixel_count = int(np.count_nonzero(edges))
    total_pixel_count = edges.size
    return edge_pixel_count / total_pixel_count


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
