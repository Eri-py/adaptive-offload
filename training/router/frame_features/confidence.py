"""Detection-confidence router features, computed from one frame's YOLO detections.

Six features: how many detections there are, and the spread of their
confidences and (image-area-normalised) box sizes — see the feature spec's
"Local-model confidence features" and the implementation plan's
feature-definitions table. `confidence_features` is a pure function over
plain sequences so it's testable without a real model; `from_results` adapts
a real `ultralytics.engine.results.Results` object to those sequences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ultralytics.engine.results import Results


@dataclass(frozen=True)
class ConfidenceFeatures:
    """The six confidence features, named to match `FrameFeatures`' columns."""

    detection_count: int
    max_confidence: float
    mean_confidence: float
    min_confidence: float
    mean_box_area: float
    min_box_area: float


def confidence_features(
    confidences: Sequence[float], box_areas: Sequence[float]
) -> ConfidenceFeatures:
    """Derive the six confidence features from per-detection confidences and box areas.

    `confidences` and `box_areas` must be the same length, one entry per
    detection, with `box_areas` already normalised (box area / image area).
    Zero detections gets a count of 0 and 0.0 for every other feature, per
    the spec's "no detections" rule, rather than raising or leaving values
    undefined for an empty sequence.
    """
    if len(confidences) != len(box_areas):
        raise ValueError(
            "confidences and box_areas must be the same length, got "
            f"{len(confidences)} and {len(box_areas)}"
        )
    if not confidences:
        return ConfidenceFeatures(
            detection_count=0,
            max_confidence=0.0,
            mean_confidence=0.0,
            min_confidence=0.0,
            mean_box_area=0.0,
            min_box_area=0.0,
        )
    return ConfidenceFeatures(
        detection_count=len(confidences),
        max_confidence=float(max(confidences)),
        mean_confidence=float(sum(confidences) / len(confidences)),
        min_confidence=float(min(confidences)),
        mean_box_area=float(sum(box_areas) / len(box_areas)),
        min_box_area=float(min(box_areas)),
    )


def sequences_from_results(results: Results) -> tuple[list[float], list[float]]:
    """Extract per-detection confidences and normalised box areas from a YOLO `Results`.

    Split out of `from_results` so a caller that needs to time the model
    call and the feature derivation separately (`router.extract_features`)
    can run the model, get these raw sequences back, and pass them to
    `confidence_features` itself as a distinct, separately-timed step.

    Uses `boxes.conf` for per-detection confidences and `boxes.xywhn`
    (normalised `[x, y, w, h]`) for box areas (`w * h`), so areas are
    already relative to the image, matching the spec's "box size, relative
    to the image area".
    """
    boxes = results.boxes
    assert boxes is not None, (
        "Results.boxes is None — only expected for a non-detection task "
        "(segmentation/pose/classification); this adapter is only used with "
        "a detection-task model (yolov8n.pt)."
    )
    confidences = [float(conf) for conf in boxes.conf.tolist()]
    box_areas = [float(w * h) for _, _, w, h in boxes.xywhn.tolist()]
    return confidences, box_areas


def from_results(results: Results) -> ConfidenceFeatures:
    """Adapt one YOLO `Results` object to `confidence_features`.

    See `sequences_from_results` for how the sequences are extracted.
    """
    confidences, box_areas = sequences_from_results(results)
    return confidence_features(confidences, box_areas)
