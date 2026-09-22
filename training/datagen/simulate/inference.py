"""Real-model inference: YOLO-backed builders plus the model-independent pieces.

`DetectionResult` is the shape a real model's inference on a single frame
produces (real measured latency, real measured accuracy). `score_accuracy`
is the pure IoU-based scoring function that turns a model's predicted boxes
plus a frame's ground-truth boxes into an accuracy value — independent of
which model produced the predictions. `apply_condition_overhead` adds
synthetic condition-driven latency overhead (device load, bandwidth, network
latency, packet loss) on top of a real measured base latency, now that a
real base latency exists; it does not touch accuracy at all, since accuracy
now comes from real IoU-based scoring rather than a synthetic model.
`build_local_inference_fn`/`build_offload_inference_fn` load a real YOLO
model once (CPU nano / GPU extra-large, respectively) and return a closure
that runs real inference on a single frame, producing a `DetectionResult`
from real measured latency and real IoU-based accuracy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np

# `ultralytics`'s top-level `__all__` builds the model names from a runtime
# tuple (`*MODELS`) rather than a literal string list, which mypy's
# no-implicit-reexport check (implied by `strict = true`) doesn't statically
# recognize as an explicit re-export — hence the ignore, not a missing-stub
# problem (the package ships a `py.typed` marker and is otherwise fully typed).
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.engine.results import Results

from datagen.config import (
    LOCAL_LATENCY_DEVICE_LOAD_COEFFICIENT_MS,
    LOCAL_LATENCY_NOISE_STD_MS,
    OFFLOAD_LATENCY_BANDWIDTH_COEFFICIENT_MS,
    OFFLOAD_LATENCY_NETWORK_LATENCY_COEFFICIENT,
    OFFLOAD_LATENCY_NOISE_STD_MS,
    OFFLOAD_LATENCY_PACKET_LOSS_PENALTY_COEFFICIENT_MS,
)
from datagen.simulate import ground_truth

# Condition-vector field order, matching `conditions.sample_condition_vectors`.
_BANDWIDTH_INDEX = 0
_NETWORK_LATENCY_INDEX = 1
_PACKET_LOSS_INDEX = 2
_DEVICE_LOAD_INDEX = 3


class DetectionResult(NamedTuple):
    """One model's real measured inference result for a single frame."""

    latency_ms: float
    accuracy: float


# Given an image path and that frame's ground-truth boxes, runs one model's
# real inference on the image and returns its real measured `DetectionResult`.
RunInferenceFn = Callable[[Path, list[ground_truth.Box]], DetectionResult]


def build_local_inference_fn() -> RunInferenceFn:
    """Load YOLOv8n once on CPU; return a closure that runs real local inference.

    The returned closure measures wall-clock latency around the model call
    itself (not `ultralytics`' own reported `speed['inference']`), so the
    measurement reflects this simulator's actual observed per-frame cost
    end-to-end, the same way a real on-device caller would experience it.
    """
    model = YOLO("yolov8n.pt")
    model.to("cpu")

    def run_inference(
        image_path: Path, ground_truth_boxes: list[ground_truth.Box]
    ) -> DetectionResult:
        start = time.perf_counter()
        # `stream=False` (the default) always returns a `list[Results]`, never
        # the `Iterator` variant `__call__`'s general signature also allows.
        results = cast(list[Results], model(image_path, device="cpu", verbose=False))
        latency_ms = (time.perf_counter() - start) * 1000.0

        predicted_boxes = _to_boxes(results[0])
        accuracy = score_accuracy(predicted_boxes, ground_truth_boxes)
        return DetectionResult(latency_ms=latency_ms, accuracy=accuracy)

    return run_inference


def build_offload_inference_fn() -> RunInferenceFn:
    """Load YOLOv8x once on GPU; return a closure that runs real offload inference.

    Same shape as `build_local_inference_fn`, but the extra-large model on
    `device="cuda"`. No CUDA-availability fallback — this simulator assumes a
    real GPU is present, per the feature's approach and key decisions.
    """
    model = YOLO("yolov8x.pt")
    model.to("cuda")

    def run_inference(
        image_path: Path, ground_truth_boxes: list[ground_truth.Box]
    ) -> DetectionResult:
        start = time.perf_counter()
        results = cast(list[Results], model(image_path, device="cuda", verbose=False))
        latency_ms = (time.perf_counter() - start) * 1000.0

        predicted_boxes = _to_boxes(results[0])
        accuracy = score_accuracy(predicted_boxes, ground_truth_boxes)
        return DetectionResult(latency_ms=latency_ms, accuracy=accuracy)

    return run_inference


def _to_boxes(result: Results) -> list[ground_truth.Box]:
    """Convert one `ultralytics.engine.results.Results` object's boxes to `Box`es.

    Resolves each detection's class index to a name via the result's own
    `names` mapping (the loaded model's COCO class-index-to-name mapping),
    and each detection's `xyxy` corners directly to `Box`'s
    `(x_min, y_min, x_max, y_max)` shape.
    """
    boxes = result.boxes
    names = result.names
    assert boxes is not None, (
        "Results.boxes is None — only expected for a non-detection task "
        "(segmentation/pose/classification), but this module only loads "
        "detection-task models (yolov8n.pt/yolov8x.pt)."
    )

    predicted_boxes = []
    for xyxy, cls_index in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), strict=True):
        x_min, y_min, x_max, y_max = xyxy
        predicted_boxes.append(
            ground_truth.Box(
                category_name=names[int(cls_index)],
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )
    return predicted_boxes


def apply_condition_overhead(
    condition: tuple[float, float, float, float],
    local_base: DetectionResult,
    offload_base: DetectionResult,
    seed: int,
) -> tuple[float, float, float, float]:
    """Add synthetic condition-driven latency overhead to real base latencies.

    `condition` is `(bandwidth_mbps, network_latency_ms, packet_loss_pct,
    device_load_pct)`, matching `conditions.sample_condition_vectors`'s
    output order. `local_base`/`offload_base` are each path's real measured
    `DetectionResult` (real base latency, real IoU-scored accuracy) for one
    frame, independent of condition. Latency noise is independent Gaussian
    noise drawn from a `seed`-derived RNG, in a fixed draw order, so the same
    `(condition, local_base, offload_base, seed)` always returns identical
    values and varying `seed` alone changes only the noise draws.

    Returns `(local_latency_ms, local_accuracy, offload_latency_ms,
    offload_accuracy)`. Accuracy is passed through unchanged from
    `local_base`/`offload_base` — condition never modifies accuracy, since
    accuracy is now real measured IoU-based scoring rather than a synthetic
    model.
    """
    bandwidth_mbps = condition[_BANDWIDTH_INDEX]
    network_latency_ms = condition[_NETWORK_LATENCY_INDEX]
    packet_loss_pct = condition[_PACKET_LOSS_INDEX]
    device_load_pct = condition[_DEVICE_LOAD_INDEX]

    rng = np.random.default_rng(seed)
    local_latency_noise = rng.normal(0.0, LOCAL_LATENCY_NOISE_STD_MS)
    offload_latency_noise = rng.normal(0.0, OFFLOAD_LATENCY_NOISE_STD_MS)

    local_latency_ms = (
        local_base.latency_ms
        + LOCAL_LATENCY_DEVICE_LOAD_COEFFICIENT_MS * device_load_pct
        + local_latency_noise
    )

    offload_latency_ms = (
        offload_base.latency_ms
        + OFFLOAD_LATENCY_BANDWIDTH_COEFFICIENT_MS / bandwidth_mbps
        + OFFLOAD_LATENCY_NETWORK_LATENCY_COEFFICIENT * network_latency_ms
        + OFFLOAD_LATENCY_PACKET_LOSS_PENALTY_COEFFICIENT_MS * packet_loss_pct
        + offload_latency_noise
    )

    return (
        float(local_latency_ms),
        local_base.accuracy,
        float(offload_latency_ms),
        offload_base.accuracy,
    )


def score_accuracy(
    predicted_boxes: list[ground_truth.Box],
    ground_truth_boxes: list[ground_truth.Box],
    *,
    iou_threshold: float = 0.5,
) -> float:
    """Score detection accuracy as the fraction of ground-truth boxes matched.

    A ground-truth box counts as matched if `predicted_boxes` contains a
    same-`category_name` box with IoU >= `iou_threshold` against it. Returns
    `1.0` when `ground_truth_boxes` is empty — vacuously nothing was missed,
    and it avoids a division by zero.
    """
    if not ground_truth_boxes:
        return 1.0

    matched = 0
    for gt_box in ground_truth_boxes:
        if any(
            pred_box.category_name == gt_box.category_name
            and _iou(pred_box, gt_box) >= iou_threshold
            for pred_box in predicted_boxes
        ):
            matched += 1

    return matched / len(ground_truth_boxes)


def _iou(a: ground_truth.Box, b: ground_truth.Box) -> float:
    """Intersection-over-union of two axis-aligned boxes; `0.0` if disjoint."""
    intersection_x_min = max(a.x_min, b.x_min)
    intersection_y_min = max(a.y_min, b.y_min)
    intersection_x_max = min(a.x_max, b.x_max)
    intersection_y_max = min(a.y_max, b.y_max)

    intersection_width = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    intersection_area = intersection_width * intersection_height

    if intersection_area == 0.0:
        return 0.0

    a_area = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    b_area = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union_area = a_area + b_area - intersection_area

    return intersection_area / union_area
