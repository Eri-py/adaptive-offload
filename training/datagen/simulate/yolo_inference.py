"""YOLO-backed real-inference builders for the local/offload paths.

`build_local_inference_fn`/`build_offload_inference_fn` load a real YOLO
model once (CPU nano / GPU extra-large, respectively) and return a closure
that runs real inference on a single frame, producing a `DetectionResult`
from real measured latency and real IoU-based accuracy (via
`datagen.simulate.inference.score_accuracy`). Split out from
`datagen.simulate.inference` so that module's model-independent pieces
(`DetectionResult`, `RunInferenceFn`, `score_accuracy`,
`apply_condition_overhead`) stay free of the `ultralytics`/`torch` import
cost — importing this module alone pulls those in.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np

# `ultralytics`'s top-level `__all__` builds the model names from a runtime
# tuple (`*MODELS`) rather than a literal string list, which mypy's
# no-implicit-reexport check (implied by `strict = true`) doesn't statically
# recognize as an explicit re-export — hence the ignore, not a missing-stub
# problem (the package ships a `py.typed` marker and is otherwise fully typed).
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.engine.results import Results

from datagen.simulate import ground_truth
from datagen.simulate.inference import DetectionResult, RunInferenceFn, score_accuracy


def build_local_inference_fn() -> RunInferenceFn:
    """Load YOLOv8n once on CPU; return a closure that runs real local inference.

    The returned closure measures wall-clock latency around the model call
    itself (not `ultralytics`' own reported `speed['inference']`), so the
    measurement reflects this simulator's actual observed per-frame cost
    end-to-end, the same way a real on-device caller would experience it.
    """
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    # Discard a warmup inference: the first real call otherwise pays for
    # lazy initialisation on top of actual inference, inflating the first
    # frame's measured latency by roughly two orders of magnitude (see
    # review finding B1).
    model(np.zeros((640, 640, 3), dtype=np.uint8), device="cpu", verbose=False)

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
    # Discard a warmup inference: the first real call otherwise pays for
    # lazy initialisation and CUDA context setup on top of actual inference,
    # inflating the first frame's measured latency by roughly two orders of
    # magnitude (see review finding B1).
    model(np.zeros((640, 640, 3), dtype=np.uint8), device="cuda", verbose=False)

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
