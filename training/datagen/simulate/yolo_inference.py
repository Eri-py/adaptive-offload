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

# mypy can't see YOLO in ultralytics' runtime-built __all__ (not a missing-stub issue).
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.engine.results import Results

from datagen import config
from datagen.simulate import ground_truth
from datagen.simulate.inference import DetectionResult, RunInferenceFn, score_accuracy


def build_local_inference_fn() -> RunInferenceFn:
    """Load YOLOv8n once on CPU; return a closure that runs real local inference.

    See `_build_inference_fn` for the shared setup/measurement behavior.
    """
    return _build_inference_fn(config.LOCAL_MODEL_WEIGHTS_PATH, "cpu")


def build_offload_inference_fn() -> RunInferenceFn:
    """Load YOLOv8x once on GPU; return a closure that runs real offload inference.

    Same shape as `build_local_inference_fn`, but the extra-large model on
    `device="cuda"`. No CUDA-availability fallback — this simulator assumes a
    real GPU is present, per the feature's approach and key decisions. See
    `_build_inference_fn` for the shared setup/measurement behavior.
    """
    return _build_inference_fn(config.OFFLOAD_MODEL_WEIGHTS_PATH, "cuda")


def _build_inference_fn(weights_path: Path, device: str) -> RunInferenceFn:
    """Load a YOLO model once on `device`; return a closure that runs real inference.

    Shared by `build_local_inference_fn` and `build_offload_inference_fn`,
    which differ only in which weights file and device they pass here.

    Raises `FileNotFoundError` if the weights aren't at `weights_path`.
    `ultralytics`'s `YOLO(...)` will otherwise silently download a fresh copy
    there instead of failing — it matches on the file's basename against its
    list of known asset names regardless of whether the given path is
    missing, so a fixed path alone isn't enough to make a missing file fail
    clearly.

    The returned closure measures wall-clock latency around the model call
    itself (not `ultralytics`' own reported `speed['inference']`), so the
    measurement reflects this simulator's actual observed per-frame cost
    end-to-end, the same way a real on-device caller would experience it.
    """
    _require_weights_file(weights_path)
    model = YOLO(weights_path)
    model.to(device)
    # Discard a warmup inference: the first real call otherwise pays for lazy
    # initialisation (and, on GPU, CUDA context setup) on top of actual
    # inference, inflating the first frame's measured latency by roughly two
    # orders of magnitude (see review finding B1).
    model(np.zeros((640, 640, 3), dtype=np.uint8), device=device, verbose=False)

    def run_inference(
        image_path: Path, ground_truth_boxes: list[ground_truth.Box]
    ) -> DetectionResult:
        start = time.perf_counter()
        # `stream=False` (the default) always returns a `list[Results]`, never
        # the `Iterator` variant `__call__`'s general signature also allows.
        results = cast(list[Results], model(image_path, device=device, verbose=False))
        latency_ms = (time.perf_counter() - start) * 1000.0

        predicted_boxes = _to_boxes(results[0])
        accuracy = score_accuracy(predicted_boxes, ground_truth_boxes)
        return DetectionResult(latency_ms=latency_ms, accuracy=accuracy)

    return run_inference


def _require_weights_file(weights_path: Path) -> None:
    """Fail fast with a clear error if `weights_path` isn't an existing file.

    `ultralytics`'s `YOLO(...)` resolves its `model` argument by basename
    against a list of known official asset names (e.g. `yolov8n.pt`), not by
    whether the given path looks like a path — so passing a fixed but
    missing path still triggers a silent multi-hundred-MB download to that
    path rather than an error. Checking existence ourselves first keeps this
    simulator's "never fetches data itself" guarantee.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Place the real "
            "YOLO weights file there before running the simulator — "
            "run-simulation does not download weights itself."
        )


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
