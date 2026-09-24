"""Export yolov8n.pt to TFLite and bundle sample images for the mobile app.

One-off script, not a registered `training/datagen/cli/` tool — see
features/mobile-inference-benchmark/spec.md for why it lives here instead.

IMPORTANT — the TFLite export (`export_tflite_model()`) must be run in an
isolated, throwaway virtualenv, NOT the shared repo-root `.venv/`. The
shared venv's pinned torch/torchvision/numpy/scipy stack conflicts with the
tflite export toolchain no matter which of ultralytics' two export paths is
used (the modern litert path wants torch<2.14, which fights torchvision's
exact torch==2.14.0 pin; the legacy TensorFlow path's auto-installed
tensorflow/numpy/scipy combo hits its own internal incompatibility). See
features/mobile-inference-benchmark/learnings.md for the full multi-attempt
history if you want the details. One-time setup that worked, run from the
repo root:

    python3 -m venv /tmp/tflite-export-venv  # or: uv venv /tmp/tflite-export-venv
    /tmp/tflite-export-venv/bin/pip install "ultralytics<8.4.83"
    /tmp/tflite-export-venv/bin/pip install "numpy==2.0.2" "scipy==1.13.1" "onnx<1.18"
    /tmp/tflite-export-venv/bin/pip install \
        "tf_keras<=2.19.0" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26"
    /tmp/tflite-export-venv/bin/python training/export_tflite_model.py

(`ultralytics<8.4.83` forces the legacy TensorFlow-based tflite exporter
instead of the newer litert path; the numpy/scipy/onnx pins route around
version conflicts introduced by ultralytics' own auto-installed
tensorflow/onnx dependencies; tf_keras/sng4onnx/onnx_graphsurgeon are
onnx2tf's own requirements, best installed explicitly since ultralytics'
auto-install for them defaults to an NVIDIA package mirror that may not be
reachable.) This is a deliberate, narrow exception to the repo's normal
shared-venv convention — justified because this script runs once, produces
a single committed artifact (`app/assets/models/yolov8n.tflite`), and has
no reason to share an environment with code that needs an incompatible set
of pinned versions. The shared `.venv/` is never touched by this process.
The throwaway venv can be deleted afterward; it isn't part of the repo.

`bundle_sample_images()` has no such constraint and can be run from the
shared repo-root `.venv/` as usual.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# mypy can't see YOLO in ultralytics' runtime-built __all__ (not a missing-stub issue).
from ultralytics import YOLO  # type: ignore[attr-defined]

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "training" / "models" / "yolov8n.pt"
COCO_VAL_DIR = REPO_ROOT / "training" / "data" / "coco" / "val2017"
APP_MODELS_DIR = REPO_ROOT / "app" / "assets" / "models"
APP_IMAGES_DIR = REPO_ROOT / "app" / "assets" / "images"
NUM_SAMPLE_IMAGES = 15


def export_tflite_model() -> Path:
    """Export the YOLOv8n weights to TFLite and copy it into app/assets/models/."""
    model = YOLO(str(WEIGHTS_PATH))
    exported_path = Path(model.export(format="tflite"))
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = APP_MODELS_DIR / "yolov8n.tflite"
    shutil.copyfile(exported_path, dest)
    return dest


def bundle_sample_images() -> list[Path]:
    """Copy NUM_SAMPLE_IMAGES real COCO val2017 photos into app/assets/images/."""
    candidates = sorted(COCO_VAL_DIR.glob("*.jpg"))
    if len(candidates) < NUM_SAMPLE_IMAGES:
        raise RuntimeError(
            f"need {NUM_SAMPLE_IMAGES} images, found {len(candidates)} in {COCO_VAL_DIR}"
        )

    APP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in candidates[:NUM_SAMPLE_IMAGES]:
        dest = APP_IMAGES_DIR / src.name
        shutil.copyfile(src, dest)
        copied.append(dest)
    return copied


if __name__ == "__main__":
    tflite_path = export_tflite_model()
    print(f"Exported TFLite model to {tflite_path}")

    image_paths = bundle_sample_images()
    print(f"Copied {len(image_paths)} sample images to {APP_IMAGES_DIR}")
