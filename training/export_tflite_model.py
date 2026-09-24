"""Export yolov8n.pt to TFLite and bundle sample images for the mobile app.

One-off script, not a registered `training/datagen/cli/` tool — see
features/mobile-inference-benchmark/spec.md for why it lives here instead.
Run with the shared repo-root .venv active: `python training/export_tflite_model.py`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

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
        raise RuntimeError(f"need {NUM_SAMPLE_IMAGES} images, found {len(candidates)} in {COCO_VAL_DIR}")

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
