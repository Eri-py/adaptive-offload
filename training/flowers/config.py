"""Paths, seed and shared per-model settings for the flowers training pipeline.

Centralizes filesystem locations (resolved from `__file__`, not cwd, so a
missing dataset/checkpoint fails clearly instead of reading from or writing
to wherever a command happens to be run from) and the small/large model
name-to-weights mapping, so `data.py`'s eval transform and `models.py`'s
`weights_for`/`build_model` (Task 2) share one definition instead of two.
"""

from pathlib import Path
from typing import Literal

from torchvision.models import ConvNeXt_Base_Weights, MobileNet_V3_Large_Weights, WeightsEnum

# --- Filesystem locations ----------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "flowers102"
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "models" / "flowers"

SEED = 42
NUM_CLASSES = 102

ModelName = Literal["small", "large"]

# Model name -> ImageNet-pretrained weights enum. "small" is the phone-sized
# model (MobileNetV3-Large), "large" is the server-sized model (ConvNeXt-Base).
MODEL_WEIGHTS: dict[ModelName, WeightsEnum] = {
    "small": MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    "large": ConvNeXt_Base_Weights.IMAGENET1K_V1,
}
