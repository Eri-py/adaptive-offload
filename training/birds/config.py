"""Paths, seed and shared per-model settings for the birds training pipeline.

Centralizes filesystem locations (resolved from `__file__`, not cwd, so a
missing dataset/checkpoint fails clearly instead of reading from or writing
to wherever a command happens to be run from) and the small/large model
name-to-weights mapping, so `data.py`'s eval transform and `models.py`'s
`weights_for`/`build_model` (Task 2) share one definition instead of two.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torchvision.models import ConvNeXt_Base_Weights, MobileNet_V3_Large_Weights, WeightsEnum

# --- Filesystem locations ----------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cub200"
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "models" / "birds"

SEED = 42
NUM_CLASSES = 102

ModelName = Literal["small", "large"]

# Model name -> ImageNet-pretrained weights enum. "small" is the phone-sized
# model (MobileNetV3-Large), "large" is the server-sized model (ConvNeXt-Base).
MODEL_WEIGHTS: dict[ModelName, WeightsEnum] = {
    "small": MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    "large": ConvNeXt_Base_Weights.IMAGENET1K_V1,
}


@dataclass(frozen=True)
class TrainSettings:
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int


# Per-model training recipe (Task 3): the large model gets fewer epochs and a
# smaller learning rate/batch since ConvNeXt-Base is far more prone to
# overfitting/instability than MobileNetV3 on this small a dataset.
TRAIN_SETTINGS: dict[ModelName, TrainSettings] = {
    "small": TrainSettings(epochs=30, lr=1e-3, weight_decay=0.05, batch_size=64),
    "large": TrainSettings(epochs=15, lr=1e-4, weight_decay=0.05, batch_size=32),
}
