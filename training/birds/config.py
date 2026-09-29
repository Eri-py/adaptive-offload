"""Paths, seed and the small/large weights mapping shared by the birds pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torchvision.models import ConvNeXt_Base_Weights, MobileNet_V3_Large_Weights, WeightsEnum

# --- Filesystem locations ----------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_DIR / "data" / "cub200"
WEIGHTS_DIR = TRAINING_DIR / "models" / "birds"

# Run history lives in its own Postgres database (MLFLOW_TRACKING_URI in training/.env),
# separate from adaptive_offload because MLflow's Alembic would collide with the project's.
# Artifacts are never logged; this path only keeps MLflow from littering the cwd.
ARTIFACTS_DIR = TRAINING_DIR / "mlartifacts"
EXPERIMENT_NAME = "birds"

SEED = 42
NUM_CLASSES = 200
NUM_WORKERS = 6
VAL_FRACTION = 0.1
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"

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


# Large model: fewer epochs, smaller lr/batch, since ConvNeXt-Base overfits this small dataset.
TRAIN_SETTINGS: dict[ModelName, TrainSettings] = {
    "small": TrainSettings(epochs=30, lr=1e-3, weight_decay=0.05, batch_size=64),
    "large": TrainSettings(epochs=15, lr=1e-4, weight_decay=0.05, batch_size=32),
}
