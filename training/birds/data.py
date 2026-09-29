"""Loads Oxford 102 Flowers splits and builds their transforms/data loaders.

Never downloads: `birds.download` is the only place that does. A missing
dataset raises `FileNotFoundError` naming that command, instead of
torchvision's generic "not found or corrupted" `RuntimeError`.
"""

from collections.abc import Callable
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102

from birds.config import DATA_DIR, MODEL_WEIGHTS, ModelName

Split = Literal["train", "val", "test"]
Transform = Callable[[Any], Any]

_DOWNLOAD_COMMAND = "python -m birds.download"

# ImageNet statistics shared by both models' pretrained weights (confirmed
# equal for MobileNet_V3_Large_Weights.IMAGENET1K_V2 and
# ConvNeXt_Base_Weights.IMAGENET1K_V1's own `transforms()`).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_split(split: Split, transform: Transform) -> Flowers102:
    """Load one split from disk, never downloading it."""
    try:
        return Flowers102(root=DATA_DIR, split=split, download=False, transform=transform)
    except RuntimeError as error:
        raise FileNotFoundError(
            f"Oxford 102 Flowers dataset not found at {DATA_DIR}. "
            f"Run `{_DOWNLOAD_COMMAND}` to fetch it first."
        ) from error


def load_train(transform: Transform) -> Flowers102:
    return _load_split("train", transform)


def load_val(transform: Transform) -> Flowers102:
    return _load_split("val", transform)


def load_test(transform: Transform) -> Flowers102:
    return _load_split("test", transform)


_SPLIT_LOADERS: dict[Split, Callable[[Transform], Flowers102]] = {
    "train": load_train,
    "val": load_val,
    "test": load_test,
}


def train_transform() -> Transform:
    """Random resized crop + horizontal flip, normalized with ImageNet stats."""
    transform: Transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    return transform


def eval_transform(model_name: ModelName) -> Transform:
    """The given model's own pretrained weights' resize/center-crop transform."""
    transform: Transform = MODEL_WEIGHTS[model_name].transforms()
    return transform


def make_data_loader(
    split: Split,
    transform: Transform,
    batch_size: int,
    shuffle: bool,
    generator: "torch.Generator | None" = None,
) -> "DataLoader[Any]":
    """`generator` seeds the shuffling order; only meaningful when `shuffle=True`."""
    dataset = _SPLIT_LOADERS[split](transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
