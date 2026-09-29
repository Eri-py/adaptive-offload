"""Loads CUB-200-2011 splits (train/val carved from the official train) and their loaders."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from birds.config import (
    DATA_DIR,
    MODEL_WEIGHTS,
    NUM_CLASSES,
    NUM_WORKERS,
    SEED,
    VAL_FRACTION,
    ModelName,
)

Split = Literal["train", "val", "test"]
Transform = Callable[[Any], Any]
Record = tuple[str, str, int]  # (photo id, path relative to images/, label 0-199)

_DOWNLOAD_COMMAND = "python -m birds.download"
_DATASET_FOLDER = "CUB_200_2011"
INDEX_FILES = ("images.txt", "image_class_labels.txt", "train_test_split.txt")

# ImageNet stats, identical in both models' pretrained-weights `transforms()`.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def dataset_root() -> Path:
    """The extracted CUB folder inside `DATA_DIR`."""
    return DATA_DIR / _DATASET_FOLDER


class CubDataset(Dataset[tuple[Any, int]]):
    """CUB photos as (transformed RGB image, label) pairs."""

    def __init__(self, root: Path, records: list[Record], transform: Transform) -> None:
        self._root = root
        self._records = records
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        _, relative_path, label = self._records[index]
        with Image.open(self._root / "images" / relative_path) as image:
            return self._transform(image.convert("RGB")), label


def _read_pairs(path: Path) -> dict[str, str]:
    return dict(line.split() for line in path.read_text().splitlines() if line.strip())


def _read_index() -> tuple[dict[str, str], dict[str, int], dict[str, bool]]:
    """Reads (id -> path, id -> 0-based label, id -> is official train)."""
    root = dataset_root()
    missing = [str(root / name) for name in INDEX_FILES if not (root / name).is_file()]
    if not root.is_dir() or missing:
        raise FileNotFoundError(
            f"CUB-200-2011 dataset not found or incomplete at {root}"
            f"{' (missing: ' + ', '.join(missing) + ')' if missing and root.is_dir() else ''}. "
            f"Run `{_DOWNLOAD_COMMAND}` to fetch it first."
        )
    paths = _read_pairs(root / "images.txt")
    labels = {k: int(v) - 1 for k, v in _read_pairs(root / "image_class_labels.txt").items()}
    is_train = {k: v == "1" for k, v in _read_pairs(root / "train_test_split.txt").items()}
    return paths, labels, is_train


def _split_from_index(
    labels: dict[str, int], is_train: dict[str, bool]
) -> tuple[list[str], list[str], list[str]]:
    official_train = sorted((k for k, t in is_train.items() if t), key=int)
    test_ids = sorted((k for k, t in is_train.items() if not t), key=int)
    # Per species in class order, one shared rng: hold out max(1, n // 10).
    rng = np.random.default_rng(SEED)
    val_ids: list[str] = []
    for species in range(NUM_CLASSES):
        species_ids = [k for k in official_train if labels[k] == species]
        rng.shuffle(species_ids)
        val_ids += species_ids[: max(1, int(len(species_ids) * VAL_FRACTION + 1e-9))]
    val_set = set(val_ids)
    train_ids = [k for k in official_train if k not in val_set]
    return train_ids, sorted(val_ids, key=int), test_ids


def split_ids() -> tuple[list[str], list[str], list[str]]:
    """Disjoint (train, val, test) photo-id lists; deterministic for a given dataset."""
    _, labels, is_train = _read_index()
    return _split_from_index(labels, is_train)


def _load_split(split: Split, transform: Transform) -> CubDataset:
    paths, labels, is_train = _read_index()
    ids = dict(zip(("train", "val", "test"), _split_from_index(labels, is_train), strict=True))[
        split
    ]
    records = [(k, paths[k], labels[k]) for k in ids]
    return CubDataset(dataset_root(), records, transform)


def load_train(transform: Transform) -> CubDataset:
    return _load_split("train", transform)


def load_val(transform: Transform) -> CubDataset:
    return _load_split("val", transform)


def load_test(transform: Transform) -> CubDataset:
    return _load_split("test", transform)


_SPLIT_LOADERS: dict[Split, Callable[[Transform], CubDataset]] = {
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
