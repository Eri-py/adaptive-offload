"""Tests for `birds.data` against a tiny fake CUB tree in `tmp_path`.

No network, GPU or real dataset: a few species with a handful of 4x4 photos.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from birds import data as data_module

# species index -> (official-train photos, official-test photos)
_LAYOUT = {0: (25, 2), 1: (12, 2), 2: (5, 1)}


def _identity_transform(image: Any) -> Any:
    return image


def _tensor_transform(image: Any) -> Any:
    return torch.zeros(3, 4, 4)


@pytest.fixture
def fake_cub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "CUB_200_2011"
    (root / "images").mkdir(parents=True)
    images, labels, splits = [], [], []
    photo_id = 0
    for species, (n_train, n_test) in _LAYOUT.items():
        folder = f"{species + 1:03d}.Species_{species}"
        (root / "images" / folder).mkdir()
        for is_train in [1] * n_train + [0] * n_test:
            photo_id += 1
            relative = f"{folder}/photo_{photo_id}.png"
            Image.new("RGB", (4, 4), (photo_id, 0, 0)).save(root / "images" / relative)
            images.append(f"{photo_id} {relative}")
            labels.append(f"{photo_id} {species + 1}")
            splits.append(f"{photo_id} {is_train}")
    (root / "images.txt").write_text("\n".join(images) + "\n")
    (root / "image_class_labels.txt").write_text("\n".join(labels) + "\n")
    (root / "train_test_split.txt").write_text("\n".join(splits) + "\n")
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "NUM_CLASSES", len(_LAYOUT))
    return root


def _species_of(ids: list[str], root: Path) -> list[int]:
    labels = dict(
        line.split() for line in (root / "image_class_labels.txt").read_text().split("\n") if line
    )
    return [int(labels[i]) - 1 for i in ids]


def test_split_is_disjoint_and_covers_official_train(fake_cub: Path) -> None:
    train, val, test = data_module.split_ids()

    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert len(train) + len(val) == sum(n for n, _ in _LAYOUT.values())


def test_validation_count_per_species(fake_cub: Path) -> None:
    _, val, _ = data_module.split_ids()

    counts = [_species_of(val, fake_cub).count(s) for s in _LAYOUT]
    assert counts == [max(1, n // 10) for n, _ in _LAYOUT.values()]  # [2, 1, 1]


def test_split_is_deterministic(fake_cub: Path) -> None:
    assert data_module.split_ids() == data_module.split_ids()


def test_test_split_is_exactly_the_official_test_ids(fake_cub: Path) -> None:
    _, _, test = data_module.split_ids()
    official = [
        line.split()[0]
        for line in (fake_cub / "train_test_split.txt").read_text().splitlines()
        if line.endswith(" 0")
    ]

    assert test == official


def test_labels_are_zero_based(fake_cub: Path) -> None:
    dataset = data_module.load_test(_identity_transform)

    labels = sorted({dataset[i][1] for i in range(len(dataset))})
    assert labels == [0, 1, 2]


def test_missing_root_raises_helpful_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="python -m birds.download"):
        data_module.load_train(_identity_transform)


def test_missing_index_file_raises_helpful_error(fake_cub: Path) -> None:
    (fake_cub / "train_test_split.txt").unlink()

    with pytest.raises(FileNotFoundError, match="python -m birds.download"):
        data_module.split_ids()


def test_make_data_loader_returns_the_named_split(
    fake_cub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_module, "NUM_WORKERS", 0)
    _, val, _ = data_module.split_ids()

    loader = data_module.make_data_loader(
        "val", transform=_tensor_transform, batch_size=2, shuffle=False
    )

    assert loader.batch_size == 2
    assert sum(len(labels) for _, labels in loader) == len(val)
    assert len(loader.dataset) == len(val)  # type: ignore[arg-type]
