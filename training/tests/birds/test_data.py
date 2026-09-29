"""Tests for `birds.data`: the missing-dataset error and per-split wiring.

`Flowers102` is always patched here — these tests never touch the network or
a real dataset on disk.
"""

from pathlib import Path
from typing import Any

import pytest

from birds import data as data_module


class _FakeFlowers102:
    """Stands in for `torchvision.datasets.Flowers102`, recording its split."""

    def __init__(
        self,
        recorded_splits: list[str],
        root: Any,
        split: str,
        download: bool,
        transform: Any = None,
    ) -> None:
        recorded_splits.append(split)
        self._items = [(0, 0), (1, 1)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self._items[index]


def _patch_flowers102(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patches `birds.data.Flowers102`, returning the list of splits it's called with."""
    recorded_splits: list[str] = []

    def fake_flowers102(*args: Any, **kwargs: Any) -> _FakeFlowers102:
        return _FakeFlowers102(recorded_splits, *args, **kwargs)

    monkeypatch.setattr(data_module, "Flowers102", fake_flowers102)
    return recorded_splits


def _identity_transform(image: Any) -> Any:
    return image


def test_missing_dataset_raises_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="python -m birds.download"):
        data_module.load_train(transform=_identity_transform)


def test_load_train_uses_train_split(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_splits = _patch_flowers102(monkeypatch)

    data_module.load_train(transform=_identity_transform)

    assert recorded_splits == ["train"]


def test_load_val_uses_val_split(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_splits = _patch_flowers102(monkeypatch)

    data_module.load_val(transform=_identity_transform)

    assert recorded_splits == ["val"]


def test_load_test_uses_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_splits = _patch_flowers102(monkeypatch)

    data_module.load_test(transform=_identity_transform)

    assert recorded_splits == ["test"]


def test_make_data_loader_dispatches_to_the_named_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_splits = _patch_flowers102(monkeypatch)

    loader = data_module.make_data_loader(
        "val", transform=_identity_transform, batch_size=4, shuffle=False
    )

    assert recorded_splits == ["val"]
    assert loader.batch_size == 4
