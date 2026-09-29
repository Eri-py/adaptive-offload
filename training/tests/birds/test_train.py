"""Tests for `birds.train`: best-validation checkpointing and data-before-model order."""

import copy
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birds import train as train_module

_NUM_CLASSES = 3
_NUM_FEATURES = 4


def _tiny_model() -> nn.Module:
    return nn.Linear(_NUM_FEATURES, _NUM_CLASSES)


def _synthetic_loader(num_samples: int, batch_size: int) -> "DataLoader[Any]":
    images = torch.randn(num_samples, _NUM_FEATURES)
    labels = torch.randint(0, _NUM_CLASSES, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_fit_saves_a_checkpoint(tmp_path: Path) -> None:
    model = _tiny_model()
    train_loader = _synthetic_loader(num_samples=8, batch_size=4)
    val_loader = _synthetic_loader(num_samples=4, batch_size=4)
    checkpoint_path = tmp_path / "small.pt"

    result = train_module.fit(
        model,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        epochs=2,
        lr=1e-3,
        weight_decay=0.0,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert len(result.history) == 2


def test_fit_keeps_the_best_epoch_not_the_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scripted validation accuracies [0.5, 0.9, 0.7]: epoch 2 (1-based) wins."""
    model = _tiny_model()
    train_loader = _synthetic_loader(num_samples=8, batch_size=4)
    val_loader = _synthetic_loader(num_samples=4, batch_size=4)
    checkpoint_path = tmp_path / "small.pt"

    scripted_accuracies = iter([0.5, 0.9, 0.7])
    snapshots_by_epoch: dict[int, dict[str, torch.Tensor]] = {}
    call_count = 0

    def fake_top1_accuracy(
        model: nn.Module, loader: "DataLoader[Any]", device: torch.device
    ) -> float:
        nonlocal call_count
        call_count += 1
        snapshots_by_epoch[call_count] = copy.deepcopy(model.state_dict())
        return next(scripted_accuracies)

    monkeypatch.setattr(train_module, "top1_accuracy", fake_top1_accuracy)
    result = train_module.fit(
        model,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        epochs=3,
        lr=1e-3,
        weight_decay=0.0,
        checkpoint_path=checkpoint_path,
    )

    assert result.best_epoch == 2
    assert result.best_val_accuracy == 0.9

    saved_state = torch.load(checkpoint_path)
    expected_state = snapshots_by_epoch[2]
    for key, expected_value in expected_state.items():
        assert torch.equal(saved_state[key], expected_value)


def test_fit_signature_has_no_test_split_parameter() -> None:
    parameters = list(inspect.signature(train_module.fit).parameters)

    assert parameters == [
        "model",
        "train_loader",
        "val_loader",
        "device",
        "epochs",
        "lr",
        "weight_decay",
        "checkpoint_path",
    ]


def _patch_main(monkeypatch: pytest.MonkeyPatch, calls: list[str], *, data_missing: bool) -> None:
    def fake_make_data_loader(split: str, *args: Any, **kwargs: Any) -> str:
        calls.append(f"data:{split}")
        if data_missing:
            raise FileNotFoundError("run `python -m birds.download`")
        return split

    def fake_require_cuda() -> torch.device:
        calls.append("cuda")
        return torch.device("cpu")

    def fake_build_model(name: str, pretrained: bool = True) -> nn.Module:
        calls.append("model")
        return _tiny_model()

    monkeypatch.setattr("sys.argv", ["train", "--model", "small"])
    monkeypatch.setattr(train_module, "make_data_loader", fake_make_data_loader)
    monkeypatch.setattr(train_module, "require_cuda", fake_require_cuda)
    monkeypatch.setattr(train_module, "build_model", fake_build_model)
    monkeypatch.setattr(train_module, "fit", lambda *args, **kwargs: calls.append("fit"))


def test_main_builds_data_loaders_before_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _patch_main(monkeypatch, calls, data_missing=False)

    train_module.main()

    assert calls == ["data:train", "data:val", "cuda", "model", "fit"]


def test_main_builds_no_model_when_the_dataset_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _patch_main(monkeypatch, calls, data_missing=True)

    with pytest.raises(FileNotFoundError, match="birds.download"):
        train_module.main()

    assert "model" not in calls
    assert "cuda" not in calls
