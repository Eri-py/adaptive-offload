"""Tests for `flowers.train`: best-validation checkpointing, not last-epoch.

Uses a tiny CPU model and synthetic tensors — no GPU, no real dataset, no
downloaded weights.
"""

import copy
import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flowers import train as train_module

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


def test_fit_keeps_the_best_epoch_not_the_last(tmp_path: Path) -> None:
    """Scripted validation accuracies [0.5, 0.9, 0.7]: epoch 2 (1-based) wins."""
    model = _tiny_model()
    train_loader = _synthetic_loader(num_samples=8, batch_size=4)
    val_loader = _synthetic_loader(num_samples=4, batch_size=4)
    checkpoint_path = tmp_path / "small.pt"

    scripted_accuracies = iter([0.5, 0.9, 0.7])
    snapshots_by_epoch: dict[int, dict[str, torch.Tensor]] = {}
    call_count = 0

    def fake_evaluate_accuracy(
        model: nn.Module, loader: "DataLoader[Any]", device: torch.device
    ) -> float:
        nonlocal call_count
        call_count += 1
        snapshots_by_epoch[call_count] = copy.deepcopy(model.state_dict())
        return next(scripted_accuracies)

    original_evaluate_accuracy = train_module._evaluate_accuracy
    train_module._evaluate_accuracy = fake_evaluate_accuracy
    try:
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
    finally:
        train_module._evaluate_accuracy = original_evaluate_accuracy

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
