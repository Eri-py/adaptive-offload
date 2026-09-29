"""Tests for `birds.evaluate`: missing checkpoint, per-model transform, timing sanity."""

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from birds import evaluate as ev


def test_mean_latency_ms_is_positive_and_finite() -> None:
    device = torch.device("cpu")
    model = nn.Linear(4, 3)
    images = torch.randn(6, 4)
    labels = torch.randint(0, 3, (6,))
    dataset = TensorDataset(images, labels)

    latency_ms = ev.mean_latency_ms(model, dataset, device, warmup=2)

    assert latency_ms > 0
    assert math.isfinite(latency_ms)


def test_mean_latency_ms_warmup_exceeds_dataset_size() -> None:
    """`warmup` larger than the dataset must still work, cycling through items."""
    device = torch.device("cpu")
    model = nn.Linear(4, 3)
    dataset = TensorDataset(torch.randn(3, 4), torch.randint(0, 3, (3,)))

    latency_ms = ev.mean_latency_ms(model, dataset, device, warmup=10)

    assert latency_ms > 0
    assert math.isfinite(latency_ms)


def test_load_checkpoint_missing_names_the_train_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "WEIGHTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=r"python -m birds\.train --model small"):
        ev._load_checkpoint("small", torch.device("cpu"))


def test_main_uses_each_models_own_eval_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    transform_names: list[str] = []
    loader_transforms: list[Any] = []

    def fake_eval_transform(name: str) -> str:
        transform_names.append(name)
        return f"transform-{name}"

    def fake_make_data_loader(split: str, transform: Any, *args: Any, **kwargs: Any) -> str:
        loader_transforms.append(transform)
        return "loader"

    monkeypatch.setattr(ev, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(ev, "eval_transform", fake_eval_transform)
    monkeypatch.setattr(ev, "make_data_loader", fake_make_data_loader)
    monkeypatch.setattr(ev, "load_test", lambda transform: transform)
    monkeypatch.setattr(ev, "_load_checkpoint", lambda name, device: nn.Identity())
    monkeypatch.setattr(ev, "top1_accuracy", lambda *args: 1.0)
    monkeypatch.setattr(ev, "mean_latency_ms", lambda *args: 1.0)

    ev.main()

    assert loader_transforms == ["transform-small", "transform-large"]
    assert transform_names.count("small") == 2 and transform_names.count("large") == 2
