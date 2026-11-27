"""Tests for app startup wiring."""

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from api.core import app as app_module


class CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.last_shape: tuple[int, ...] = ()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_shape = tuple(batch.shape)
        return torch.zeros(batch.shape[0], 200)


def test_startup_runs_one_warmup_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    model = CountingModel()
    monkeypatch.setattr(app_module, "load_model", lambda _path: model)
    monkeypatch.setattr(app_module, "load_species", lambda _path: ["x"] * 200)

    with TestClient(app_module.create_app(load_resources=True)):
        assert model.calls == 1
        assert model.last_shape == (1, 3, 224, 224)


def test_stubbed_app_does_not_warm_up() -> None:
    model = CountingModel()
    application = app_module.create_app(load_resources=False)
    application.state.model = model

    with TestClient(application):
        assert model.calls == 0
