"""Tests for `birds.models`: 200-way head shape and the GPU gate (never downloads weights)."""

import pytest
import torch

from birds import models as models_module
from birds.config import NUM_CLASSES


def test_build_small_model_outputs_num_classes() -> None:
    model = models_module.build_model("small", pretrained=False)

    output = model(torch.rand(2, 3, 224, 224))

    assert output.shape == (2, 200)
    assert NUM_CLASSES == 200


def test_build_large_model_outputs_num_classes() -> None:
    model = models_module.build_model("large", pretrained=False)

    output = model(torch.rand(2, 3, 224, 224))

    assert output.shape == (2, 200)
    assert NUM_CLASSES == 200


def test_require_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        models_module.require_cuda()


def test_require_cuda_returns_cuda_device_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = models_module.require_cuda()

    assert device.type == "cuda"
