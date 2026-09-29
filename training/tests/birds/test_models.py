"""Tests for `flowers.models`: classifier head shape and the GPU gate.

`build_model` is always called with `pretrained=False` here so these tests
never download ImageNet weights.
"""

import pytest
import torch

from flowers import models as models_module
from flowers.config import NUM_CLASSES


def test_build_small_model_outputs_102_classes() -> None:
    model = models_module.build_model("small", pretrained=False)

    output = model(torch.rand(2, 3, 224, 224))

    assert output.shape == (2, NUM_CLASSES)


def test_build_large_model_outputs_102_classes() -> None:
    model = models_module.build_model("large", pretrained=False)

    output = model(torch.rand(2, 3, 224, 224))

    assert output.shape == (2, NUM_CLASSES)


def test_require_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        models_module.require_cuda()


def test_require_cuda_returns_cuda_device_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = models_module.require_cuda()

    assert device.type == "cuda"
