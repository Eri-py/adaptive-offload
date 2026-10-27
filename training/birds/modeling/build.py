"""Builds the small/large bird classifiers (200-way head) and gates training on a GPU."""

from typing import cast

import torch
from torch import nn
from torchvision.models import WeightsEnum, convnext_base, mobilenet_v3_large

from birds.config import MODEL_WEIGHTS, NUM_CLASSES, ModelName


def weights_for(name: ModelName) -> WeightsEnum:
    """The ImageNet-pretrained weights enum for the given model name."""
    return MODEL_WEIGHTS[name]


def _replace_head(classifier: nn.Sequential, index: int) -> None:
    """Swaps the `Linear` at `index` for a fresh NUM_CLASSES-way one."""
    old_head = cast(nn.Linear, classifier[index])
    classifier[index] = nn.Linear(old_head.in_features, NUM_CLASSES)


def build_model(name: ModelName, pretrained: bool = True) -> nn.Module:
    """Builds the named torchvision model with a NUM_CLASSES-way classifier head."""
    weights = weights_for(name) if pretrained else None

    if name == "small":
        small_model: nn.Module = mobilenet_v3_large(weights=weights)
        _replace_head(cast(nn.Sequential, small_model.classifier), index=3)
        return small_model

    large_model: nn.Module = convnext_base(weights=weights)
    _replace_head(cast(nn.Sequential, large_model.classifier), index=2)
    return large_model


def require_cuda() -> torch.device:
    """Returns the CUDA device, refusing to proceed without a GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU available. Training requires a GPU; run this on a "
            "machine with one instead."
        )
    return torch.device("cuda")
