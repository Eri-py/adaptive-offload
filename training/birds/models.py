"""Builds the small/large bird classifiers and gates training on a GPU.

Both torchvision backbones ship with a 1000-way ImageNet head; `build_model`
swaps it for a fresh `Linear(in_features, NUM_CLASSES)` classifying the
102 Oxford Flowers species instead.
"""

from typing import cast

import torch
from torch import nn
from torchvision.models import WeightsEnum, convnext_base, mobilenet_v3_large

from birds.config import MODEL_WEIGHTS, NUM_CLASSES, ModelName


def weights_for(name: ModelName) -> WeightsEnum:
    """The ImageNet-pretrained weights enum for the given model name."""
    return MODEL_WEIGHTS[name]


def _replace_head(classifier: nn.Sequential, index: int) -> None:
    """Swaps the `Linear` layer at `index` for a fresh 102-way one, in place."""
    old_head = cast(nn.Linear, classifier[index])
    classifier[index] = nn.Linear(old_head.in_features, NUM_CLASSES)


def build_model(name: ModelName, pretrained: bool = True) -> nn.Module:
    """Builds the named torchvision model with a 102-way classifier head."""
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
