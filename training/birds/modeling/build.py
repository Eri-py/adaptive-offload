"""Builds the small/large bird classifiers (200-way head) and gates training on a GPU."""

from pathlib import Path
from typing import cast

import torch
from torch import nn
from torchvision.models import WeightsEnum, convnext_base, mobilenet_v3_large

from birds.config import MODEL_WEIGHTS, NUM_CLASSES, WEIGHTS_DIR, ModelName

_TRAIN_COMMAND = "python -m birds.train --model {name}"


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


def require_checkpoint(name: ModelName) -> Path:
    """Path of `name`'s trained checkpoint; raises an actionable error if it is missing."""
    checkpoint_path = WEIGHTS_DIR / f"{name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for the '{name}' model at {checkpoint_path}. "
            f"Run `{_TRAIN_COMMAND.format(name=name)}` first."
        )
    return checkpoint_path


def load_checkpoint(name: ModelName, device: torch.device) -> nn.Module:
    """Builds `name`'s model (untrained head) and loads its trained checkpoint."""
    checkpoint_path = require_checkpoint(name)
    model = build_model(name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def require_cuda() -> torch.device:
    """Returns the CUDA device, refusing to proceed without a GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU available. Training requires a GPU; run this on a "
            "machine with one instead."
        )
    return torch.device("cuda")
