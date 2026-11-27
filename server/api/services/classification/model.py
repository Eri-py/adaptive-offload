"""Loads the large bird classifier and the species list; both happen once at startup."""

import json
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torchvision.models import convnext_base

NUM_CLASSES = 200
_HEAD_INDEX = 2


def require_cuda() -> torch.device:
    """The CUDA device; no CPU fallback because the reported compute time is a GPU figure."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU available. The server reports GPU compute time, so it refuses "
            "to start on CPU; run it on a machine with a CUDA GPU."
        )
    return torch.device("cuda")


def build_large_model() -> nn.Module:
    """ConvNeXt-Base with a 200-way head, mirroring training's `birds/modeling/build.py`."""
    model: nn.Module = convnext_base(weights=None)
    classifier = cast(nn.Sequential, model.classifier)
    old_head = cast(nn.Linear, classifier[_HEAD_INDEX])
    classifier[_HEAD_INDEX] = nn.Linear(old_head.in_features, NUM_CLASSES)
    return model


def load_model(checkpoint_path: Path) -> nn.Module:
    """Builds the model, loads its trained weights, and returns it in eval mode on CUDA."""
    device = require_cuda()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing large-model checkpoint at {checkpoint_path}. "
            "Set BIRDS_LARGE_CHECKPOINT or run the training export."
        )
    model = build_large_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model.to(device)


def load_species(labels_path: Path) -> list[str]:
    """The species names in class-index order; must be exactly NUM_CLASSES strings."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing species labels at {labels_path}. Set BIRDS_SPECIES_LABELS."
        )
    species = json.loads(labels_path.read_text())
    if (
        not isinstance(species, list)
        or len(species) != NUM_CLASSES
        or not all(isinstance(name, str) for name in species)
    ):
        raise ValueError(f"{labels_path} must be a JSON array of exactly {NUM_CLASSES} strings.")
    return cast(list[str], species)
