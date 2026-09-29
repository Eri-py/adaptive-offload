"""Accuracy metric shared by validation and test evaluation."""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def top1_accuracy(model: nn.Module, loader: "DataLoader[Any]", device: torch.device) -> float:
    """Top-1 accuracy over `loader`, in plain fp32 so val and test are measured alike.

    `model` must already be on `device`; only the batches are moved.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    return correct / total
