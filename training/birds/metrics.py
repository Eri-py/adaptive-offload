"""Loss and accuracy, measured one way for validation and test."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from birds.config import LABEL_SMOOTHING


@dataclass(frozen=True)
class EvalResult:
    """One evaluation pass: mean per-sample loss and top-1 accuracy."""

    loss: float
    accuracy: float


def loss_criterion() -> nn.Module:
    """The single loss used for training, validation and test, so the curves compare."""
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def loss_and_accuracy(
    model: nn.Module, loader: "DataLoader[Any]", device: torch.device
) -> EvalResult:
    """Mean loss and top-1 accuracy over `loader` in one fp32 pass; `model` already on `device`."""
    criterion = loss_criterion()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_size = int(labels.size(0))
            total_loss += criterion(outputs, labels).item() * batch_size
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += batch_size
    return EvalResult(loss=total_loss / total, accuracy=correct / total)
