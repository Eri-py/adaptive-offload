"""Trains one flower classifier, keeping only its best validation checkpoint.

`fit` never receives a test-split loader — by construction, this module has
no way to touch the test split, satisfying the spec's train/val/test
isolation rule (the test split is used once, in `evaluate.py`).
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from flowers.config import SEED, TRAIN_SETTINGS, WEIGHTS_DIR, ModelName
from flowers.data import eval_transform, make_data_loader, train_transform
from flowers.models import build_model, require_cuda


@dataclass
class EpochResult:
    epoch: int  # 1-based
    train_loss: float
    val_accuracy: float


@dataclass
class FitResult:
    best_val_accuracy: float
    best_epoch: int  # 1-based
    history: list[EpochResult] = field(default_factory=list)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate_accuracy(
    model: nn.Module, loader: "DataLoader[Any]", device: torch.device
) -> float:
    """Top-1 accuracy of `model` over every batch in `loader`."""
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


def fit(
    model: nn.Module,
    train_loader: "DataLoader[Any]",
    val_loader: "DataLoader[Any]",
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: Path,
) -> FitResult:
    """Trains `model`, saving the state dict with the best validation accuracy.

    Mixed precision uses bf16 (no `GradScaler`: unlike fp16, bf16's exponent
    range matches fp32's, so it needs no loss scaling) when `device` is CUDA;
    it's a no-op autocast on CPU.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.type == "cuda"

    best_val_accuracy = -1.0
    best_epoch = -1
    history: list[EpochResult] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        scheduler.step()

        train_loss = running_loss / num_batches
        val_accuracy = _evaluate_accuracy(model, val_loader, device)
        print(
            f"epoch {epoch}/{epochs} - train loss {train_loss:.4f} - "
            f"val accuracy {val_accuracy:.4f}"
        )
        history.append(EpochResult(epoch=epoch, train_loss=train_loss, val_accuracy=val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return FitResult(best_val_accuracy=best_val_accuracy, best_epoch=best_epoch, history=history)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a flower classifier.")
    parser.add_argument("--model", choices=["small", "large"], required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_name: ModelName = args.model

    _seed_everything(SEED)
    device = require_cuda()
    settings = TRAIN_SETTINGS[model_name]

    model = build_model(model_name, pretrained=True)
    shuffle_generator = torch.Generator().manual_seed(SEED)
    train_loader = make_data_loader(
        "train",
        train_transform(),
        settings.batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = make_data_loader(
        "val", eval_transform(model_name), settings.batch_size, shuffle=False
    )

    checkpoint_path = WEIGHTS_DIR / f"{model_name}.pt"
    result = fit(
        model,
        train_loader,
        val_loader,
        device,
        epochs=settings.epochs,
        lr=settings.lr,
        weight_decay=settings.weight_decay,
        checkpoint_path=checkpoint_path,
    )
    print(result)


if __name__ == "__main__":
    main()
