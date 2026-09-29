"""Trains one bird classifier, keeping its best validation checkpoint (never sees test)."""

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from birds.config import EXPERIMENT_NAME, NUM_CLASSES, SEED, TRAIN_SETTINGS, WEIGHTS_DIR, ModelName
from birds.data import eval_transform, make_data_loader, train_transform
from birds.metrics import top1_accuracy
from birds.models import build_model, require_cuda
from shared import tracking


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


def fit(
    model: nn.Module,
    train_loader: "DataLoader[Any]",
    val_loader: "DataLoader[Any]",
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: Path,
    *,
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> FitResult:
    """Trains `model`, saving the best-validation state dict; bf16 autocast on CUDA, no scaler.

    `on_epoch_end`, if given, is called with each epoch's result after its validation.
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
        val_accuracy = top1_accuracy(model, val_loader, device)
        print(
            f"epoch {epoch}/{epochs} - train loss {train_loss:.4f} - "
            f"val accuracy {val_accuracy:.4f}"
        )
        epoch_result = EpochResult(epoch=epoch, train_loss=train_loss, val_accuracy=val_accuracy)
        history.append(epoch_result)
        if on_epoch_end is not None:
            on_epoch_end(epoch_result)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return FitResult(best_val_accuracy=best_val_accuracy, best_epoch=best_epoch, history=history)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a bird classifier.")
    parser.add_argument("--model", choices=["small", "large"], required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_name: ModelName = args.model

    _seed_everything(SEED)
    settings = TRAIN_SETTINGS[model_name]

    # Loaders come first: a missing dataset must fail before any weights download.
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

    device = require_cuda()
    model = build_model(model_name, pretrained=True)

    checkpoint_path = WEIGHTS_DIR / f"{model_name}.pt"
    params = {
        "model": model_name,
        "epochs": settings.epochs,
        "lr": settings.lr,
        "weight_decay": settings.weight_decay,
        "batch_size": settings.batch_size,
        "seed": SEED,
        "dataset": "cub200",
        "num_classes": NUM_CLASSES,
    }
    with tracking.run(f"train-{model_name}", params, experiment=EXPERIMENT_NAME) as handle:
        # Before fit, not after: fit overwrites the checkpoint the moment an epoch improves,
        # so an interrupted run would otherwise leave a new model beside the old run's id.
        tracking.record_run_id(checkpoint_path, handle.id)

        def log_epoch(epoch_result: EpochResult) -> None:
            tracking.log_metrics(
                handle,
                {
                    "train_loss": epoch_result.train_loss,
                    "val_accuracy": epoch_result.val_accuracy,
                },
                step=epoch_result.epoch,
            )

        result = fit(
            model,
            train_loader,
            val_loader,
            device,
            epochs=settings.epochs,
            lr=settings.lr,
            weight_decay=settings.weight_decay,
            checkpoint_path=checkpoint_path,
            on_epoch_end=log_epoch,
        )
        tracking.log_metrics(
            handle,
            {
                "best_val_accuracy": result.best_val_accuracy,
                "best_epoch": float(result.best_epoch),
            },
        )
    print(result)


if __name__ == "__main__":
    main()
