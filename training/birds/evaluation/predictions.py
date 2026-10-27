"""Per-photo predictions: what a model said about each photo, with its confidence."""

from dataclasses import dataclass
from typing import Any, get_args

import torch
from common.db import get_engine
from sqlalchemy import Engine
from torch import nn
from torch.utils.data import DataLoader

from birds.config import TRAIN_SETTINGS, WEIGHTS_DIR, ModelName
from birds.dataset.splits import Split, eval_transform, make_data_loader, split_labels
from birds.evaluation.evaluate import _MODEL_NAMES, _load_checkpoint
from birds.modeling.build import require_cuda
from shared import tracking

DATASET_NAME = "cub-200-2011"
_SPLITS: tuple[Split, ...] = get_args(Split)


@dataclass(frozen=True)
class PhotoPrediction:
    """One photo's top-1 outcome and the model's confidence in it."""

    photo_id: str
    true_label: int
    predicted_label: int
    is_correct: bool
    max_probability: float
    top2_margin: float


def predict_photos(
    model: nn.Module,
    loader: "DataLoader[Any]",
    device: torch.device,
    photo_ids: list[str],
    expected_labels: list[int],
) -> list[PhotoPrediction]:
    """Per-photo records in loader order; `loader` must be unshuffled over the split of `photo_ids`.

    `expected_labels` are the split's labels in `photo_ids` order; any disagreement with the
    labels the loader yields means the order is not what `photo_ids` assumes, so it raises.
    """
    if len(photo_ids) != len(expected_labels):
        raise ValueError(f"{len(photo_ids)} photo ids but {len(expected_labels)} expected labels")
    model.eval()
    true_labels: list[int] = []
    predicted: list[int] = []
    max_probs: list[float] = []
    margins: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            probs = model(images.to(device)).softmax(dim=1)
            top2 = probs.topk(2, dim=1).values
            true_labels += labels.tolist()
            predicted += probs.argmax(dim=1).tolist()
            max_probs += top2[:, 0].tolist()
            margins += (top2[:, 0] - top2[:, 1]).tolist()
    if true_labels != expected_labels:
        raise ValueError(
            "Loader label order does not match the split's labels "
            f"({len(true_labels)} collected vs {len(expected_labels)} expected); "
            "refusing to attach photo ids to the wrong predictions."
        )
    return [
        PhotoPrediction(
            photo_id=photo_id,
            true_label=true,
            predicted_label=pred,
            is_correct=pred == true,
            max_probability=max_prob,
            top2_margin=margin,
        )
        for photo_id, true, pred, max_prob, margin in zip(
            photo_ids, true_labels, predicted, max_probs, margins, strict=True
        )
    ]


def record_split(
    engine: Engine,
    model: nn.Module,
    model_name: ModelName,
    split: Split,
    device: torch.device,
    training_run_id: str | None,
) -> int:
    """Computes and stores `model_name`'s records over `split`; returns the row count."""
    from birds.evaluation.store import write_predictions  # Deferred: store imports this module.

    labeled = dict(zip(_SPLITS, split_labels(), strict=True))[split]
    photo_ids = [photo_id for photo_id, _ in labeled]
    loader = make_data_loader(
        split, eval_transform(model_name), TRAIN_SETTINGS[model_name].batch_size, shuffle=False
    )
    records = predict_photos(model, loader, device, photo_ids, [label for _, label in labeled])
    write_predictions(engine, DATASET_NAME, split, model_name, training_run_id, records)
    return len(records)


def main() -> None:
    split_labels()  # Raises the download-command error before any GPU or checkpoint work.
    device = require_cuda()
    engine = get_engine()
    for name in _MODEL_NAMES:
        model = _load_checkpoint(name, device)
        run_id = tracking.read_run_id(WEIGHTS_DIR / f"{name}.pt")
        for split in _SPLITS:
            count = record_split(engine, model, name, split, device, run_id)
            print(f"{name:<8} {split:<6} {count:>6} rows")


if __name__ == "__main__":
    main()
