"""Per-photo predictions: what a model said about each photo, with its confidence."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


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
