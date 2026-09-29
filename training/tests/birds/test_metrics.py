"""Tests for `birds.metrics`: hand-computed loss/accuracy and the shared criterion."""

from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birds import evaluate, metrics, train
from birds.config import LABEL_SMOOTHING
from birds.metrics import loss_and_accuracy

_LOGITS = torch.tensor(
    [
        [0.0, 5.0, 1.0],  # argmax 1, label 1: correct
        [5.0, 0.0, 1.0],  # argmax 0, label 0: correct
        [0.0, 1.0, 5.0],  # argmax 2, label 0: wrong
        [0.0, 5.0, 1.0],  # argmax 1, label 2: wrong
    ]
)
_LABELS = torch.tensor([1, 0, 0, 2])


def _identity_loader(batch_size: int) -> "DataLoader[Any]":
    """Loader whose inputs are the logits directly, so `nn.Identity` is the model."""
    return DataLoader(TensorDataset(_LOGITS, _LABELS), batch_size=batch_size)


def test_accuracy_matches_hand_computed_value() -> None:
    result = loss_and_accuracy(nn.Identity(), _identity_loader(batch_size=2), torch.device("cpu"))

    assert result.accuracy == 0.5


def test_loss_matches_the_shared_criterion_over_the_whole_split() -> None:
    result = loss_and_accuracy(nn.Identity(), _identity_loader(batch_size=2), torch.device("cpu"))

    expected = metrics.loss_criterion()(_LOGITS, _LABELS).item()
    assert result.loss == pytest.approx(expected)


def test_loss_weights_batches_by_size_not_equally() -> None:
    """A 3+1 split must give the same mean as one batch of 4 — not (mean_of_3 + one) / 2."""
    device = torch.device("cpu")

    one_batch = loss_and_accuracy(nn.Identity(), _identity_loader(batch_size=4), device)
    uneven = loss_and_accuracy(nn.Identity(), _identity_loader(batch_size=3), device)

    assert uneven.loss == pytest.approx(one_batch.loss)
    assert uneven.accuracy == one_batch.accuracy


def test_the_criterion_is_label_smoothed() -> None:
    criterion = metrics.loss_criterion()

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == LABEL_SMOOTHING


def test_training_validation_and_test_share_one_metrics_implementation() -> None:
    """A second local implementation in `train` or `evaluate` would measure loss differently."""
    assert vars(train)["loss_and_accuracy"] is metrics.loss_and_accuracy
    assert vars(train)["loss_criterion"] is metrics.loss_criterion
    assert vars(evaluate)["loss_and_accuracy"] is metrics.loss_and_accuracy
