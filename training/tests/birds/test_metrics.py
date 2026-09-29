"""Tests for `birds.metrics`: top-1 accuracy on a hand-computed case."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birds.metrics import top1_accuracy


def test_top1_accuracy_matches_hand_computed_value() -> None:
    """Identity model makes the inputs the logits: argmax [1, 0, 2, 1] vs labels [1, 0, 0, 2]."""
    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0],  # argmax 1, label 1: correct
            [5.0, 0.0, 1.0],  # argmax 0, label 0: correct
            [0.0, 1.0, 5.0],  # argmax 2, label 0: wrong
            [0.0, 5.0, 1.0],  # argmax 1, label 2: wrong
        ]
    )
    labels = torch.tensor([1, 0, 0, 2])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    assert top1_accuracy(nn.Identity(), loader, torch.device("cpu")) == 0.5
