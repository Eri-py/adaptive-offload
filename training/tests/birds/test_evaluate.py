"""Tests for `birds.evaluate`: accuracy on known predictions, timing sanity.

No GPU, no real dataset, no downloaded weights. `evaluate` is imported as a
module (not `from birds.evaluate import test_accuracy`) so pytest never
mistakes `evaluate.test_accuracy` for a test function of this file's own.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birds import evaluate as ev


def test_top1_accuracy_matches_hand_computed_value() -> None:
    """`nn.Identity` returns its input unchanged, so the "logits" fed in are
    the model's exact predictions by construction — accuracy is then just
    comparing argmax(logits) to the labels by hand.

    logits -> argmax: [1, 0, 2, 1]; labels: [1, 0, 0, 2].
    Matches: sample 0 and 1 only, so accuracy is 2/4 = 0.5.
    """
    model = nn.Identity()
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

    accuracy = ev.test_accuracy(model, loader, torch.device("cpu"))

    assert accuracy == 0.5


def test_mean_latency_ms_is_positive_and_finite() -> None:
    device = torch.device("cpu")
    model = nn.Linear(4, 3)
    images = torch.randn(6, 4)
    labels = torch.randint(0, 3, (6,))
    dataset = TensorDataset(images, labels)

    latency_ms = ev.mean_latency_ms(model, dataset, device, warmup=2)

    assert latency_ms > 0
    assert math.isfinite(latency_ms)


def test_mean_latency_ms_warmup_exceeds_dataset_size() -> None:
    """`warmup` larger than the dataset must still work, cycling through items."""
    device = torch.device("cpu")
    model = nn.Linear(4, 3)
    dataset = TensorDataset(torch.randn(3, 4), torch.randint(0, 3, (3,)))

    latency_ms = ev.mean_latency_ms(model, dataset, device, warmup=10)

    assert latency_ms > 0
    assert math.isfinite(latency_ms)
