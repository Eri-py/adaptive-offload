"""Tests for `birds.evaluation.predictions`: hand-computed per-photo records and alignment."""

from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birds.evaluation.predictions import PhotoPrediction, predict_photos

_LOGITS = torch.tensor(
    [
        [0.0, 5.0, 1.0],  # argmax 1, label 1: correct
        [5.0, 0.0, 1.0],  # argmax 0, label 2: wrong
        [0.0, 0.0, 0.0],  # uniform: probabilities 1/3 each, margin 0
    ]
)
_LABELS = [1, 2, 0]
_IDS = ["10", "11", "12"]


def _loader(batch_size: int = 2) -> "DataLoader[Any]":
    return DataLoader(TensorDataset(_LOGITS, torch.tensor(_LABELS)), batch_size=batch_size)


def _run(batch_size: int = 2) -> list[PhotoPrediction]:
    return predict_photos(nn.Identity(), _loader(batch_size), torch.device("cpu"), _IDS, _LABELS)


def test_records_match_hand_computed_values() -> None:
    probs = _LOGITS.softmax(dim=1)
    records = _run()

    assert [r.photo_id for r in records] == _IDS
    assert [r.true_label for r in records] == _LABELS
    assert [r.predicted_label for r in records] == [1, 0, 0]
    assert [r.is_correct for r in records] == [True, False, True]
    assert records[0].max_probability == pytest.approx(probs[0, 1].item())
    assert records[0].top2_margin == pytest.approx((probs[0, 1] - probs[0, 2]).item())
    assert records[1].max_probability == pytest.approx(probs[1, 0].item())
    assert records[1].top2_margin == pytest.approx((probs[1, 0] - probs[1, 2]).item())
    assert records[2].max_probability == pytest.approx(1 / 3)
    assert records[2].top2_margin == pytest.approx(0.0, abs=1e-7)


def test_batch_size_does_not_change_the_records() -> None:
    assert _run(batch_size=1) == _run(batch_size=3)


def test_misaligned_label_order_raises() -> None:
    swapped_ids_labels = [_LABELS[1], _LABELS[0], _LABELS[2]]

    with pytest.raises(ValueError, match="does not match"):
        predict_photos(nn.Identity(), _loader(), torch.device("cpu"), _IDS, swapped_ids_labels)


def test_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="photo ids"):
        predict_photos(nn.Identity(), _loader(), torch.device("cpu"), _IDS[:2], _LABELS)


def test_loader_yielding_fewer_photos_than_the_split_raises() -> None:
    short = DataLoader(TensorDataset(_LOGITS[:2], torch.tensor(_LABELS[:2])), batch_size=2)

    with pytest.raises(ValueError, match="does not match"):
        predict_photos(nn.Identity(), short, torch.device("cpu"), _IDS, _LABELS)
