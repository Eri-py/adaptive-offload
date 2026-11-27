import math
import time

import pytest
import torch
from torch import nn

from api.services.classification.classify import classify

# Logits are logs of these weights, so softmax returns weight / sum(weights) exactly:
# weights 2, 8, 1, 4, 1 sum to 16 -> probabilities 0.125, 0.5, 0.0625, 0.25, 0.0625.
_WEIGHTS = [2.0, 8.0, 1.0, 4.0, 1.0]
_SPECIES = ["zero", "one", "two", "three", "four"]


class _WeightedStub(nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[math.log(w) for w in _WEIGHTS]]).repeat(batch.shape[0], 1)


@pytest.fixture
def image_tensor() -> torch.Tensor:
    return torch.zeros(3, 224, 224)


def test_top1_and_max_probability(image_tensor: torch.Tensor) -> None:
    result = classify(_WeightedStub(), _SPECIES, image_tensor, torch.device("cpu"))

    assert result.top1.index == 1
    assert result.top1.name == "one"
    assert result.max_probability == pytest.approx(0.5)  # 8/16


def test_margin_is_top1_minus_top2(image_tensor: torch.Tensor) -> None:
    result = classify(_WeightedStub(), _SPECIES, image_tensor, torch.device("cpu"))

    assert result.margin == pytest.approx(0.25)  # 8/16 - 4/16


def test_top3_is_ordered_by_probability(image_tensor: torch.Tensor) -> None:
    result = classify(_WeightedStub(), _SPECIES, image_tensor, torch.device("cpu"))

    assert [(p.index, p.name) for p in result.top3] == [(1, "one"), (3, "three"), (0, "zero")]
    assert [p.probability for p in result.top3] == pytest.approx([0.5, 0.25, 0.125])


def test_compute_time_is_a_non_negative_millisecond_figure(image_tensor: torch.Tensor) -> None:
    result = classify(_WeightedStub(), _SPECIES, image_tensor, torch.device("cpu"))

    assert 0.0 <= result.compute_ms < 5000.0


def test_compute_time_covers_only_the_forward_pass(image_tensor: torch.Tensor) -> None:
    class _SlowStub(_WeightedStub):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            time.sleep(0.05)
            return super().forward(batch)

    result = classify(_SlowStub(), _SPECIES, image_tensor, torch.device("cpu"))

    assert result.compute_ms >= 50.0
