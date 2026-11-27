"""Runs the large model on a preprocessed image and derives the confidence measures."""

import time
from dataclasses import dataclass

import torch
from torch import nn

TOP_K = 3


@dataclass(frozen=True)
class Prediction:
    index: int
    name: str
    probability: float


@dataclass(frozen=True)
class Classification:
    top1: Prediction
    max_probability: float
    margin: float  # top-1 probability minus top-2 probability
    top3: list[Prediction]
    compute_ms: float  # forward pass only


def classify(
    model: nn.Module, species: list[str], tensor: torch.Tensor, device: torch.device
) -> Classification:
    """Classifies one (3, H, W) normalized tensor; `compute_ms` times only the forward pass."""
    batch = tensor.unsqueeze(0).to(device)
    on_cuda = device.type == "cuda"
    with torch.inference_mode():
        # CUDA kernels launch asynchronously, so sync on both sides or the timing is meaningless.
        if on_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(batch)
        if on_cuda:
            torch.cuda.synchronize(device)
        compute_ms = (time.perf_counter() - start) * 1000.0

        probabilities = torch.softmax(logits[0].float(), dim=0)
        top_probabilities, top_indices = torch.topk(probabilities, TOP_K)

    top3 = [
        Prediction(index=index, name=species[index], probability=probability)
        for index, probability in zip(top_indices.tolist(), top_probabilities.tolist(), strict=True)
    ]
    return Classification(
        top1=top3[0],
        max_probability=top3[0].probability,
        margin=top3[0].probability - top3[1].probability,
        top3=top3,
        compute_ms=compute_ms,
    )
