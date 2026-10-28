"""Shared record type and dataset key for per-photo predictions; imports nothing heavy."""

from dataclasses import dataclass

DATASET_NAME = "cub-200-2011"


@dataclass(frozen=True)
class PhotoPrediction:
    """One photo's top-1 outcome and the model's confidence in it."""

    photo_id: str
    true_label: int
    predicted_label: int
    is_correct: bool
    max_probability: float
    top2_margin: float
