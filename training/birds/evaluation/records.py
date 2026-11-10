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


def numeric_photo_id(photo_id: str) -> int:
    """Integer value of a photo id, the one place that assumes ids are stringified ints.

    "10" must follow "2", matching splits.py's ordering; a plain string sort would not.
    Another dataset sharing the table with non-numeric ids needs its own ordering.
    """
    try:
        return int(photo_id)
    except ValueError:
        raise ValueError(
            f"photo id {photo_id!r} is not an integer; stored predictions are ordered "
            "by numeric photo id, which only holds for datasets with integer ids"
        ) from None
