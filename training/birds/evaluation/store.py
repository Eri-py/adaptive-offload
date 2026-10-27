"""Persist and read back per-photo bird predictions in the shared `bird_predictions` table."""

from collections.abc import Sequence

import pandas as pd
from common.models import BirdPrediction
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from birds.evaluation.predictions import PhotoPrediction


def write_predictions(
    engine: Engine,
    dataset: str,
    split: str,
    model_name: str,
    training_run_id: str | None,
    records: Sequence[PhotoPrediction],
) -> None:
    """Replace the rows for exactly (dataset, split, model_name) with `records`."""
    with Session(engine) as session:
        session.execute(
            delete(BirdPrediction).where(
                BirdPrediction.dataset == dataset,
                BirdPrediction.split == split,
                BirdPrediction.model_name == model_name,
            )
        )
        session.add_all(
            BirdPrediction(
                dataset=dataset,
                photo_id=record.photo_id,
                model_name=model_name,
                split=split,
                true_label=record.true_label,
                predicted_label=record.predicted_label,
                is_correct=record.is_correct,
                max_probability=record.max_probability,
                top2_margin=record.top2_margin,
                training_run_id=training_run_id,
            )
            for record in records
        )
        session.commit()


def read_predictions(engine: Engine, dataset: str, split: str, model_name: str) -> pd.DataFrame:
    """Stored rows for one (dataset, split, model_name), ordered by numeric photo id."""
    query = select(BirdPrediction).where(
        BirdPrediction.dataset == dataset,
        BirdPrediction.split == split,
        BirdPrediction.model_name == model_name,
    )
    frame = pd.read_sql(query, engine)
    # Photo ids are stringified ints; "10" must follow "2", matching splits.py's ordering.
    return frame.sort_values("photo_id", key=lambda ids: ids.astype(int), ignore_index=True)
