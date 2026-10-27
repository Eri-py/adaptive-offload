"""Tests for `birds.evaluation.store`: round trip, scoped replacement, and numeric ordering."""

from sqlalchemy import Engine

from birds.evaluation.predictions import PhotoPrediction
from birds.evaluation.store import read_predictions, write_predictions

_DATASET = "cub"


def _record(photo_id: str, predicted: int = 1) -> PhotoPrediction:
    return PhotoPrediction(
        photo_id=photo_id,
        true_label=1,
        predicted_label=predicted,
        is_correct=predicted == 1,
        max_probability=0.75,
        top2_margin=0.5,
    )


def test_write_then_read_round_trips(postgres_engine: Engine) -> None:
    records = [_record("2"), _record("10", predicted=3)]
    write_predictions(postgres_engine, _DATASET, "val", "small", "run-1", records)

    frame = read_predictions(postgres_engine, _DATASET, "val", "small")

    assert frame["photo_id"].tolist() == ["2", "10"]
    assert frame["predicted_label"].tolist() == [1, 3]
    assert frame["is_correct"].tolist() == [True, False]
    assert frame["max_probability"].tolist() == [0.75, 0.75]
    assert set(frame["training_run_id"]) == {"run-1"}
    assert set(frame["split"]) == {"val"}


def test_rewrite_replaces_only_that_split_and_model(postgres_engine: Engine) -> None:
    write_predictions(postgres_engine, _DATASET, "val", "small", None, [_record("1"), _record("2")])
    write_predictions(postgres_engine, _DATASET, "test", "small", None, [_record("3")])
    write_predictions(postgres_engine, _DATASET, "val", "large", None, [_record("1"), _record("2")])

    write_predictions(postgres_engine, _DATASET, "val", "small", None, [_record("2", predicted=5)])

    replaced = read_predictions(postgres_engine, _DATASET, "val", "small")
    assert replaced["photo_id"].tolist() == ["2"]
    assert replaced["predicted_label"].tolist() == [5]
    assert len(read_predictions(postgres_engine, _DATASET, "test", "small")) == 1
    assert len(read_predictions(postgres_engine, _DATASET, "val", "large")) == 2


def test_read_orders_photo_ids_numerically(postgres_engine: Engine) -> None:
    records = [_record(photo_id) for photo_id in ["10", "2", "100", "1"]]
    write_predictions(postgres_engine, _DATASET, "val", "small", None, records)

    frame = read_predictions(postgres_engine, _DATASET, "val", "small")

    assert frame["photo_id"].tolist() == ["1", "2", "10", "100"]
