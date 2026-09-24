"""Integration tests for `router.extract_features`'s orchestration function.

Runs against real ephemeral Postgres (`postgres_engine` fixture, per
`training/tests/conftest.py`) and small synthetic JPEGs written to
`tmp_path`, with a stub `predict` so no real YOLO model loads — mirrors
`training/tests/datagen/cli/test_run_simulation.py`'s fake-inference-closure
pattern.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from common.models import FrameFeatures, SceneComplexity
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from router.extract_features import Predict, PredictFactory, extract_features
from router.feature_store import get_known_feature_file_names

DATASET = "test_dataset_extract_features"


def _write_images(tmp_path: Path, file_names: list[str]) -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    for file_name in file_names:
        cv2.imwrite(str(tmp_path / file_name), image)


def _seed_scene_complexity(engine: Engine, dataset: str, file_names: list[str]) -> None:
    with Session(engine) as session:
        session.add_all(
            SceneComplexity(dataset=dataset, file_name=file_name, scene_complexity=0.5)
            for file_name in file_names
        )
        session.commit()


def _make_stub_predict(
    by_file_name: dict[str, tuple[Sequence[float], Sequence[float]]],
    call_count: dict[str, int] | None = None,
) -> Predict:
    """A stub `Predict` returning canned (confidences, box_areas) per file name.

    Files not listed in `by_file_name` default to one detection with a fixed
    confidence/area — matches the fake-inference-closure shape used by
    `test_run_simulation.py`.
    """

    def predict(path: Path) -> tuple[Sequence[float], Sequence[float]]:
        if call_count is not None:
            call_count["n"] += 1
        return by_file_name.get(path.name, ([0.75], [0.1]))

    return predict


def _make_predict_factory_spy(predict: Predict, call_count: dict[str, int]) -> PredictFactory:
    """A `PredictFactory` recording how many times it was called before returning `predict`."""

    def factory() -> Predict:
        call_count["n"] += 1
        return predict

    return factory


def test_stores_one_row_per_frame_with_every_column_populated(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    file_names = ["a.jpg", "b.jpg"]
    _write_images(tmp_path, file_names)
    _seed_scene_complexity(postgres_engine, DATASET, file_names)

    summary = extract_features(
        postgres_engine, DATASET, tmp_path, lambda: _make_stub_predict({})
    )

    assert summary is not None
    assert get_known_feature_file_names(postgres_engine, DATASET) == set(file_names)

    with Session(postgres_engine) as session:
        for file_name in file_names:
            row = session.get(FrameFeatures, (DATASET, file_name))
            assert row is not None
            assert row.sharpness is not None
            assert row.brightness is not None
            assert row.contrast is not None
            assert row.colorfulness is not None
            assert row.entropy is not None
            assert row.detection_count == 1
            assert row.max_confidence == 0.75
            assert row.mean_confidence == 0.75
            assert row.min_confidence == 0.75
            assert row.mean_box_area == 0.1
            assert row.min_box_area == 0.1


def test_zero_detection_frame_is_stored_with_zeros(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    file_names = ["empty.jpg"]
    _write_images(tmp_path, file_names)
    _seed_scene_complexity(postgres_engine, DATASET, file_names)

    extract_features(
        postgres_engine,
        DATASET,
        tmp_path,
        lambda: _make_stub_predict({"empty.jpg": ([], [])}),
    )

    with Session(postgres_engine) as session:
        row = session.get(FrameFeatures, (DATASET, "empty.jpg"))
        assert row is not None
        assert row.detection_count == 0
        assert row.max_confidence == 0.0
        assert row.mean_confidence == 0.0
        assert row.min_confidence == 0.0
        assert row.mean_box_area == 0.0
        assert row.min_box_area == 0.0


def test_second_run_computes_nothing_new_and_leaves_existing_rows_unchanged(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    file_names = ["a.jpg", "b.jpg"]
    _write_images(tmp_path, file_names)
    _seed_scene_complexity(postgres_engine, DATASET, file_names)

    extract_features(postgres_engine, DATASET, tmp_path, lambda: _make_stub_predict({}))

    call_count = {"n": 0}
    # A predict that would raise if ever called proves the second run
    # doesn't recompute anything, not just that it leaves rows unchanged.
    def _predict_that_must_not_be_called(path: Path) -> tuple[Sequence[float], Sequence[float]]:
        call_count["n"] += 1
        raise AssertionError(f"predict() should not be called again for {path.name}")

    summary = extract_features(
        postgres_engine, DATASET, tmp_path, lambda: _predict_that_must_not_be_called
    )

    assert summary is None
    assert call_count["n"] == 0
    assert get_known_feature_file_names(postgres_engine, DATASET) == set(file_names)


def test_missing_image_raises_before_any_row_is_written(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    file_names = ["present.jpg", "missing.jpg"]
    _write_images(tmp_path, ["present.jpg"])  # "missing.jpg" is never written
    _seed_scene_complexity(postgres_engine, DATASET, file_names)
    predict_calls: dict[str, int] = {"n": 0}
    factory_calls: dict[str, int] = {"n": 0}

    with pytest.raises(FileNotFoundError) as exc_info:
        extract_features(
            postgres_engine,
            DATASET,
            tmp_path,
            _make_predict_factory_spy(_make_stub_predict({}, predict_calls), factory_calls),
        )

    assert "missing.jpg" in str(exc_info.value)
    assert predict_calls["n"] == 0
    # The predictor must never be built when an image is missing (S2).
    assert factory_calls["n"] == 0
    assert get_known_feature_file_names(postgres_engine, DATASET) == set()


def test_nothing_pending_returns_none(postgres_engine: Engine, tmp_path: Path) -> None:
    file_names = ["a.jpg"]
    _write_images(tmp_path, file_names)
    _seed_scene_complexity(postgres_engine, DATASET, file_names)
    extract_features(postgres_engine, DATASET, tmp_path, lambda: _make_stub_predict({}))

    factory_calls: dict[str, int] = {"n": 0}
    summary = extract_features(
        postgres_engine,
        DATASET,
        tmp_path,
        _make_predict_factory_spy(_make_stub_predict({}), factory_calls),
    )

    assert summary is None
    # The predictor must never be built when nothing is pending (S2).
    assert factory_calls["n"] == 0
