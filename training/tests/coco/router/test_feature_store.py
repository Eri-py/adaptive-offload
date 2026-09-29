"""Round-trip tests for `coco.router.feature_store` against real Postgres.

Runs against a fresh, disposable database per test (`postgres_engine` fixture
in `training/tests/conftest.py`) rather than SQLite, per the same dialect-gap
reasoning as `training/tests/coco/datagen/test_persistence.py`.
"""

from common.models import FrameFeatures, SceneComplexity
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from coco.router.feature_store import (
    FeatureRow,
    get_known_feature_file_names,
    list_dataset_frames,
    store_features,
)
from coco.router.frame_features.confidence import ConfidenceFeatures
from coco.router.frame_features.image_stats import ImageStats

DATASET = "coco_val2017"


def _make_row(file_name: str) -> FeatureRow:
    return FeatureRow(
        file_name=file_name,
        image_stats=ImageStats(
            sharpness=120.5,
            brightness=0.42,
            contrast=0.18,
            colorfulness=33.7,
            entropy=6.9,
        ),
        confidence=ConfidenceFeatures(
            detection_count=3,
            max_confidence=0.95,
            mean_confidence=0.71,
            min_confidence=0.40,
            mean_box_area=0.12,
            min_box_area=0.02,
        ),
    )


def _store_scene_complexity(engine: Engine, dataset: str, file_names: list[str]) -> None:
    with Session(engine) as session:
        session.add_all(
            SceneComplexity(dataset=dataset, file_name=file_name, scene_complexity=0.5)
            for file_name in file_names
        )
        session.commit()


def test_list_dataset_frames_orders_by_file_name(postgres_engine: Engine) -> None:
    _store_scene_complexity(
        postgres_engine, DATASET, ["000000000285.jpg", "000000000139.jpg", "000000000632.jpg"]
    )

    assert list_dataset_frames(postgres_engine, DATASET) == [
        "000000000139.jpg",
        "000000000285.jpg",
        "000000000632.jpg",
    ]


def test_list_dataset_frames_is_scoped_to_dataset(postgres_engine: Engine) -> None:
    _store_scene_complexity(postgres_engine, DATASET, ["000000000139.jpg"])
    _store_scene_complexity(postgres_engine, "some_other_dataset", ["img1.jpg"])

    assert list_dataset_frames(postgres_engine, DATASET) == ["000000000139.jpg"]
    assert list_dataset_frames(postgres_engine, "some_other_dataset") == ["img1.jpg"]


def test_store_and_get_known_feature_file_names_round_trips(postgres_engine: Engine) -> None:
    rows = [_make_row("000000000139.jpg"), _make_row("000000000285.jpg")]

    store_features(postgres_engine, DATASET, rows)

    assert get_known_feature_file_names(postgres_engine, DATASET) == {
        "000000000139.jpg",
        "000000000285.jpg",
    }

    with Session(postgres_engine) as session:
        stored = session.get(FrameFeatures, (DATASET, "000000000139.jpg"))
        assert stored is not None
        assert stored.sharpness == 120.5
        assert stored.brightness == 0.42
        assert stored.contrast == 0.18
        assert stored.colorfulness == 33.7
        assert stored.entropy == 6.9
        assert stored.detection_count == 3
        assert stored.max_confidence == 0.95
        assert stored.mean_confidence == 0.71
        assert stored.min_confidence == 0.40
        assert stored.mean_box_area == 0.12
        assert stored.min_box_area == 0.02


def test_store_features_skips_already_present_file_names(postgres_engine: Engine) -> None:
    store_features(postgres_engine, DATASET, [_make_row("000000000139.jpg")])

    # Same file name, different values: must be skipped, not overwritten or raise.
    stale_row = FeatureRow(
        file_name="000000000139.jpg",
        image_stats=ImageStats(
            sharpness=0.0, brightness=0.0, contrast=0.0, colorfulness=0.0, entropy=0.0
        ),
        confidence=ConfidenceFeatures(
            detection_count=0,
            max_confidence=0.0,
            mean_confidence=0.0,
            min_confidence=0.0,
            mean_box_area=0.0,
            min_box_area=0.0,
        ),
    )
    store_features(postgres_engine, DATASET, [stale_row, _make_row("000000000285.jpg")])

    assert get_known_feature_file_names(postgres_engine, DATASET) == {
        "000000000139.jpg",
        "000000000285.jpg",
    }
    with Session(postgres_engine) as session:
        stored = session.get(FrameFeatures, (DATASET, "000000000139.jpg"))
        assert stored is not None
        assert stored.sharpness == 120.5


def test_get_known_feature_file_names_is_scoped_to_dataset(postgres_engine: Engine) -> None:
    store_features(postgres_engine, DATASET, [_make_row("000000000139.jpg")])
    store_features(postgres_engine, "some_other_dataset", [_make_row("img1.jpg")])

    assert get_known_feature_file_names(postgres_engine, DATASET) == {"000000000139.jpg"}
    assert get_known_feature_file_names(postgres_engine, "some_other_dataset") == {"img1.jpg"}
