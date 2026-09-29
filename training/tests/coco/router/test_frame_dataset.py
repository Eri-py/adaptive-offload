"""Tests for `coco.router.frame_dataset` against real Postgres.

Runs against a fresh, disposable database per test (`postgres_engine`
fixture in `training/tests/conftest.py`), per the same dialect-gap reasoning
as `training/tests/coco/datagen/test_persistence.py`.
"""

from common.models import Label
from sqlalchemy import Engine

from coco.datagen.persistence import (
    ResultRow,
    RunConfig,
    create_run,
    store_complexity_scores,
    store_model_inference,
    store_results,
)
from coco.datagen.simulate.inference import DetectionResult
from coco.router.feature_store import FeatureRow, store_features
from coco.router.frame_dataset import (
    load_frame_table,
    load_simulated_frame_ids,
    load_simulated_rows,
)
from coco.router.frame_features.confidence import ConfidenceFeatures
from coco.router.frame_features.image_stats import ImageStats

DATASET = "coco_val2017"


def _feature_row(file_name: str, *, sharpness: float = 100.0) -> FeatureRow:
    return FeatureRow(
        file_name=file_name,
        image_stats=ImageStats(
            sharpness=sharpness, brightness=0.4, contrast=0.2, colorfulness=30.0, entropy=6.5
        ),
        confidence=ConfidenceFeatures(
            detection_count=3,
            max_confidence=0.9,
            mean_confidence=0.7,
            min_confidence=0.4,
            mean_box_area=0.1,
            min_box_area=0.02,
        ),
    )


def _seed_frame(
    engine: Engine,
    dataset: str,
    file_name: str,
    *,
    scene_complexity: float = 0.5,
    local_accuracy: float = 0.8,
    offload_accuracy: float = 0.9,
    with_features: bool = True,
) -> None:
    """Seeds `scene_complexity`, both `model_inference` rows, and (unless
    `with_features` is False) a `frame_features` row for one frame.
    """
    store_complexity_scores(engine, dataset, {file_name: scene_complexity})
    store_model_inference(
        engine,
        dataset,
        {
            file_name: {
                Label.LOCAL: DetectionResult(latency_ms=50.0, accuracy=local_accuracy),
                Label.OFFLOAD: DetectionResult(latency_ms=150.0, accuracy=offload_accuracy),
            }
        },
    )
    if with_features:
        store_features(engine, dataset, [_feature_row(file_name)])


def _seed_simulation_row(engine: Engine, dataset: str, frame_id: str) -> None:
    run_id = create_run(
        engine,
        RunConfig(
            dataset=dataset,
            preset_name="baseline",
            frame_count=1,
            condition_vector_count=1,
            condition_ranges={"bandwidth_mbps": [0.5, 100.0]},
            seed=42,
            lambda_value=0.005,
        ),
    )
    store_results(
        engine,
        run_id,
        [
            ResultRow(
                frame_id=frame_id,
                network_bandwidth_mbps=12.5,
                network_latency_ms=80.0,
                network_packet_loss_pct=1.5,
                device_load_pct=40.0,
                local_latency_ms=120.0,
                local_accuracy=0.82,
                offload_latency_ms=200.0,
                offload_accuracy=0.91,
                label=Label.OFFLOAD,
            )
        ],
    )


def test_load_frame_table_joins_and_computes_gap(postgres_engine: Engine) -> None:
    _seed_frame(
        postgres_engine,
        DATASET,
        "000000000139.jpg",
        scene_complexity=0.37,
        local_accuracy=0.75,
        offload_accuracy=0.91,
    )

    df = load_frame_table(postgres_engine, DATASET)

    assert list(df["file_name"]) == ["000000000139.jpg"]
    row = df.iloc[0]
    assert row["scene_complexity"] == 0.37
    assert row["sharpness"] == 100.0
    assert row["detection_count"] == 3
    assert row["mean_box_area"] == 0.1
    assert row["local_accuracy"] == 0.75
    assert row["offload_accuracy"] == 0.91
    assert row["gap"] == 0.91 - 0.75


def test_load_frame_table_excludes_frame_missing_frame_features(postgres_engine: Engine) -> None:
    _seed_frame(postgres_engine, DATASET, "000000000139.jpg", with_features=True)
    _seed_frame(postgres_engine, DATASET, "000000000285.jpg", with_features=False)

    df = load_frame_table(postgres_engine, DATASET)

    assert list(df["file_name"]) == ["000000000139.jpg"]


def test_load_frame_table_is_sorted_and_scoped_to_dataset(postgres_engine: Engine) -> None:
    _seed_frame(postgres_engine, DATASET, "000000000285.jpg")
    _seed_frame(postgres_engine, DATASET, "000000000139.jpg")
    _seed_frame(postgres_engine, "some_other_dataset", "img1.jpg")

    df = load_frame_table(postgres_engine, DATASET)

    assert list(df["file_name"]) == ["000000000139.jpg", "000000000285.jpg"]


def test_load_simulated_frame_ids(postgres_engine: Engine) -> None:
    _seed_frame(postgres_engine, DATASET, "000000000139.jpg")
    _seed_frame(postgres_engine, DATASET, "000000000285.jpg")
    _seed_simulation_row(postgres_engine, DATASET, "000000000139.jpg")

    assert load_simulated_frame_ids(postgres_engine, DATASET) == {"000000000139.jpg"}


def test_load_simulated_frame_ids_is_scoped_to_dataset(postgres_engine: Engine) -> None:
    _seed_frame(postgres_engine, DATASET, "000000000139.jpg")
    _seed_frame(postgres_engine, "some_other_dataset", "000000000139.jpg")
    _seed_simulation_row(postgres_engine, DATASET, "000000000139.jpg")

    assert load_simulated_frame_ids(postgres_engine, "some_other_dataset") == set()


def test_load_simulated_rows_joins_frame_features_without_duplicating_scene_complexity(
    postgres_engine: Engine,
) -> None:
    _seed_frame(postgres_engine, DATASET, "000000000139.jpg", scene_complexity=0.6)
    _seed_simulation_row(postgres_engine, DATASET, "000000000139.jpg")

    df = load_simulated_rows(postgres_engine, DATASET)

    assert list(df["frame_id"]) == ["000000000139.jpg"]
    assert df.iloc[0]["scene_complexity"] == 0.6
    assert df.iloc[0]["sharpness"] == 100.0
    assert df.iloc[0]["mean_confidence"] == 0.7
    assert list(df.columns).count("scene_complexity") == 1
