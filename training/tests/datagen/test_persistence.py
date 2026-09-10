"""Round-trip tests for `datagen.persistence` against real Postgres.

Runs against a fresh, disposable database per test (`postgres_engine` fixture
in `training/tests/conftest.py`) rather than SQLite, per the same dialect-gap
reasoning as `database/tests/common/test_models.py`.
"""

from common.models import Label, SimulationResult, SimulationRun
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from datagen.persistence import (
    ResultRow,
    RunConfig,
    create_run,
    get_known_complexity,
    store_complexity_scores,
    store_results,
)

DATASET = "coco_val2017"


def test_store_and_get_known_complexity_round_trips(postgres_engine: Engine) -> None:
    scores = {"000000000139.jpg": 0.37, "000000000285.jpg": 0.61}
    store_complexity_scores(postgres_engine, DATASET, scores)

    fetched = get_known_complexity(postgres_engine, DATASET)

    assert fetched == scores


def test_store_complexity_scores_skips_already_present_file_names(
    postgres_engine: Engine,
) -> None:
    store_complexity_scores(postgres_engine, DATASET, {"000000000139.jpg": 0.37})

    # Re-storing the same file name with a different score must not overwrite
    # it (and must not raise on the primary-key collision) — only genuinely
    # new file names get inserted.
    store_complexity_scores(
        postgres_engine,
        DATASET,
        {"000000000139.jpg": 0.99, "000000000285.jpg": 0.61},
    )

    fetched = get_known_complexity(postgres_engine, DATASET)

    assert fetched == {"000000000139.jpg": 0.37, "000000000285.jpg": 0.61}


def test_get_known_complexity_is_scoped_to_dataset(postgres_engine: Engine) -> None:
    store_complexity_scores(postgres_engine, DATASET, {"000000000139.jpg": 0.37})
    store_complexity_scores(postgres_engine, "some_other_dataset", {"img1.jpg": 0.5})

    assert get_known_complexity(postgres_engine, DATASET) == {"000000000139.jpg": 0.37}
    assert get_known_complexity(postgres_engine, "some_other_dataset") == {"img1.jpg": 0.5}


def test_create_run_and_store_results_round_trips_with_fk_linkage(
    postgres_engine: Engine,
) -> None:
    config = RunConfig(
        dataset=DATASET,
        preset_name="baseline",
        frame_count=500,
        condition_vector_count=50,
        condition_ranges={"bandwidth_mbps": [0.5, 100.0], "device_load_pct": [0.0, 100.0]},
        seed=42,
        lambda_value=0.005,
    )

    run_id = create_run(postgres_engine, config)

    assert run_id

    rows = [
        ResultRow(
            frame_id="000000000139.jpg",
            network_bandwidth_mbps=12.5,
            network_latency_ms=80.0,
            network_packet_loss_pct=1.5,
            device_load_pct=40.0,
            local_latency_ms=120.0,
            local_accuracy=0.82,
            offload_latency_ms=200.0,
            offload_accuracy=0.91,
            label=Label.OFFLOAD,
        ),
        ResultRow(
            frame_id="000000000285.jpg",
            network_bandwidth_mbps=3.0,
            network_latency_ms=250.0,
            network_packet_loss_pct=8.0,
            device_load_pct=10.0,
            local_latency_ms=55.0,
            local_accuracy=0.79,
            offload_latency_ms=400.0,
            offload_accuracy=0.60,
            label=Label.LOCAL,
        ),
    ]

    store_results(postgres_engine, run_id, rows)

    with Session(postgres_engine) as session:
        fetched_run = session.get(SimulationRun, run_id)
        assert fetched_run is not None
        assert fetched_run.dataset == DATASET
        assert fetched_run.preset_name == "baseline"
        assert fetched_run.frame_count == 500
        assert fetched_run.condition_vector_count == 50
        assert fetched_run.condition_ranges == config.condition_ranges
        assert fetched_run.seed == 42
        assert fetched_run.lambda_value == 0.005

        fetched_results = (
            session.query(SimulationResult)
            .filter_by(run_id=run_id)
            .order_by(SimulationResult.frame_id)
            .all()
        )
        assert len(fetched_results) == 2

        first, second = fetched_results
        assert first.run_id == run_id
        assert first.frame_id == "000000000139.jpg"
        assert first.network_bandwidth_mbps == 12.5
        assert first.network_latency_ms == 80.0
        assert first.network_packet_loss_pct == 1.5
        assert first.device_load_pct == 40.0
        assert first.local_latency_ms == 120.0
        assert first.local_accuracy == 0.82
        assert first.offload_latency_ms == 200.0
        assert first.offload_accuracy == 0.91
        assert first.label == Label.OFFLOAD

        assert second.run_id == run_id
        assert second.frame_id == "000000000285.jpg"
        assert second.label == Label.LOCAL
