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
    get_known_model_inference,
    get_run_results,
    store_complexity_scores,
    store_model_inference,
    store_results,
)
from datagen.simulate.inference import DetectionResult

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


def test_store_and_get_known_model_inference_round_trips(postgres_engine: Engine) -> None:
    results = {
        "000000000139.jpg": {
            Label.LOCAL: DetectionResult(latency_ms=42.0, accuracy=0.75),
            Label.OFFLOAD: DetectionResult(latency_ms=120.0, accuracy=0.91),
        }
    }
    store_model_inference(postgres_engine, DATASET, results)

    fetched = get_known_model_inference(postgres_engine, DATASET)

    assert fetched == results


def test_store_model_inference_skips_already_known_pairs(postgres_engine: Engine) -> None:
    store_model_inference(
        postgres_engine,
        DATASET,
        {"000000000139.jpg": {Label.LOCAL: DetectionResult(latency_ms=42.0, accuracy=0.75)}},
    )

    # Re-storing the same (file_name, model_path) pair with different values
    # must not overwrite it (and must not raise on the primary-key
    # collision); a genuinely new pair on the same file still gets inserted.
    store_model_inference(
        postgres_engine,
        DATASET,
        {
            "000000000139.jpg": {
                Label.LOCAL: DetectionResult(latency_ms=999.0, accuracy=0.01),
                Label.OFFLOAD: DetectionResult(latency_ms=120.0, accuracy=0.91),
            }
        },
    )

    fetched = get_known_model_inference(postgres_engine, DATASET)

    assert fetched == {
        "000000000139.jpg": {
            Label.LOCAL: DetectionResult(latency_ms=42.0, accuracy=0.75),
            Label.OFFLOAD: DetectionResult(latency_ms=120.0, accuracy=0.91),
        }
    }


def test_get_known_model_inference_is_empty_for_unknown_dataset(postgres_engine: Engine) -> None:
    assert get_known_model_inference(postgres_engine, "no_such_dataset") == {}


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


def test_get_run_results_round_trips_all_fields(postgres_engine: Engine) -> None:
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

    rows = [
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
    ]
    store_results(postgres_engine, run_id, rows)

    fetched = get_run_results(postgres_engine, run_id)

    # get_run_results orders by frame_id, so the fetched order differs from
    # the insertion order above — assert against a frame_id-sorted expectation.
    assert fetched == sorted(rows, key=lambda row: row.frame_id)


def test_get_run_results_orders_stably_within_a_shared_frame_id(postgres_engine: Engine) -> None:
    # A real run has condition_vector_count rows per frame_id, all sharing
    # the same frame_id and thus indistinguishable to a plain
    # `.order_by(frame_id)` — Postgres is free to return those in any order
    # between calls. Store several rows under one frame_id in a known order
    # and confirm get_run_results returns them in that same order, and does
    # so identically across repeated calls.
    config = RunConfig(
        dataset=DATASET,
        preset_name="baseline",
        frame_count=1,
        condition_vector_count=5,
        condition_ranges={"bandwidth_mbps": [0.5, 100.0]},
        seed=42,
        lambda_value=0.005,
    )
    run_id = create_run(postgres_engine, config)

    rows = [
        ResultRow(
            frame_id="000000000139.jpg",
            network_bandwidth_mbps=float(i),
            network_latency_ms=80.0,
            network_packet_loss_pct=1.5,
            device_load_pct=40.0,
            local_latency_ms=120.0,
            local_accuracy=0.82,
            offload_latency_ms=200.0,
            offload_accuracy=0.91,
            label=Label.OFFLOAD,
        )
        for i in range(5)
    ]
    store_results(postgres_engine, run_id, rows)

    first_call = get_run_results(postgres_engine, run_id)
    second_call = get_run_results(postgres_engine, run_id)

    # Insertion order is preserved (id ascending), and repeated calls agree.
    assert first_call == rows
    assert second_call == rows


def test_get_run_results_is_empty_for_run_with_no_results(postgres_engine: Engine) -> None:
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

    assert get_run_results(postgres_engine, run_id) == []


def test_get_run_results_is_empty_for_nonexistent_run(postgres_engine: Engine) -> None:
    assert get_run_results(postgres_engine, "00000000-0000-0000-0000-000000000000") == []
