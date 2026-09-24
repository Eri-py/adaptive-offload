"""Round-trip test for the shared ORM models against real Postgres.

Runs against a fresh, disposable database per test (see `tests/conftest.py`'s
`postgres_engine` fixture) rather than SQLite — SQLite's dialect differs
enough on native `ENUM`, JSON columns, and FK enforcement to not be
trustworthy here.
"""

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from common.models import (
    Label,
    ModelInference,
    SceneComplexity,
    SimulationResult,
    SimulationRun,
)


def test_round_trips_all_four_tables(postgres_engine: Engine) -> None:
    with Session(postgres_engine) as session:
        run = SimulationRun(
            run_id="run-1",
            dataset="coco_val2017",
            preset_name="baseline",
            frame_count=500,
            condition_vector_count=50,
            condition_ranges={"bandwidth_mbps": [0.5, 100.0], "device_load_pct": [0.0, 100.0]},
            seed=42,
            lambda_value=0.1,
        )
        result = SimulationResult(
            run_id="run-1",
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
        )
        complexity = SceneComplexity(
            dataset="coco_val2017",
            file_name="000000000139.jpg",
            scene_complexity=0.37,
        )
        inference = ModelInference(
            dataset="coco_val2017",
            file_name="000000000139.jpg",
            model_path=Label.LOCAL,
            latency_ms=45.0,
            accuracy=0.88,
        )

        # No ORM `relationship()` links these two mappers, so the unit of work
        # won't infer insert order from the `run_id` FK on its own — flush the
        # parent row first. Postgres enforces this FK; SQLite (the old fixture)
        # silently didn't, which is exactly the dialect gap this switch closes.
        session.add(run)
        session.flush()
        session.add_all([result, complexity, inference])
        session.commit()

    with Session(postgres_engine) as session:
        fetched_run = session.get(SimulationRun, "run-1")
        assert fetched_run is not None
        assert fetched_run.dataset == "coco_val2017"
        assert fetched_run.preset_name == "baseline"
        assert fetched_run.frame_count == 500
        assert fetched_run.condition_vector_count == 50
        assert fetched_run.condition_ranges == {
            "bandwidth_mbps": [0.5, 100.0],
            "device_load_pct": [0.0, 100.0],
        }
        assert fetched_run.seed == 42
        assert fetched_run.lambda_value == 0.1
        assert fetched_run.created_at is not None

        fetched_results = session.query(SimulationResult).all()
        assert len(fetched_results) == 1
        fetched_result = fetched_results[0]
        assert fetched_result.run_id == "run-1"
        assert fetched_result.frame_id == "000000000139.jpg"
        assert fetched_result.network_bandwidth_mbps == 12.5
        assert fetched_result.network_latency_ms == 80.0
        assert fetched_result.network_packet_loss_pct == 1.5
        assert fetched_result.device_load_pct == 40.0
        assert fetched_result.local_latency_ms == 120.0
        assert fetched_result.local_accuracy == 0.82
        assert fetched_result.offload_latency_ms == 200.0
        assert fetched_result.offload_accuracy == 0.91
        assert fetched_result.label == Label.OFFLOAD
        assert fetched_result.created_at is not None

        fetched_complexity = session.get(SceneComplexity, ("coco_val2017", "000000000139.jpg"))
        assert fetched_complexity is not None
        assert fetched_complexity.scene_complexity == 0.37
        assert fetched_complexity.computed_at is not None

        fetched_inference = session.get(
            ModelInference, ("coco_val2017", "000000000139.jpg", Label.LOCAL)
        )
        assert fetched_inference is not None
        assert fetched_inference.latency_ms == 45.0
        assert fetched_inference.accuracy == 0.88
        assert fetched_inference.computed_at is not None
