"""Round-trip test for the three shared ORM models against an in-memory SQLite engine.

SQLite stands in for Postgres here per `.claude/coding-guidelines.md`/`CLAUDE.md`
so this test never needs a live database connection.
"""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from common.models import Base, Label, SceneComplexity, SimulationResult, SimulationRun


@pytest.fixture
def engine() -> Iterator[Engine]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def test_round_trips_all_three_tables(engine: Engine) -> None:
    with Session(engine) as session:
        run = SimulationRun(
            run_id="run-1",
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

        session.add_all([run, result, complexity])
        session.commit()

    with Session(engine) as session:
        fetched_run = session.get(SimulationRun, "run-1")
        assert fetched_run is not None
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
