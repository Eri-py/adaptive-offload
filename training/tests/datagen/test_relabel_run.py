"""Tests for the standalone re-labeling CLI.

Runs against a fresh, disposable database per test (`postgres_engine`
fixture in `training/tests/conftest.py`), per this feature's Postgres-only
testing convention (see `learnings.md`'s Task 1/2 notes) — this CLI's core
function is a thin composition of a real `persistence` read plus
`labeling.compute_label`, so it needs real Postgres to seed a run from.
"""

from __future__ import annotations

import pytest
from common.models import Label, SimulationResult, SimulationRun
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from datagen import labeling
from datagen.persistence import ResultRow, RunConfig, create_run, get_run_results, store_results
from datagen.relabel_run import relabel_run

DATASET = "coco_val2017"
STORED_LAMBDA = 0.1
NEW_LAMBDA = 1.0

# Chosen so that, going from STORED_LAMBDA to NEW_LAMBDA, "flip_row" flips
# from OFFLOAD to LOCAL (offload's higher latency gets penalized harder as
# lambda grows) while "steady_row" stays LOCAL at both lambdas.
_ROW_SPECS = {
    "flip_row.jpg": {
        "local_latency_ms": 100.0,
        "local_accuracy": 0.60,
        "offload_latency_ms": 800.0,
        "offload_accuracy": 0.70,
    },
    "steady_row.jpg": {
        "local_latency_ms": 50.0,
        "local_accuracy": 0.90,
        "offload_latency_ms": 2000.0,
        "offload_accuracy": 0.50,
    },
}


def _seed_run(engine: Engine) -> str:
    config = RunConfig(
        dataset=DATASET,
        preset_name="baseline",
        frame_count=2,
        condition_vector_count=1,
        condition_ranges={"bandwidth_mbps": [0.5, 100.0]},
        seed=42,
        lambda_value=STORED_LAMBDA,
    )
    run_id = create_run(engine, config)

    rows = [
        ResultRow(
            frame_id=frame_id,
            network_bandwidth_mbps=10.0,
            network_latency_ms=20.0,
            network_packet_loss_pct=0.0,
            device_load_pct=10.0,
            local_latency_ms=spec["local_latency_ms"],
            local_accuracy=spec["local_accuracy"],
            offload_latency_ms=spec["offload_latency_ms"],
            offload_accuracy=spec["offload_accuracy"],
            label=labeling.compute_label(
                spec["local_latency_ms"],
                spec["local_accuracy"],
                spec["offload_latency_ms"],
                spec["offload_accuracy"],
                STORED_LAMBDA,
            ),
        )
        for frame_id, spec in _ROW_SPECS.items()
    ]
    store_results(engine, run_id, rows)
    return run_id


def test_relabel_run_reports_flips_correctly(postgres_engine: Engine) -> None:
    run_id = _seed_run(postgres_engine)

    # Confirm the fixture actually sets up the flip/no-flip split it claims to.
    stored = {row.frame_id: row.label for row in get_run_results(postgres_engine, run_id)}
    assert stored["flip_row.jpg"] == Label.OFFLOAD
    assert stored["steady_row.jpg"] == Label.LOCAL

    relabeled = relabel_run(postgres_engine, run_id, NEW_LAMBDA)

    by_frame_id = {
        frame_id: (stored_label, recomputed) for frame_id, stored_label, recomputed in relabeled
    }
    assert by_frame_id["flip_row.jpg"] == (Label.OFFLOAD, Label.LOCAL)
    assert by_frame_id["steady_row.jpg"] == (Label.LOCAL, Label.LOCAL)

    flip_count = sum(1 for _, stored_label, recomputed in relabeled if stored_label != recomputed)
    assert flip_count == 1


def test_relabel_run_persists_nothing(postgres_engine: Engine) -> None:
    run_id = _seed_run(postgres_engine)

    before = get_run_results(postgres_engine, run_id)
    with Session(postgres_engine) as session:
        run_count_before = len(session.execute(select(SimulationRun)).scalars().all())

    relabel_run(postgres_engine, run_id, NEW_LAMBDA)

    after = get_run_results(postgres_engine, run_id)
    with Session(postgres_engine) as session:
        run_count_after = len(session.execute(select(SimulationRun)).scalars().all())
        result_count_after = len(
            session.execute(
                select(SimulationResult).where(SimulationResult.run_id == run_id)
            )
            .scalars()
            .all()
        )

    # No new run row, and the existing result rows (including their stored
    # labels) are completely unchanged.
    assert run_count_after == run_count_before
    assert result_count_after == len(before)
    assert after == before


def test_relabel_run_raises_clear_error_for_run_with_no_results(postgres_engine: Engine) -> None:
    with pytest.raises(ValueError, match="nonexistent-run-id"):
        relabel_run(postgres_engine, "nonexistent-run-id", NEW_LAMBDA)
