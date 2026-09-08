"""End-to-end integration test for `datagen.run_simulation`'s core pipeline.

Runs the full pipeline (Tasks 3-10) against a small fake image pool (20
synthetic images written to `tmp_path`, via an injected `resolve_image`) and
the real ephemeral-Postgres fixture from `training/tests/conftest.py` — no
SQLite, no real COCO network access, per this feature's Postgres-only
testing convention (see `learnings.md`'s Task 1b/6 notes).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from common.models import SceneComplexity, SimulationResult, SimulationRun
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from datagen import config
from datagen import run_simulation as run_simulation_module
from datagen.coco import ImageRecord
from datagen.run_simulation import run_simulation

POOL_SIZE = 20
FRAME_COUNT = 10
CONDITION_VECTOR_COUNT = 5
BUCKET_COUNT = 5
SEED = 123
LAMBDA_VALUE = 0.01
PRESET_NAME = "baseline"


def _write_fake_pool(tmp_path: Path) -> list[ImageRecord]:
    """Write `POOL_SIZE` synthetic images to `tmp_path`, with varying complexity.

    Each image blends a flat background with random noise at an
    index-controlled density (0/19 .. 19/19), so the pool's edge-density
    complexity scores spread out rather than clustering or exactly tying —
    real Canny-based scores from distinct images essentially never tie, and
    this mirrors that on synthetic data too.
    """
    rng = np.random.default_rng(0)
    records = []
    for i in range(POOL_SIZE):
        file_name = f"fake_{i:03d}.jpg"
        noise_fraction = i / (POOL_SIZE - 1)
        background = np.full((64, 64, 3), 128, dtype=np.uint8)
        noise = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        mask = rng.random((64, 64)) < noise_fraction
        image = np.where(mask[..., None], noise, background).astype(np.uint8)
        cv2.imwrite(str(tmp_path / file_name), image)
        records.append(ImageRecord(image_id=i, file_name=file_name))
    return records


def _make_resolve_image(tmp_path: Path, call_count: dict[str, int]) -> Callable[[str], Path]:
    def resolve_image(file_name: str) -> Path:
        call_count["n"] += 1
        return tmp_path / file_name

    return resolve_image


def _fetch_results(engine: Engine, run_id: str) -> list[SimulationResult]:
    with Session(engine) as session:
        return (
            session.query(SimulationResult)
            .filter_by(run_id=run_id)
            .order_by(SimulationResult.id)
            .all()
        )


def test_run_simulation_creates_expected_rows_with_full_linkage(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    image_records = _write_fake_pool(tmp_path)
    call_count = {"n": 0}
    resolve_image = _make_resolve_image(tmp_path, call_count)

    run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=resolve_image,
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    assert run_id
    # Every pool image is new the first time, so every one gets resolved.
    assert call_count["n"] == POOL_SIZE

    with Session(postgres_engine) as session:
        run = session.get(SimulationRun, run_id)
        assert run is not None
        assert run.dataset == config.DATASET_NAME
        assert run.preset_name == PRESET_NAME
        assert run.frame_count == FRAME_COUNT
        assert run.condition_vector_count == CONDITION_VECTOR_COUNT
        assert run.seed == SEED
        assert run.lambda_value == LAMBDA_VALUE
        preset = config.PRESETS[PRESET_NAME]
        expected_ranges: dict[str, list[float]] = {
            "bandwidth_mbps": list(preset["bandwidth_mbps"]),
            "network_latency_ms": list(preset["network_latency_ms"]),
            "packet_loss_pct": list(preset["packet_loss_pct"]),
            "device_load_pct": list(preset["device_load_pct"]),
        }
        assert run.condition_ranges == expected_ranges

        complexity_rows = (
            session.query(SceneComplexity).filter_by(dataset=config.DATASET_NAME).all()
        )
        assert len(complexity_rows) == POOL_SIZE

    results = _fetch_results(postgres_engine, run_id)
    assert len(results) == FRAME_COUNT * CONDITION_VECTOR_COUNT

    pool_file_names = {record.file_name for record in image_records}
    for row in results:
        assert row.run_id == run_id
        assert row.frame_id in pool_file_names
        assert row.network_bandwidth_mbps is not None
        assert row.network_latency_ms is not None
        assert row.network_packet_loss_pct is not None
        assert row.device_load_pct is not None
        assert row.local_latency_ms > 0
        assert row.offload_latency_ms > 0
        assert 0.0 <= row.local_accuracy <= 1.0
        assert 0.0 <= row.offload_accuracy <= 1.0
        assert row.label is not None
        assert row.created_at is not None


def test_run_simulation_is_reproducible_and_reuses_known_complexity(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    image_records = _write_fake_pool(tmp_path)

    first_call_count = {"n": 0}
    first_run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, first_call_count),
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    second_call_count = {"n": 0}
    second_run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, second_call_count),
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    assert first_run_id != second_run_id
    # The second run must not recompute complexity for any already-known
    # image — no image should be resolved at all.
    assert second_call_count["n"] == 0

    with Session(postgres_engine) as session:
        complexity_rows = (
            session.query(SceneComplexity).filter_by(dataset=config.DATASET_NAME).all()
        )
        assert len(complexity_rows) == POOL_SIZE

    first_results = _fetch_results(postgres_engine, first_run_id)
    second_results = _fetch_results(postgres_engine, second_run_id)
    assert len(first_results) == len(second_results) == FRAME_COUNT * CONDITION_VECTOR_COUNT

    for first_row, second_row in zip(first_results, second_results, strict=True):
        assert first_row.frame_id == second_row.frame_id
        assert first_row.network_bandwidth_mbps == second_row.network_bandwidth_mbps
        assert first_row.network_latency_ms == second_row.network_latency_ms
        assert first_row.network_packet_loss_pct == second_row.network_packet_loss_pct
        assert first_row.device_load_pct == second_row.device_load_pct
        assert first_row.local_latency_ms == second_row.local_latency_ms
        assert first_row.local_accuracy == second_row.local_accuracy
        assert first_row.offload_latency_ms == second_row.offload_latency_ms
        assert first_row.offload_accuracy == second_row.offload_accuracy
        assert first_row.label == second_row.label


def test_complexity_scoring_flushes_to_postgres_in_batches(
    postgres_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New complexity scores land in Postgres progressively during the scoring
    loop, not only once after every image in the pool has been scored (review
    finding S2) — a small batch size makes intermediate flushes observable
    within a single test run against the `POOL_SIZE`-image fake pool.
    """
    batch_size = 6
    monkeypatch.setattr(run_simulation_module, "COMPLEXITY_SCORE_FLUSH_BATCH_SIZE", batch_size)

    image_records = _write_fake_pool(tmp_path)
    # `store_complexity_scores` is imported (not defined) in run_simulation.py,
    # so mypy's `no_implicit_reexport` (part of `strict`) treats accessing it
    # as an attribute of that module from here as unexported.
    real_store_complexity_scores = (
        run_simulation_module.store_complexity_scores  # type: ignore[attr-defined]
    )
    call_sizes: list[int] = []
    row_counts_after_call: list[int] = []

    def spy_store_complexity_scores(
        engine: Engine, dataset: str, scores: dict[str, float]
    ) -> None:
        real_store_complexity_scores(engine, dataset, scores)
        call_sizes.append(len(scores))
        with Session(engine) as session:
            row_counts_after_call.append(
                session.query(SceneComplexity).filter_by(dataset=dataset).count()
            )

    monkeypatch.setattr(
        run_simulation_module, "store_complexity_scores", spy_store_complexity_scores
    )

    run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    # POOL_SIZE=20 at batch_size=6 flushes as 6, 6, 6, 2 — more than one call,
    # proving the loop flushes incrementally rather than accumulating
    # everything and calling `store_complexity_scores` exactly once at the end.
    assert call_sizes == [6, 6, 6, 2]
    # Each flush's resulting row count is visible in Postgres immediately
    # (not just after the whole loop finishes), confirming partial progress
    # is durable mid-run.
    assert row_counts_after_call == [6, 12, 18, 20]
