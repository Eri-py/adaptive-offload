"""End-to-end integration test for `datagen.cli.run_simulation`'s core pipeline.

Runs the full pipeline (Tasks 3-10) against a small fake image pool (20
synthetic images written to `tmp_path`, via an injected `resolve_image`) and
the real ephemeral-Postgres fixture from `training/tests/conftest.py` — no
SQLite, no real COCO network access, per this feature's Postgres-only
testing convention (see `learnings.md`'s Task 1b/6 notes).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from common.models import ModelInference, SceneComplexity, SimulationResult, SimulationRun
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from datagen import config, presets
from datagen.cli.run_simulation import _require_images_dir, run_simulation
from datagen.persistence import store_complexity_scores
from datagen.sampling.complexity import scene_complexity
from datagen.simulate.ground_truth import Box
from datagen.simulate.inference import DetectionResult, RunInferenceFn
from datagen.sourcing.image_source import ImageRecord

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


def _seed_complexity_scores(
    engine: Engine, dataset: str, image_records: list[ImageRecord], tmp_path: Path
) -> None:
    """Persist real `scene_complexity` scores for `image_records` under `dataset`.

    `run_simulation()` no longer computes complexity itself -- that's
    `score-complexity`'s job exclusively -- so tests must seed it directly to
    get real sampling instead of an empty pool. Scores each image via the
    real `scene_complexity()` against the synthetic file `_write_fake_pool`
    already wrote to `tmp_path`, mirroring what `score-complexity` would
    persist, rather than maintaining a second, synthetic definition of
    complexity in this test file.
    """
    scores = {
        record.file_name: scene_complexity(tmp_path / record.file_name)
        for record in image_records
    }
    store_complexity_scores(engine, dataset, scores)


def _make_fake_inference_fn(call_count: dict[str, int] | None = None) -> RunInferenceFn:
    """A small, fast, deterministic stand-in for a real YOLO inference closure.

    Returns a fixed `DetectionResult` regardless of the image path or
    ground-truth boxes given -- no real model, no real inference. When
    `call_count` is given, increments `call_count["n"]` on every call, so
    tests can assert on how many times (or how few) the fake was invoked --
    mirrors `_make_resolve_image`'s call-counting shape above.
    """

    def run_inference(image_path: Path, ground_truth_boxes: list[Box]) -> DetectionResult:
        if call_count is not None:
            call_count["n"] += 1
        return DetectionResult(latency_ms=50.0, accuracy=0.8)

    return run_inference


def _make_ground_truth_sensitive_inference_fn(
    call_count: dict[str, int] | None = None,
) -> RunInferenceFn:
    """A fake whose `accuracy` is an exact, invertible function of the
    number of ground-truth boxes it's called with (`min(1.0, n / 3.0)`).

    Unlike `_make_fake_inference_fn` above (fixed output regardless of
    input), this lets a test recover exactly how many ground-truth boxes a
    given call received by reading the persisted accuracy back -- used by
    the S4 ground-truth-plumbing test below, which needs to prove the right
    `image_id`'s boxes (not the wrong key's, not an empty/dropped list)
    reached each frame's inference call.
    """

    def run_inference(image_path: Path, ground_truth_boxes: list[Box]) -> DetectionResult:
        if call_count is not None:
            call_count["n"] += 1
        return DetectionResult(latency_ms=50.0, accuracy=min(1.0, len(ground_truth_boxes) / 3.0))

    return run_inference


def _make_box() -> Box:
    """A single ground-truth box with arbitrary coordinates.

    Only `len(ground_truth_boxes)` matters to
    `_make_ground_truth_sensitive_inference_fn` above -- the coordinates
    and category are irrelevant filler.
    """
    return Box(category_name="fake-category", x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)


def _fetch_results(engine: Engine, run_id: str) -> list[SimulationResult]:
    with Session(engine) as session:
        return (
            session.query(SimulationResult)
            .filter_by(run_id=run_id)
            .order_by(SimulationResult.id)
            .all()
        )


def test_require_images_dir_accepts_existing_directory(tmp_path: Path) -> None:
    _require_images_dir(tmp_path)


def test_require_images_dir_raises_for_nonexistent_path(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError) as exc_info:
        _require_images_dir(missing_dir)

    assert str(missing_dir) in str(exc_info.value)


def test_run_simulation_creates_expected_rows_with_full_linkage(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)
    call_count = {"n": 0}
    resolve_image = _make_resolve_image(tmp_path, call_count)

    run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=resolve_image,
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    assert run_id
    # Complexity is pre-seeded (not computed by `run_simulation()` anymore),
    # so every pool image is only resolved once, by the model-inference loop
    # (`_compute_missing_model_inference`).
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
        preset = presets.PRESETS[PRESET_NAME]
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


def test_run_simulation_stores_rows_under_caller_supplied_dataset(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    """Review finding S1: `--dataset`/`dataset=` must land scores and runs
    under the caller's dataset key, not `config.DATASET_NAME` -- the
    feature's headline behavior, previously uncovered by any test.
    """
    custom_dataset = "test_dataset_xyz"
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, custom_dataset, image_records, tmp_path)

    run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
        dataset=custom_dataset,
    )

    with Session(postgres_engine) as session:
        run = session.get(SimulationRun, run_id)
        assert run is not None
        assert run.dataset == custom_dataset

        assert (
            session.query(SceneComplexity).filter_by(dataset=custom_dataset).count()
            == POOL_SIZE
        )
        assert (
            session.query(SceneComplexity).filter_by(dataset=config.DATASET_NAME).count() == 0
        )


def test_run_simulation_reads_seeded_complexity_consistently_across_calls(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    """`run_simulation()` no longer computes or persists complexity itself --
    that's `score-complexity`'s job exclusively -- so this test's original
    premise (a second call skips recomputing already-known complexity)
    no longer applies; there's nothing left for it to recompute. What's
    still meaningful and worth guarding: two calls against the same
    pre-seeded dataset both read the same `scene_complexity` rows via
    `get_known_complexity` and produce identical sampling/results, and
    neither call writes any additional `scene_complexity` rows.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)

    first_call_count = {"n": 0}
    first_run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, first_call_count),
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
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
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    assert first_run_id != second_run_id
    # The second run must not recompute model inference for any
    # already-known image either -- no image should be resolved at all.
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


def test_run_simulation_excludes_out_of_pool_frames_from_sampling(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    """Review finding S1: `scene_complexity` can hold rows for frames outside
    the current pool (e.g. a wider or different image set scored under the
    same `--dataset` name in an earlier run). Sampling must never draw one of
    those in -- it has no `model_inference` entry, which used to raise a bare
    `KeyError` after the run row was already created. Seeds an extra
    `scene_complexity` row for a file name that is not part of this test's
    fake pool, under the same dataset the run below uses, and asserts the run
    both completes and never persists that file name into
    `simulation_results`.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)
    out_of_pool_file_name = "not_in_pool.jpg"

    with Session(postgres_engine) as session:
        session.add(
            SceneComplexity(
                dataset=config.DATASET_NAME,
                file_name=out_of_pool_file_name,
                # A mid-range value, not an extreme outlier -- so if the
                # filter were missing, this row would be a plausible pick
                # rather than one `stratified_sample` would skip anyway.
                scene_complexity=0.5,
            )
        )
        session.commit()

    run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    results = _fetch_results(postgres_engine, run_id)
    assert len(results) == FRAME_COUNT * CONDITION_VECTOR_COUNT
    sampled_frame_ids = {row.frame_id for row in results}
    assert out_of_pool_file_name not in sampled_frame_ids


def _expected_condition_ranges(preset_name: str) -> dict[str, list[float]]:
    preset = presets.PRESETS[preset_name]
    return {
        "bandwidth_mbps": list(preset["bandwidth_mbps"]),
        "network_latency_ms": list(preset["network_latency_ms"]),
        "packet_loss_pct": list(preset["packet_loss_pct"]),
        "device_load_pct": list(preset["device_load_pct"]),
    }


def test_run_simulation_never_conflates_two_different_preset_runs(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    """Two runs on different presets against the same engine (spec acceptance
    criterion: "rows from the two runs are never conflated") must each carry
    their own `run_id`, their own row count with no cross-run leakage, and
    their own preset's `condition_ranges` on their `SimulationRun` record —
    not the other run's.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)
    baseline_preset = "baseline"
    stress_preset = "network-stress"

    baseline_run_id = run_simulation(
        postgres_engine,
        baseline_preset,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )
    # A different condition_vector_count so the two runs' expected row counts
    # differ too -- a leaked/duplicated row would then also show up as a
    # wrong row count, not just a wrong run_id.
    stress_condition_vector_count = CONDITION_VECTOR_COUNT + 2
    stress_run_id = run_simulation(
        postgres_engine,
        stress_preset,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=stress_condition_vector_count,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    assert baseline_run_id != stress_run_id

    with Session(postgres_engine) as session:
        baseline_run = session.get(SimulationRun, baseline_run_id)
        stress_run = session.get(SimulationRun, stress_run_id)
        assert baseline_run is not None
        assert stress_run is not None
        assert baseline_run.preset_name == baseline_preset
        assert stress_run.preset_name == stress_preset
        # Each run's persisted condition_ranges matches its own preset, not
        # the other one -- the crux of what this test guards against.
        assert baseline_run.condition_ranges == _expected_condition_ranges(baseline_preset)
        assert stress_run.condition_ranges == _expected_condition_ranges(stress_preset)
        assert baseline_run.condition_ranges != stress_run.condition_ranges

    baseline_results = _fetch_results(postgres_engine, baseline_run_id)
    stress_results = _fetch_results(postgres_engine, stress_run_id)

    expected_baseline_count = FRAME_COUNT * CONDITION_VECTOR_COUNT
    expected_stress_count = FRAME_COUNT * stress_condition_vector_count
    # No leakage in either direction: each run's row count matches exactly
    # what its own configuration should produce, no more and no less.
    assert len(baseline_results) == expected_baseline_count
    assert len(stress_results) == expected_stress_count

    # No row from either run carries the other run's run_id, and no row is
    # double-counted across both queries.
    baseline_ids = {row.id for row in baseline_results}
    stress_ids = {row.id for row in stress_results}
    assert baseline_ids.isdisjoint(stress_ids)
    for row in baseline_results:
        assert row.run_id == baseline_run_id
    for row in stress_results:
        assert row.run_id == stress_run_id

    with Session(postgres_engine) as session:
        total_results = session.query(SimulationResult).count()
    assert total_results == expected_baseline_count + expected_stress_count


def test_run_simulation_warns_when_sample_comes_back_short(
    postgres_engine: Engine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When the pool is too small (relative to frame_count/bucket_count) for
    `stratified_sample` to fill every bucket's share, it silently returns
    fewer frames than requested (documented at `sampling.py:39-42`) -- review
    finding N1: `run_simulation` must log a warning noting both numbers
    rather than quietly persisting the short count as if it were correct.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)
    # POOL_SIZE=20 split into BUCKET_COUNT=5 buckets gives 4 items per
    # bucket; requesting far more than 5x that per bucket (i.e. more than
    # POOL_SIZE total) guarantees every bucket comes up short.
    requested_frame_count = POOL_SIZE * 5

    with caplog.at_level(logging.WARNING, logger="datagen.cli.run_simulation"):
        run_simulation(
            postgres_engine,
            PRESET_NAME,
            image_records=image_records,
            resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
            run_local_inference=_make_fake_inference_fn(),
            run_offload_inference=_make_fake_inference_fn(),
            ground_truth={},
            frame_count=requested_frame_count,
            condition_vector_count=CONDITION_VECTOR_COUNT,
            bucket_count=BUCKET_COUNT,
            seed=SEED,
            lambda_value=LAMBDA_VALUE,
        )

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(POOL_SIZE) in message
    assert str(requested_frame_count) in message


def test_run_simulation_does_not_warn_when_sample_meets_target(
    postgres_engine: Engine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The normal/happy-path case (pool large enough to fill every bucket's
    share, per this module's own default test config) must not log the
    short-sample warning -- the negative counterpart to the test above.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)

    with caplog.at_level(logging.WARNING, logger="datagen.cli.run_simulation"):
        run_simulation(
            postgres_engine,
            PRESET_NAME,
            image_records=image_records,
            resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
            run_local_inference=_make_fake_inference_fn(),
            run_offload_inference=_make_fake_inference_fn(),
            ground_truth={},
            frame_count=FRAME_COUNT,
            condition_vector_count=CONDITION_VECTOR_COUNT,
            bucket_count=BUCKET_COUNT,
            seed=SEED,
            lambda_value=LAMBDA_VALUE,
        )

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == []


def test_run_simulation_excludes_unscored_pool_images_and_warns(
    postgres_engine: Engine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`run_simulation()` no longer auto-scores unscored pool images -- it
    excludes them from sampling and logs a loud (not hard-failing) warning
    instead. Seeds `scene_complexity` for only half the pool, confirms the
    run still completes, confirms no unscored image ever appears as a
    `frame_id` in the persisted results, and confirms the warning names the
    correct missing/total counts and dataset.
    """
    image_records = _write_fake_pool(tmp_path)
    scored_records = image_records[: POOL_SIZE // 2]
    unscored_records = image_records[POOL_SIZE // 2 :]
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, scored_records, tmp_path)

    with caplog.at_level(logging.WARNING, logger="datagen.cli.run_simulation"):
        run_id = run_simulation(
            postgres_engine,
            PRESET_NAME,
            image_records=image_records,
            resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
            run_local_inference=_make_fake_inference_fn(),
            run_offload_inference=_make_fake_inference_fn(),
            ground_truth={},
            frame_count=FRAME_COUNT,
            condition_vector_count=CONDITION_VECTOR_COUNT,
            bucket_count=BUCKET_COUNT,
            seed=SEED,
            lambda_value=LAMBDA_VALUE,
        )

    unscored_file_names = {record.file_name for record in unscored_records}
    results = _fetch_results(postgres_engine, run_id)
    sampled_frame_ids = {row.frame_id for row in results}
    assert sampled_frame_ids, "expected the scored half of the pool to still yield frames"
    assert sampled_frame_ids.isdisjoint(unscored_file_names)

    coverage_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "have no scene_complexity score" in record.getMessage()
    ]
    assert len(coverage_warnings) == 1
    message = coverage_warnings[0].getMessage()
    assert str(len(unscored_records)) in message
    assert str(POOL_SIZE) in message
    assert config.DATASET_NAME in message
    assert "score-complexity" in message


def test_run_simulation_does_not_warn_about_pool_coverage_when_fully_scored(
    postgres_engine: Engine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Negative counterpart: when every pool image already has a
    `scene_complexity` score, no pool-coverage warning is logged.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)

    with caplog.at_level(logging.WARNING, logger="datagen.cli.run_simulation"):
        run_simulation(
            postgres_engine,
            PRESET_NAME,
            image_records=image_records,
            resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
            run_local_inference=_make_fake_inference_fn(),
            run_offload_inference=_make_fake_inference_fn(),
            ground_truth={},
            frame_count=FRAME_COUNT,
            condition_vector_count=CONDITION_VECTOR_COUNT,
            bucket_count=BUCKET_COUNT,
            seed=SEED,
            lambda_value=LAMBDA_VALUE,
        )

    coverage_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "have no scene_complexity score" in record.getMessage()
    ]
    assert coverage_warnings == []


def test_condition_never_changes_accuracy_for_a_given_frame(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    """Spec acceptance criterion: for a given frame, every `simulation_results`
    row across all sampled conditions carries the same `local_accuracy`/
    `offload_accuracy` -- accuracy comes from real IoU-based scoring on the
    frame alone, and `apply_condition_overhead` never touches it, only
    latency. Latency is allowed (expected, given the fixed-`DetectionResult`
    fake plus `apply_condition_overhead`'s condition-driven overhead) to
    differ across conditions.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)

    run_id = run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_fake_inference_fn(),
        run_offload_inference=_make_fake_inference_fn(),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    results = _fetch_results(postgres_engine, run_id)
    results_by_frame: dict[str, list[SimulationResult]] = {}
    for row in results:
        results_by_frame.setdefault(row.frame_id, []).append(row)

    assert len(results_by_frame) == FRAME_COUNT
    for frame_id, frame_rows in results_by_frame.items():
        assert len(frame_rows) == CONDITION_VECTOR_COUNT
        local_accuracies = {row.local_accuracy for row in frame_rows}
        offload_accuracies = {row.offload_accuracy for row in frame_rows}
        assert len(local_accuracies) == 1, (
            f"frame {frame_id!r} has varying local_accuracy across conditions: "
            f"{[row.local_accuracy for row in frame_rows]}"
        )
        assert len(offload_accuracies) == 1, (
            f"frame {frame_id!r} has varying offload_accuracy across conditions: "
            f"{[row.offload_accuracy for row in frame_rows]}"
        )
        # Latency, unlike accuracy, is allowed to vary across conditions --
        # asserting it actually does (rather than merely allowing it) guards
        # against a degenerate fake that would make this test vacuous.
        local_latencies = {row.local_latency_ms for row in frame_rows}
        offload_latencies = {row.offload_latency_ms for row in frame_rows}
        assert len(local_latencies) > 1 or len(offload_latencies) > 1


def test_run_simulation_reuses_known_model_inference_on_second_call(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    """Mirrors `test_run_simulation_is_reproducible_and_reuses_known_complexity`'s
    pattern for `model_inference`: a second `run_simulation()` call against
    the same dataset/pool must not re-run real inference for any frame
    already covered by both `Label.LOCAL` and `Label.OFFLOAD` in Postgres.
    """
    image_records = _write_fake_pool(tmp_path)
    _seed_complexity_scores(postgres_engine, config.DATASET_NAME, image_records, tmp_path)

    first_local_call_count = {"n": 0}
    first_offload_call_count = {"n": 0}
    run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_fake_inference_fn(first_local_call_count),
        run_offload_inference=_make_fake_inference_fn(first_offload_call_count),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )
    # Every pool image is new the first time, so both fakes are called once
    # per image in the pool.
    assert first_local_call_count["n"] == POOL_SIZE
    assert first_offload_call_count["n"] == POOL_SIZE

    second_local_call_count = {"n": 0}
    second_offload_call_count = {"n": 0}
    run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_fake_inference_fn(second_local_call_count),
        run_offload_inference=_make_fake_inference_fn(second_offload_call_count),
        ground_truth={},
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )
    # The second run must not recompute inference for any already-known
    # image -- neither fake should be called at all.
    assert second_local_call_count["n"] == 0
    assert second_offload_call_count["n"] == 0

    with Session(postgres_engine) as session:
        inference_rows = (
            session.query(ModelInference).filter_by(dataset=config.DATASET_NAME).all()
        )
    # Two rows (LOCAL, OFFLOAD) per pool image, not doubled by the second run.
    assert len(inference_rows) == POOL_SIZE * 2


def test_run_simulation_passes_correct_ground_truth_to_inference_functions(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    """Review finding S4: `_compute_missing_model_inference` must look up
    each frame's ground truth by `record.image_id`
    (`ground_truth.get(record.image_id, [])`) and pass exactly those boxes
    through to `run_local_inference`/`run_offload_inference` -- not
    `record.file_name`, and not an empty/dropped list regardless of what was
    actually given.

    Gives three distinct pool images three distinct ground-truth box counts
    (3, 0, 1 -- the zero-box one deliberately has no `ground_truth` entry at
    all, exercising the `.get(..., [])` default path) and a fake inference
    function whose returned `accuracy` is an exact, invertible function of
    `len(ground_truth_boxes)`. Then checks each image's persisted
    `model_inference.accuracy` decodes back to exactly the box count that
    image (and only that image) was given. If the lookup used the wrong key
    (e.g. `file_name`) or the boxes were silently dropped before reaching
    the inference functions, every image would instead show the zero-box
    accuracy (0.0) -- this test fails loudly in either case, unlike every
    other test in this module, which passes `ground_truth={}` and so could
    never catch this bug class.
    """
    image_records = _write_fake_pool(tmp_path)

    three_box_image_id = 0
    zero_box_image_id = 1
    one_box_image_id = 2
    ground_truth: dict[int, list[Box]] = {
        three_box_image_id: [_make_box(), _make_box(), _make_box()],
        one_box_image_id: [_make_box()],
        # zero_box_image_id has no entry -- must default to zero boxes via
        # `.get(record.image_id, [])`, not raise or reuse another image's.
    }

    run_simulation(
        postgres_engine,
        PRESET_NAME,
        image_records=image_records,
        resolve_image=_make_resolve_image(tmp_path, {"n": 0}),
        run_local_inference=_make_ground_truth_sensitive_inference_fn(),
        run_offload_inference=_make_ground_truth_sensitive_inference_fn(),
        ground_truth=ground_truth,
        frame_count=FRAME_COUNT,
        condition_vector_count=CONDITION_VECTOR_COUNT,
        bucket_count=BUCKET_COUNT,
        seed=SEED,
        lambda_value=LAMBDA_VALUE,
    )

    file_name_by_image_id = {record.image_id: record.file_name for record in image_records}
    with Session(postgres_engine) as session:
        inference_rows = (
            session.query(ModelInference).filter_by(dataset=config.DATASET_NAME).all()
        )
    accuracy_by_file_name = {row.file_name: row.accuracy for row in inference_rows}

    for image_id, expected_box_count in (
        (three_box_image_id, 3),
        (zero_box_image_id, 0),
        (one_box_image_id, 1),
    ):
        file_name = file_name_by_image_id[image_id]
        expected_accuracy = min(1.0, expected_box_count / 3.0)
        assert accuracy_by_file_name[file_name] == pytest.approx(expected_accuracy), (
            f"image_id={image_id} (file_name={file_name!r}) expected accuracy "
            f"{expected_accuracy} from {expected_box_count} ground-truth boxes, "
            f"got {accuracy_by_file_name[file_name]} -- the wrong ground-truth "
            "boxes (or none) reached this frame's inference call."
        )
