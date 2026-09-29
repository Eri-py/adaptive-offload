"""End-to-end test for `coco.router.feature_experiment.run_experiment`, against a
small synthetic dataset seeded in ephemeral Postgres (per Task 1's tests'
approach, since this integration exercises the real SQL joins).

Follows `test_two_stage.py`'s fixture strategy: local/offload accuracy
alternate which path is better by frame index (even -> offload better, odd
-> local better), so both stage-1's `gap <= 0` classifier and stage-2's
ACCEPT/ESCALATE and LOCAL/OFFLOAD labels see both classes, in any 80/20
photo split, rather than only the single-class fallback.
"""

from __future__ import annotations

import math

import numpy as np
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
from coco.router.feature_experiment import run_experiment
from coco.router.feature_store import FeatureRow, store_features
from coco.router.frame_features.confidence import ConfidenceFeatures
from coco.router.frame_features.image_stats import ImageStats

DATASET = "coco_val2017"

# 10 frames stay outside `simulation_results`, for stage 1's training pool.
N_STAGE1_ONLY = 10
# 20 frames also get simulation rows, enough that an 80/20 photo split
# leaves both train (~16) and test (~4) non-empty.
N_SIMULATED = 20
ROWS_PER_FRAME = 2
LOCAL_LATENCY_MS = 50.0
OFFLOAD_LATENCY_MS = 150.0


def _frame_name(i: int) -> str:
    return f"frame_{i:03d}.jpg"


def _seed_dataset(engine: Engine, dataset: str, seed: int = 0) -> None:
    """Seeds `N_STAGE1_ONLY + N_SIMULATED` frames with varied features and
    both accuracies (so diagnostics get non-degenerate, finite correlations),
    then adds `simulation_results` rows for the last `N_SIMULATED` of them.

    Gap magnitude (0.05-0.25) stays above the cascade's escalate threshold at
    `OFFLOAD_LATENCY_MS = 150` (`DEFAULT_LAMBDA * offload_latency_ms / 1000 =
    0.045`), so the alternating sign also determines the cascade label, the
    same way it determines the stored decide-first label.
    """
    rng = np.random.default_rng(seed)
    n_frames = N_STAGE1_ONLY + N_SIMULATED
    local_accuracy = rng.uniform(0.4, 0.9, n_frames)
    sign = np.where(np.arange(n_frames) % 2 == 0, 1.0, -1.0)
    gap_magnitude = rng.uniform(0.05, 0.25, n_frames)
    offload_accuracy = np.clip(local_accuracy + sign * gap_magnitude, 0.0, 1.0)

    result_rows: list[ResultRow] = []
    for i in range(n_frames):
        file_name = _frame_name(i)
        store_complexity_scores(engine, dataset, {file_name: float(rng.uniform(0.0, 1.0))})
        store_model_inference(
            engine,
            dataset,
            {
                file_name: {
                    Label.LOCAL: DetectionResult(
                        latency_ms=LOCAL_LATENCY_MS, accuracy=float(local_accuracy[i])
                    ),
                    Label.OFFLOAD: DetectionResult(
                        latency_ms=OFFLOAD_LATENCY_MS, accuracy=float(offload_accuracy[i])
                    ),
                }
            },
        )
        store_features(
            engine,
            dataset,
            [
                FeatureRow(
                    file_name=file_name,
                    image_stats=ImageStats(
                        sharpness=float(rng.uniform(0.0, 200.0)),
                        brightness=float(rng.uniform(0.0, 1.0)),
                        contrast=float(rng.uniform(0.0, 1.0)),
                        colorfulness=float(rng.uniform(0.0, 100.0)),
                        entropy=float(rng.uniform(0.0, 8.0)),
                    ),
                    confidence=ConfidenceFeatures(
                        detection_count=int(rng.integers(0, 10)),
                        max_confidence=float(rng.uniform(0.0, 1.0)),
                        mean_confidence=float(rng.uniform(0.0, 1.0)),
                        min_confidence=float(rng.uniform(0.0, 1.0)),
                        mean_box_area=float(rng.uniform(0.0, 1.0)),
                        min_box_area=float(rng.uniform(0.0, 1.0)),
                    ),
                )
            ],
        )

        if i >= N_STAGE1_ONLY:
            label = Label.OFFLOAD if i % 2 == 0 else Label.LOCAL
            result_rows.extend(
                ResultRow(
                    frame_id=file_name,
                    network_bandwidth_mbps=float(rng.uniform(1.0, 50.0)),
                    network_latency_ms=float(rng.uniform(5.0, 100.0)),
                    network_packet_loss_pct=float(rng.uniform(0.0, 5.0)),
                    device_load_pct=float(rng.uniform(0.0, 100.0)),
                    local_latency_ms=LOCAL_LATENCY_MS,
                    local_accuracy=float(local_accuracy[i]),
                    offload_latency_ms=OFFLOAD_LATENCY_MS,
                    offload_accuracy=float(offload_accuracy[i]),
                    label=label,
                )
                for _ in range(ROWS_PER_FRAME)
            )

    run_id = create_run(
        engine,
        RunConfig(
            dataset=dataset,
            preset_name="baseline",
            frame_count=N_SIMULATED,
            condition_vector_count=ROWS_PER_FRAME,
            condition_ranges={"bandwidth_mbps": [0.5, 100.0]},
            seed=42,
            lambda_value=0.3,
        ),
    )
    store_results(engine, run_id, result_rows)


def test_run_experiment_is_deterministic_with_all_routers_finite(postgres_engine: Engine) -> None:
    _seed_dataset(postgres_engine, DATASET)

    first = run_experiment(postgres_engine, DATASET)
    second = run_experiment(postgres_engine, DATASET)

    assert first.diagnostics.equals(second.diagnostics)
    assert first.train_frame_count == second.train_frame_count
    assert first.test_frame_count == second.test_frame_count
    assert first.routers == second.routers

    assert first.train_frame_count > 0
    assert first.test_frame_count > 0

    expected_names = {
        "decide_first_linear",
        "decide_first_gbt",
        "cascade_linear",
        "cascade_gbt",
        "decide_first_linear_costaware",
        "decide_first_gbt_costaware",
        "cascade_linear_costaware",
        "cascade_gbt_costaware",
    }
    assert {router.name for router in first.routers} == expected_names
    assert len(first.routers) == 8

    for value in first.diagnostics["spearman_rho"]:
        assert math.isfinite(value)
    for value in first.diagnostics["p_value"]:
        assert math.isfinite(value)

    for router in first.routers:
        summary = router.summary
        bootstrap = router.bootstrap
        for value in (
            summary.avg_utility_router,
            summary.avg_utility_always_local,
            summary.avg_utility_always_offload,
            summary.avg_utility_oracle,
            summary.headroom_share,
            bootstrap.ci_low,
            bootstrap.ci_high,
        ):
            assert math.isfinite(value)

        # Cascade ceiling: `None` for decide_first rows, a finite number that
        # is at least that router's own average utility for cascade rows (it
        # is the best any ACCEPT/ESCALATE assignment on these rows could do).
        if router.design == "cascade":
            assert router.cascade_ceiling is not None
            assert math.isfinite(router.cascade_ceiling)
            assert router.cascade_ceiling >= summary.avg_utility_router
        else:
            assert router.cascade_ceiling is None

    assert math.isfinite(first.cascade_ceiling)
    assert math.isfinite(first.always_escalate)
