"""End-to-end test for `router.budget_experiment.run_experiment`, against a
small synthetic dataset seeded in ephemeral Postgres (per Task 1's tests'
approach, since this integration exercises the real SQL joins).

Reuses `test_feature_experiment.py`'s fixture strategy (alternating
`local`/`offload` accuracy so stage 1's `gap <= 0` classifier sees both
classes in any 80/20 photo split), but simulated rows here also carry widely
varied local latency and a real 1/bandwidth offload-latency relationship, so
budget-only and the cascade both sometimes offload/escalate and sometimes
stay local across the whole 100-500ms budget range — a constant latency (as
`test_feature_experiment.py` uses, since it never routes by latency) would
make every budget's routing decision degenerate to the same action.
"""

from __future__ import annotations

import math

import numpy as np
from common.models import Label
from sqlalchemy import Engine

from datagen.persistence import (
    ResultRow,
    RunConfig,
    create_run,
    store_complexity_scores,
    store_model_inference,
    store_results,
)
from datagen.simulate.inference import DetectionResult
from router.budget_experiment import DEFAULT_BUDGETS_MS, SCORE_NAMES, run_experiment
from router.feature_store import FeatureRow, store_features
from router.frame_features.confidence import ConfidenceFeatures
from router.frame_features.image_stats import ImageStats

DATASET = "coco_val2017"

# 10 frames stay outside `simulation_results`, for stage 1's training pool.
N_STAGE1_ONLY = 10
# 20 frames also get simulation rows, enough that an 80/20 photo split
# leaves both train (~16) and test (~4) non-empty.
N_SIMULATED = 20
ROWS_PER_FRAME = 2


def _frame_name(i: int) -> str:
    return f"frame_{i:03d}.jpg"


def _seed_dataset(engine: Engine, dataset: str, seed: int = 0) -> None:
    """Seeds `N_STAGE1_ONLY + N_SIMULATED` frames with varied features and
    both accuracies, then adds `simulation_results` rows for the last
    `N_SIMULATED` of them with widely varied local latency (Uniform(20, 400))
    and offload latency following a real 1/bandwidth relationship (roughly
    10-1000ms) — both spanning the whole 100-500ms budget range, so every
    budget sees a genuine mix of fitting and non-fitting rows for both
    policies rather than a degenerate all-one-action split.

    Gap sign alternates by frame index (even -> offload better, odd -> local
    better), same as `test_feature_experiment.py`, so stage 1's `gap <= 0`
    classifier sees both classes in any contiguous training-pool slice.
    """
    rng = np.random.default_rng(seed)
    n_frames = N_STAGE1_ONLY + N_SIMULATED
    local_accuracy = rng.uniform(0.3, 0.95, n_frames)
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
                        latency_ms=50.0, accuracy=float(local_accuracy[i])
                    ),
                    Label.OFFLOAD: DetectionResult(
                        latency_ms=150.0, accuracy=float(offload_accuracy[i])
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
            for _ in range(ROWS_PER_FRAME):
                bandwidth = float(rng.uniform(1.0, 100.0))
                offload_latency_ms = max(1000.0 / bandwidth + float(rng.normal(0.0, 5.0)), 1.0)
                result_rows.append(
                    ResultRow(
                        frame_id=file_name,
                        network_bandwidth_mbps=bandwidth,
                        network_latency_ms=float(rng.uniform(5.0, 100.0)),
                        network_packet_loss_pct=float(rng.uniform(0.0, 5.0)),
                        device_load_pct=float(rng.uniform(0.0, 100.0)),
                        local_latency_ms=float(rng.uniform(20.0, 400.0)),
                        local_accuracy=float(local_accuracy[i]),
                        offload_latency_ms=offload_latency_ms,
                        offload_accuracy=float(offload_accuracy[i]),
                        label=label,
                    )
                )

    run_id = create_run(
        engine,
        RunConfig(
            dataset=dataset,
            preset_name="baseline",
            frame_count=N_SIMULATED,
            condition_vector_count=ROWS_PER_FRAME,
            condition_ranges={"bandwidth_mbps": [1.0, 100.0]},
            seed=42,
            lambda_value=0.3,
        ),
    )
    store_results(engine, run_id, result_rows)


def test_run_experiment_is_deterministic_with_every_budget_and_policy_finite(
    postgres_engine: Engine,
) -> None:
    _seed_dataset(postgres_engine, DATASET)

    first = run_experiment(postgres_engine, DATASET)
    second = run_experiment(postgres_engine, DATASET)

    assert first == second
    assert first.train_frame_count > 0
    assert first.test_frame_count > 0

    for share in (first.offload_dominance_share, first.local_strictly_better_share):
        assert math.isfinite(share)
        assert 0.0 <= share <= 1.0

    for value in (
        first.offload_latency_test_r2,
        first.offload_latency_test_mae,
        first.offload_latency_train_r2,
        first.offload_latency_train_mae,
    ):
        assert math.isfinite(value)
    assert first.offload_latency_test_mae >= 0.0
    assert first.offload_latency_train_mae >= 0.0

    assert len(first.budgets) == len(DEFAULT_BUDGETS_MS)
    assert [budget.budget_ms for budget in first.budgets] == list(DEFAULT_BUDGETS_MS)

    for budget in first.budgets:
        policies = {
            "always-local": budget.always_local,
            "always-offload": budget.always_offload,
            "budget-only": budget.budget_only,
            "oracle": budget.oracle,
        }
        assert set(SCORE_NAMES) == set(budget.scores)
        for score_name in SCORE_NAMES:
            policies[f"cascade-{score_name}"] = budget.scores[score_name].tuned_metrics

        for name, m in policies.items():
            for value in (m.mean_accuracy, m.on_time_accuracy, m.mean_latency_ms):
                assert math.isfinite(value), f"{name} at {budget.budget_ms}ms"
            assert math.isfinite(m.over_budget_share)

        for score_name in SCORE_NAMES:
            score_result = budget.scores[score_name]
            assert math.isfinite(score_result.tuned_threshold)
            assert math.isfinite(score_result.share_score_ge_tuned_threshold)
            assert 0.0 <= score_result.share_score_ge_tuned_threshold <= 1.0
            for threshold, sweep_metrics in score_result.sweep:
                assert math.isfinite(threshold)
                assert math.isfinite(sweep_metrics.on_time_accuracy)
            for bootstrap in (score_result.vs_budget_only, score_result.vs_always_local):
                assert math.isfinite(bootstrap.ci_low)
                assert math.isfinite(bootstrap.ci_high)

        for bootstrap in (
            budget.budget_only_vs_always_local,
            budget.budget_only_vs_always_offload,
        ):
            assert math.isfinite(bootstrap.ci_low)
            assert math.isfinite(bootstrap.ci_high)

        assert math.isfinite(budget.budget_only_true_on_time)
        assert 0.0 <= budget.budget_only_true_on_time <= 1.0

        # The oracle knows every row's true latencies/accuracies, so its
        # on-time accuracy is at least every other policy's at every budget.
        oracle_on_time = budget.oracle.on_time_accuracy
        for name, m in policies.items():
            if name == "oracle":
                continue
            assert oracle_on_time >= m.on_time_accuracy - 1e-9, f"{name} at {budget.budget_ms}ms"
        # Budget-only with true offload latency also only ever picks among
        # LOCAL/OFFLOAD, never ESCALATE, so the oracle's max over all three
        # options bounds it the same way.
        assert oracle_on_time >= budget.budget_only_true_on_time - 1e-9
