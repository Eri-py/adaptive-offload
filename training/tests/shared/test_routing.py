"""Tests for `shared.routing` — the routing primitives both attempts use.

Avoids `import pytest` (per `training/tests/coco/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv). Plain
`def test_...()` functions are enough for pytest to collect and run these.

"Fits the budget" means latency <= budget_ms, consistently tested at the
boundary (latency == budget_ms counts as fitting, on time, and not
over-budget).
"""

from __future__ import annotations

import math

import numpy as np

from shared.routing import (
    bootstrap_mean_difference,
    budget_only_actions,
    cascade_actions,
    constant_actions,
    metrics,
    on_time_accuracy_per_row,
)


def test_budget_only_offloads_within_budget_and_stays_local_otherwise() -> None:
    predicted = np.array([50.0, 100.0, 150.0])
    actions = budget_only_actions(predicted, budget_ms=100.0)
    # 50 and 100 fit (<=), 150 doesn't — boundary at exactly the budget offloads.
    assert list(actions) == ["OFFLOAD", "OFFLOAD", "LOCAL"]


def test_cascade_keeps_local_escalates_when_it_fits_and_falls_back_otherwise() -> None:
    local_latency = np.array([10.0, 10.0, 10.0, 10.0])
    predicted_offload = np.array([50.0, 50.0, 90.0, 200.0])
    score = np.array([0.9, 0.3, 0.3, 0.3])
    actions = cascade_actions(
        local_latency, predicted_offload, score, threshold=0.5, budget_ms=100.0
    )
    # Row 0 stays local; rows 1-2 escalate (fits, incl. boundary); row 3 falls back (over budget).
    assert list(actions) == ["LOCAL", "ESCALATE", "ESCALATE", "LOCAL"]


def test_constant_actions_fills_length_and_value_without_truncation() -> None:
    actions = constant_actions(3, "ESCALATE")
    assert len(actions) == 3
    # "ESCALATE" (8 chars) must survive intact, not truncate to "ESCALAT".
    assert list(actions) == ["ESCALATE", "ESCALATE", "ESCALATE"]


def test_cascade_threshold_zero_never_escalates() -> None:
    local_latency = np.array([10.0, 10.0, 10.0])
    predicted_offload = np.array([20.0, 20.0, 20.0])  # comfortably within budget
    score = np.array([0.0, 0.5, 1.0])
    actions = cascade_actions(
        local_latency, predicted_offload, score, threshold=0.0, budget_ms=1000.0
    )
    # Every score is >= 0, so score >= threshold always holds and nothing escalates.
    assert list(actions) == ["LOCAL", "LOCAL", "LOCAL"]


def test_on_time_accuracy_per_row_zeroes_late_results_at_the_boundary() -> None:
    latency_ms = np.array([99.0, 100.0, 101.0])
    accuracy = np.array([0.5, 0.6, 0.7])
    on_time = on_time_accuracy_per_row(latency_ms, accuracy, budget_ms=100.0)
    # 99 and 100 are on time (<=), 101 missed the budget and counts as 0.
    assert list(on_time) == [0.5, 0.6, 0.0]


def test_metrics_hand_computed_including_boundary() -> None:
    latency_ms = np.array([50.0, 100.0, 150.0])
    accuracy = np.array([0.6, 0.7, 0.8])
    result = metrics(latency_ms, accuracy, budget_ms=100.0)

    assert np.isclose(result.mean_accuracy, 0.7)  # (0.6+0.7+0.8)/3
    assert np.isclose(result.on_time_accuracy, 1.3 / 3)  # 150ms row zeroed
    assert np.isclose(result.mean_latency_ms, 100.0)  # (50+100+150)/3
    # Only the 150ms row exceeds the budget; the 100ms row is exactly at it.
    assert np.isclose(result.over_budget_share, 1 / 3)


def test_bootstrap_mean_difference_matches_hand_built_case() -> None:
    # A constant per-row diff means every resample's mean is that same
    # constant regardless of which groups get drawn — an exact expected CI.
    group_ids = np.array(["group_0", "group_0", "group_1", "group_2"])
    diff = np.array([0.1, 0.1, 0.1, 0.1])
    result = bootstrap_mean_difference(group_ids, diff, seed=42)
    assert math.isclose(result.ci_low, 0.1)
    assert math.isclose(result.ci_high, 0.1)
    assert result.useful is True


def test_bootstrap_mean_difference_is_deterministic_for_seed() -> None:
    rng = np.random.default_rng(0)
    # Several rows per group, so the group-weighted resampling actually matters.
    group_ids = np.repeat([f"group_{i}" for i in range(20)], 3)
    diff = rng.normal(0.1, 0.05, size=group_ids.shape[0])
    first = bootstrap_mean_difference(group_ids, diff, seed=42)
    second = bootstrap_mean_difference(group_ids, diff, seed=42)
    assert first == second
