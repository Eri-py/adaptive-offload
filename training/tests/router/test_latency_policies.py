"""Tests for `router.latency_policies`.

Avoids `import pytest` (per `training/tests/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv). Plain
`def test_...()` functions are enough for pytest to collect and run these.

"Fits the budget" means latency <= budget_ms, consistently tested at the
boundary (latency == budget_ms counts as fitting, on time, and not
over-budget).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.latency_policies import (
    budget_only_actions,
    cascade_actions,
    metrics,
    on_time_accuracy_per_row,
    oracle_actions,
    outcomes,
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
    # Row 0: score >= threshold -> stays local.
    # Row 1: below threshold, 10+50=60 <= 100 -> escalates.
    # Row 2: below threshold, 10+90=100 <= 100 -> escalates (boundary, inclusive).
    # Row 3: below threshold, 10+200=210 > 100 -> doesn't fit, falls back to local.
    assert list(actions) == ["LOCAL", "ESCALATE", "ESCALATE", "LOCAL"]


def test_cascade_threshold_zero_never_escalates() -> None:
    local_latency = np.array([10.0, 10.0, 10.0])
    predicted_offload = np.array([20.0, 20.0, 20.0])  # comfortably within budget
    score = np.array([0.0, 0.5, 1.0])
    actions = cascade_actions(
        local_latency, predicted_offload, score, threshold=0.0, budget_ms=1000.0
    )
    # Every score is >= 0, so score >= threshold always holds and nothing escalates.
    assert list(actions) == ["LOCAL", "LOCAL", "LOCAL"]


def _rows(
    local_latency: list[float],
    local_accuracy: list[float],
    offload_latency: list[float],
    offload_accuracy: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "local_latency_ms": local_latency,
            "local_accuracy": local_accuracy,
            "offload_latency_ms": offload_latency,
            "offload_accuracy": offload_accuracy,
        }
    )


def test_outcomes_charges_escalate_local_plus_offload_latency_and_offload_accuracy() -> None:
    rows = _rows(
        local_latency=[100.0, 100.0, 100.0],
        local_accuracy=[0.5, 0.5, 0.5],
        offload_latency=[200.0, 200.0, 200.0],
        offload_accuracy=[0.9, 0.9, 0.9],
    )
    actions = np.array(["LOCAL", "OFFLOAD", "ESCALATE"])
    latency_ms, accuracy = outcomes(rows, actions)

    assert list(latency_ms) == [100.0, 200.0, 300.0]  # ESCALATE: 100 + 200
    assert list(accuracy) == [0.5, 0.9, 0.9]  # ESCALATE gets offload accuracy


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


def test_oracle_only_one_option_fits() -> None:
    rows = _rows(
        local_latency=[50.0],
        local_accuracy=[0.7],
        offload_latency=[150.0],
        offload_accuracy=[0.9],
    )
    # LOCAL fits (50<=100); OFFLOAD (150) and ESCALATE (200) don't.
    assert list(oracle_actions(rows, budget_ms=100.0)) == ["LOCAL"]


def test_oracle_falls_back_to_local_when_nothing_fits() -> None:
    rows = _rows(
        local_latency=[150.0],
        local_accuracy=[0.7],
        offload_latency=[150.0],
        offload_accuracy=[0.9],
    )
    # LOCAL (150), OFFLOAD (150) and ESCALATE (300) all exceed the 100ms budget.
    assert list(oracle_actions(rows, budget_ms=100.0)) == ["LOCAL"]


def test_oracle_picks_highest_accuracy_among_fitting_options() -> None:
    rows = _rows(
        local_latency=[30.0],
        local_accuracy=[0.6],
        offload_latency=[90.0],
        offload_accuracy=[0.95],
    )
    # LOCAL (30, acc 0.6) and OFFLOAD (90, acc 0.95) fit; ESCALATE (120) doesn't.
    # OFFLOAD has the higher accuracy among what fits.
    assert list(oracle_actions(rows, budget_ms=100.0)) == ["OFFLOAD"]


def test_oracle_breaks_accuracy_ties_by_lower_latency() -> None:
    rows = _rows(
        local_latency=[80.0],
        local_accuracy=[0.8],
        offload_latency=[60.0],
        offload_accuracy=[0.8],  # same accuracy as local
    )
    # LOCAL (80) and OFFLOAD (60) tie on accuracy; OFFLOAD is faster.
    # ESCALATE (140) doesn't fit the 100ms budget.
    assert list(oracle_actions(rows, budget_ms=100.0)) == ["OFFLOAD"]


def test_oracle_breaks_remaining_ties_by_local_offload_escalate_order() -> None:
    rows = _rows(
        local_latency=[50.0],
        local_accuracy=[0.75],
        offload_latency=[50.0],
        offload_accuracy=[0.75],  # same accuracy and latency as local
    )
    # LOCAL and OFFLOAD tie on both accuracy (0.75) and latency (50), so the
    # LOCAL/OFFLOAD/ESCALATE order picks LOCAL. ESCALATE (100) also fits but
    # has higher latency, so it never enters the tie.
    assert list(oracle_actions(rows, budget_ms=100.0)) == ["LOCAL"]


def test_oracle_on_time_accuracy_dominates_budget_only_and_cascade_per_row() -> None:
    rng = np.random.default_rng(7)
    n = 300
    budget_ms = 150.0
    local_latency = rng.uniform(20.0, 200.0, n)
    local_accuracy = rng.uniform(0.3, 0.95, n)
    offload_latency = rng.uniform(20.0, 250.0, n)
    offload_accuracy = rng.uniform(0.3, 0.95, n)
    rows = _rows(
        list(local_latency), list(local_accuracy), list(offload_latency), list(offload_accuracy)
    )
    # A noisy prediction of offload latency, as the router would actually use.
    predicted_offload = offload_latency + rng.normal(0.0, 20.0, n)
    score = rng.uniform(0.0, 1.0, n)

    oracle = oracle_actions(rows, budget_ms)
    oracle_latency, oracle_accuracy = outcomes(rows, oracle)
    oracle_on_time = on_time_accuracy_per_row(oracle_latency, oracle_accuracy, budget_ms)

    budget_only = budget_only_actions(predicted_offload, budget_ms)
    bo_latency, bo_accuracy = outcomes(rows, budget_only)
    bo_on_time = on_time_accuracy_per_row(bo_latency, bo_accuracy, budget_ms)
    assert (oracle_on_time >= bo_on_time - 1e-9).all()

    for threshold in (0.0, 0.25, 0.5, 0.75, 1.0):
        cascade = cascade_actions(local_latency, predicted_offload, score, threshold, budget_ms)
        cascade_latency, cascade_accuracy = outcomes(rows, cascade)
        cascade_on_time = on_time_accuracy_per_row(cascade_latency, cascade_accuracy, budget_ms)
        assert (oracle_on_time >= cascade_on_time - 1e-9).all()
