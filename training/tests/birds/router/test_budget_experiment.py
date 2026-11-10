"""Tests for `birds.router.budget_experiment` on hand-built records.

Avoids `import pytest` (it breaks mypy against this venv); plain `def test_...()` functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from birds.router.budget_experiment import (
    Cell,
    PhotoRecords,
    evaluate_cell,
    fewest_escalations_to_match,
    records_from_frames,
    tune_threshold,
    validate_budgets,
)
from birds.router.policies import Band

# Photos 1-4. Small right on 1 and 3, large right on 1 and 2, so the oracle gets 3 of 4.
SMALL = [1.0, 0.0, 1.0, 0.0]
LARGE = [1.0, 1.0, 0.0, 0.0]
MAX_PROB = [0.9, 0.2, 0.8, 0.1]
MARGIN = [0.5, 0.1, 0.4, 0.05]


def _records(
    small: list[float], large: list[float], max_prob: list[float], margin: list[float]
) -> PhotoRecords:
    return PhotoRecords(
        photo_ids=np.array([str(i + 1) for i in range(len(small))]),
        small_correct=np.array(small),
        large_correct=np.array(large),
        confidence={"max_probability": np.array(max_prob), "top2_margin": np.array(margin)},
    )


def _tiny() -> PhotoRecords:
    return _records(SMALL, LARGE, MAX_PROB, MARGIN)


def test_hand_computed_on_time_accuracy_for_all_five_policies() -> None:
    result = evaluate_cell(_tiny(), _tiny(), Cell(local_ms=10.0, offload_ms=30.0, budget_ms=50.0))
    assert result.always_local.on_time_accuracy == 0.5
    assert result.always_offload.on_time_accuracy == 0.5
    assert result.budget_only.on_time_accuracy == 0.5
    assert result.oracle.on_time_accuracy == 0.75
    for measure, threshold in (("max_probability", 0.8), ("top2_margin", 0.4)):
        cascade = result.cascades[measure]
        assert cascade.threshold == threshold
        assert cascade.metrics.on_time_accuracy == 0.75
        assert cascade.metrics.mean_latency_ms == 25.0
        assert cascade.metrics.over_budget_share == 0.0
        assert cascade.escalated_share == 0.5
    assert result.band is Band.EVERYTHING
    assert result.lockout_width_ms == 10.0


def test_no_escalation_when_local_plus_offload_exceeds_budget() -> None:
    # Offload fits (30 <= 35) but escalation does not (40 > 35).
    result = evaluate_cell(_tiny(), _tiny(), Cell(local_ms=10.0, offload_ms=30.0, budget_ms=35.0))
    assert result.band is Band.OFFLOAD_NO_ESCALATION
    for cascade in result.cascades.values():
        assert cascade.escalated_share == 0.0
        assert cascade.metrics.mean_latency_ms == 10.0
        assert cascade.metrics.on_time_accuracy == result.always_local.on_time_accuracy


def test_everything_is_zero_when_local_cost_alone_exceeds_budget() -> None:
    result = evaluate_cell(_tiny(), _tiny(), Cell(local_ms=10.0, offload_ms=30.0, budget_ms=5.0))
    assert result.band is Band.NOTHING_FITS
    policies = [result.always_local, result.always_offload, result.budget_only, result.oracle]
    policies += [c.metrics for c in result.cascades.values()]
    assert all(p.on_time_accuracy == 0.0 for p in policies)


def test_oracle_is_at_least_every_policy_in_all_four_bands() -> None:
    seen = set()
    for budget in (5.0, 20.0, 35.0, 60.0):
        result = evaluate_cell(
            _tiny(), _tiny(), Cell(local_ms=10.0, offload_ms=30.0, budget_ms=budget)
        )
        seen.add(result.band)
        others = [result.always_local, result.always_offload, result.budget_only]
        others += [c.metrics for c in result.cascades.values()]
        assert all(result.oracle.on_time_accuracy >= o.on_time_accuracy for o in others)
    assert seen == set(Band)


def test_static_baselines_are_all_on_time_or_all_late() -> None:
    for budget in (5.0, 20.0, 35.0, 60.0):
        result = evaluate_cell(
            _tiny(), _tiny(), Cell(local_ms=10.0, offload_ms=30.0, budget_ms=budget)
        )
        assert result.always_local.over_budget_share in (0.0, 1.0)
        assert result.always_offload.over_budget_share in (0.0, 1.0)


def test_threshold_is_tuned_on_validation_records_only() -> None:
    cell = Cell(local_ms=10.0, offload_ms=30.0, budget_ms=50.0)
    other_test = _records(LARGE, SMALL, MAX_PROB[::-1], MARGIN[::-1])
    a = evaluate_cell(_tiny(), _tiny(), cell)
    b = evaluate_cell(_tiny(), other_test, cell)
    for measure in a.cascades:
        assert a.cascades[measure].threshold == b.cascades[measure].threshold
        assert a.cascades[measure].threshold == tune_threshold(_tiny(), measure, cell)


def test_two_runs_over_unchanged_records_are_equal() -> None:
    cell = Cell(local_ms=10.0, offload_ms=30.0, budget_ms=50.0)
    assert evaluate_cell(_tiny(), _tiny(), cell) == evaluate_cell(_tiny(), _tiny(), cell)


def test_fewest_escalations_to_match_always_offload() -> None:
    records = _records([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.1, 0.2, 0.3, 0.4], MARGIN)
    fits = Cell(local_ms=10.0, offload_ms=30.0, budget_ms=50.0)
    assert fewest_escalations_to_match(records, "max_probability", fits) == 2
    # Escalation cannot fit the budget, so the large model's accuracy is out of reach.
    locked = Cell(local_ms=10.0, offload_ms=30.0, budget_ms=35.0)
    assert fewest_escalations_to_match(records, "max_probability", locked) is None


def test_records_align_per_photo_by_numeric_id() -> None:
    def frame(correct: list[bool], prob: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "photo_id": ["10", "2"],
                "is_correct": correct,
                "max_probability": prob,
                "top2_margin": prob,
            }
        )

    small = frame([True, False], [0.9, 0.3])
    records = records_from_frames(small, frame([False, True], [0.5, 0.6]))
    assert records.photo_ids.tolist() == ["2", "10"]
    assert records.small_correct.tolist() == [0.0, 1.0]
    assert records.large_correct.tolist() == [1.0, 0.0]
    assert records.confidence["max_probability"].tolist() == [0.3, 0.9]


def test_validate_budgets_requires_a_lockout_and_an_escalation_budget() -> None:
    kwargs = {"local_ms": 5.0, "good_offload_ms": 30.0, "poor_offload_ms": 300.0}
    validate_budgets([20.0, 40.0], **kwargs)
    for bad in ([400.0], [20.0]):
        try:
            validate_budgets(bad, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")
