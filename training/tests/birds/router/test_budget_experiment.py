"""Tests for `birds.router.budget_experiment` on hand-built records.

Avoids `import pytest` (it breaks mypy against this venv); plain `def test_...()` functions.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from birds.evaluation.records import DATASET_NAME, PhotoPrediction
from birds.evaluation.store import write_predictions
from birds.router.boundary_plot import CASCADE
from birds.router.budget_experiment import (
    RECORD_COMMAND,
    Cell,
    PhotoRecords,
    evaluate_cell,
    fewest_escalations_to_match,
    format_report,
    grid_accuracy,
    records_from_frames,
    run_on_engine,
    tune_accuracy_threshold,
    tune_threshold,
    tuned_thresholds,
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


def test_tuned_threshold_is_the_same_in_every_escalation_band_cell() -> None:
    # Two different cells that both let everything fit: the threshold ignores L, O and B.
    first = Cell(local_ms=10.0, offload_ms=30.0, budget_ms=50.0)
    second = Cell(local_ms=3.0, offload_ms=7.0, budget_ms=500.0)
    once = tuned_thresholds(_tiny())
    for measure in once:
        assert tune_threshold(_tiny(), measure, first) == once[measure]
        assert tune_threshold(_tiny(), measure, second) == once[measure]
        assert tune_accuracy_threshold(_tiny(), measure) == once[measure]
    # Changing the *test* records leaves the validation-tuned threshold untouched.
    other_test = _records(LARGE, SMALL, MAX_PROB[::-1], MARGIN[::-1])
    for cell in (first, second):
        a = evaluate_cell(_tiny(), _tiny(), cell, once)
        b = evaluate_cell(_tiny(), other_test, cell, once)
        for measure in once:
            assert a.cascades[measure].threshold == b.cascades[measure].threshold == once[measure]


def test_precomputed_thresholds_do_not_change_the_result() -> None:
    for budget in (5.0, 20.0, 35.0, 60.0):
        cell = Cell(local_ms=10.0, offload_ms=30.0, budget_ms=budget)
        assert evaluate_cell(_tiny(), _tiny(), cell) == evaluate_cell(
            _tiny(), _tiny(), cell, tuned_thresholds(_tiny())
        )


def test_grid_matches_evaluate_cell_in_every_band() -> None:
    local_ms, offload_ms = [3.0, 10.0, 25.0], [5.0, 30.0, 60.0, 90.0]
    tuned = tuned_thresholds(_tiny())
    for budget in (5.0, 20.0, 35.0, 60.0, 120.0):
        grid = grid_accuracy(
            _tiny(),
            budget_ms=budget,
            local_ms=local_ms,
            offload_ms=offload_ms,
            threshold=tuned["top2_margin"],
            measure="top2_margin",
        )
        for i, offload in enumerate(offload_ms):
            for j, local in enumerate(local_ms):
                result = evaluate_cell(_tiny(), _tiny(), Cell(local, offload, budget))
                expected = {
                    "always-local": result.always_local,
                    "always-offload": result.always_offload,
                    "budget-only": result.budget_only,
                    CASCADE: result.cascades["top2_margin"].metrics,
                }
                for policy, m in expected.items():
                    assert grid[policy][i, j] == m.on_time_accuracy, (budget, local, offload)


def _seed(engine: Engine, splits: tuple[str, ...]) -> None:
    """Twelve photos per split (ids are unique across splits), the small model right on the
    confident ones."""
    for offset, split in enumerate(splits):
        for model, right_below in (("small", 8), ("large", 11)):
            rows = [
                PhotoPrediction(
                    photo_id=str(i + 100 * offset),
                    true_label=1,
                    predicted_label=1 if i <= right_below else 2,
                    is_correct=i <= right_below,
                    max_probability=0.1 + i / 20,
                    top2_margin=i / 20,
                )
                for i in range(1, 13)
            ]
            write_predictions(engine, DATASET_NAME, split, model, None, rows)


def _run(engine: Engine, figure: Path) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_on_engine(
            engine,
            budgets=[50.0, 100.0, 200.0, 400.0],
            server_compute_ms=10.0,
            payload_kb=11.39,
            figure_path=figure,
        )
    return out.getvalue()


def test_entry_point_prints_tables_and_writes_the_figure(postgres_engine: Engine) -> None:
    _seed(postgres_engine, ("val", "test"))
    with tempfile.TemporaryDirectory() as tmp:
        figure = Path(tmp) / "boundary.png"
        first = _run(postgres_engine, figure)
        assert figure.stat().st_size > 0
        figure.unlink()
        # Two runs over unchanged records print the same report.
        assert _run(postgres_engine, figure) == first
    assert "--- Budget: 50 ms ---" in first
    assert "cascade (top2_margin)" in first and "oracle" in first
    assert "Band summary" in first and "lockout" in first


def test_missing_records_stop_the_run_and_write_nothing(postgres_engine: Engine) -> None:
    _seed(postgres_engine, ("val",))  # the test split was never recorded
    with tempfile.TemporaryDirectory() as tmp:
        figure = Path(tmp) / "boundary.png"
        try:
            _run(postgres_engine, figure)
        except ValueError as exc:
            assert RECORD_COMMAND in str(exc)
        else:
            raise AssertionError("missing records should stop the run")
        assert not figure.exists()


def test_report_shows_both_bootstrap_intervals() -> None:
    """The spec requires an interval against budget-only *and* against always-local."""
    result = evaluate_cell(_tiny(), _tiny(), Cell(local_ms=10.0, offload_ms=30.0, budget_ms=50.0))
    report = format_report({50.0: {"good": [result]}}, {"good": 30.0})

    for measure in ("max_probability", "top2_margin"):
        assert f"cascade ({measure}): threshold" in report
    assert "vs budget-only 95% CI" in report
    assert "vs always-local 95% CI" in report
