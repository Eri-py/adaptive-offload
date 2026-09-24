"""Tests for the utility-based win/loss label."""

from __future__ import annotations

from common.models import Label

from datagen import config
from datagen.labeling import compute_label

_TOLERANCE = 1e-9


def _isclose(actual: float, expected: float) -> bool:
    return abs(actual - expected) < _TOLERANCE


def test_local_wins_when_it_has_much_lower_latency_and_comparable_accuracy() -> None:
    # utility = accuracy - lambda * (latency_ms / 1000)
    # local:   0.70 - 1.0 * (50 / 1000)   = 0.70 - 0.05 = 0.65
    # offload: 0.72 - 1.0 * (2000 / 1000) = 0.72 - 2.0  = -1.28
    local_latency_ms = 50.0
    local_accuracy = 0.70
    offload_latency_ms = 2000.0
    offload_accuracy = 0.72
    lambda_value = 1.0

    local_utility = local_accuracy - lambda_value * (local_latency_ms / 1000.0)
    offload_utility = offload_accuracy - lambda_value * (offload_latency_ms / 1000.0)
    assert _isclose(local_utility, 0.65)
    assert _isclose(offload_utility, -1.28)
    assert local_utility > offload_utility

    label = compute_label(
        local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy, lambda_value
    )
    assert label is Label.LOCAL


def test_offload_wins_when_it_has_much_lower_latency_and_comparable_accuracy() -> None:
    # local:   0.60 - 1.0 * (2000 / 1000) = 0.60 - 2.0  = -1.4
    # offload: 0.62 - 1.0 * (50 / 1000)   = 0.62 - 0.05 = 0.57
    local_latency_ms = 2000.0
    local_accuracy = 0.60
    offload_latency_ms = 50.0
    offload_accuracy = 0.62
    lambda_value = 1.0

    local_utility = local_accuracy - lambda_value * (local_latency_ms / 1000.0)
    offload_utility = offload_accuracy - lambda_value * (offload_latency_ms / 1000.0)
    assert _isclose(local_utility, -1.4)
    assert _isclose(offload_utility, 0.57)
    assert offload_utility > local_utility

    label = compute_label(
        local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy, lambda_value
    )
    assert label is Label.OFFLOAD


def test_exact_tie_breaks_to_local() -> None:
    # local:   0.5 - 1.0 * (0 / 1000)     = 0.5
    # offload: 1.0 - 1.0 * (500 / 1000)   = 1.0 - 0.5 = 0.5
    local_latency_ms = 0.0
    local_accuracy = 0.5
    offload_latency_ms = 500.0
    offload_accuracy = 1.0
    lambda_value = 1.0

    local_utility = local_accuracy - lambda_value * (local_latency_ms / 1000.0)
    offload_utility = offload_accuracy - lambda_value * (offload_latency_ms / 1000.0)
    assert _isclose(local_utility, 0.5)
    assert _isclose(offload_utility, 0.5)

    label = compute_label(
        local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy, lambda_value
    )
    assert label is Label.LOCAL


def test_near_tie_still_picks_the_strictly_higher_utility_path() -> None:
    # local:   0.5    - 1.0 * (0 / 1000) = 0.5
    # offload: 0.5001 - 1.0 * (0 / 1000) = 0.5001, wins by a tiny margin
    local_latency_ms = 0.0
    local_accuracy = 0.5
    offload_latency_ms = 0.0
    offload_accuracy = 0.5001
    lambda_value = 1.0

    local_utility = local_accuracy - lambda_value * (local_latency_ms / 1000.0)
    offload_utility = offload_accuracy - lambda_value * (offload_latency_ms / 1000.0)
    assert offload_utility > local_utility

    label = compute_label(
        local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy, lambda_value
    )
    assert label is Label.OFFLOAD


def test_accepts_lambda_as_a_parameter_rather_than_a_hardcoded_default() -> None:
    # Same raw latency/accuracy inputs, two different lambda weightings, flip
    # the label — confirms lambda is actually threaded through, not ignored.
    local_latency_ms = 100.0
    local_accuracy = 0.70
    offload_latency_ms = 400.0
    offload_accuracy = 0.75

    # Small lambda: latency barely matters, offload's higher accuracy wins.
    # local:   0.70 - 0.001 * (100 / 1000) = 0.70 - 0.0001 = 0.6999
    # offload: 0.75 - 0.001 * (400 / 1000) = 0.75 - 0.0004 = 0.7496
    small_lambda_label = compute_label(
        local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy, lambda_value=0.001
    )
    assert small_lambda_label is Label.OFFLOAD

    # Large lambda: latency dominates, local's much lower latency wins.
    # local:   0.70 - 2.0 * (100 / 1000) = 0.70 - 0.2 = 0.50
    # offload: 0.75 - 2.0 * (400 / 1000) = 0.75 - 0.8 = -0.05
    large_lambda_label = compute_label(
        local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy, lambda_value=2.0
    )
    assert large_lambda_label is Label.LOCAL


def test_default_lambda_lets_a_realistic_latency_gap_flip_the_label() -> None:
    # Regression test for the bug where DEFAULT_LAMBDA (previously 0.005) was
    # scaled twice — once by the config value itself, once again by
    # labeling._utility's own /1000 — making the effective per-ms weight
    # (5e-6) too small for latency to ever outweigh a realistic accuracy gap.
    #
    # Both cases below share the same accuracy inputs (a realistic gap
    # between the local/offload stub paths' base accuracies, config.py's
    # LOCAL_BASE_ACCURACY=0.78 and OFFLOAD_BASE_ACCURACY=0.85) and differ only
    # in the size of the latency gap, to isolate the latency term's effect.
    local_accuracy = 0.78
    offload_accuracy = 0.85

    # Large latency gap (360ms, plausible under network-stress-like
    # conditions): local's latency advantage outweighs offload's higher
    # accuracy, so LOCAL wins.
    # local:   0.78 - 0.3 * (100 / 1000) = 0.78 - 0.03  = 0.75
    # offload: 0.85 - 0.3 * (460 / 1000) = 0.85 - 0.138 = 0.712
    large_gap_label = compute_label(
        local_latency_ms=100.0,
        local_accuracy=local_accuracy,
        offload_latency_ms=460.0,
        offload_accuracy=offload_accuracy,
        lambda_value=config.DEFAULT_LAMBDA,
    )
    assert large_gap_label is Label.LOCAL

    # Small latency gap (50ms): the accuracy gap dominates instead, so
    # OFFLOAD wins — confirms the label tracks the latency gap's size rather
    # than always favoring one path.
    # local:   0.78 - 0.3 * (100 / 1000) = 0.78 - 0.03  = 0.75
    # offload: 0.85 - 0.3 * (150 / 1000) = 0.85 - 0.045 = 0.805
    small_gap_label = compute_label(
        local_latency_ms=100.0,
        local_accuracy=local_accuracy,
        offload_latency_ms=150.0,
        offload_accuracy=offload_accuracy,
        lambda_value=config.DEFAULT_LAMBDA,
    )
    assert small_gap_label is Label.OFFLOAD
