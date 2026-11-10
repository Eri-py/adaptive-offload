"""Tests for `birds.router.policies`.

Avoids `import pytest` (it breaks mypy against this venv); plain `def test_...()` functions.
"""

from __future__ import annotations

import numpy as np

from birds.router.policies import Band, classify_band, lockout_width_ms, oracle_actions, outcomes
from shared.routing import ESCALATE, LOCAL, OFFLOAD

SMALL = np.array([1.0, 0.0, 1.0, 0.0])
LARGE = np.array([1.0, 1.0, 0.0, 0.0])


def test_escalated_photo_pays_both_costs_and_takes_large_correctness() -> None:
    actions = np.array([LOCAL, ESCALATE, OFFLOAD, ESCALATE], dtype="<U8")
    latency, correct = outcomes(
        actions, local_ms=10.0, offload_ms=30.0, small_correct=SMALL, large_correct=LARGE
    )
    assert latency.tolist() == [10.0, 40.0, 30.0, 40.0]
    assert correct.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_oracle_picks_best_fitting_option_and_falls_back_to_local() -> None:
    def pick(budget: float) -> list[str]:
        actions = oracle_actions(
            local_ms=10.0,
            offload_ms=30.0,
            small_correct=SMALL,
            large_correct=LARGE,
            budget_ms=budget,
        )
        return [str(a) for a in actions]

    # Everything fits: ties go to the cheaper, earlier option (LOCAL before OFFLOAD).
    assert pick(50.0) == [LOCAL, OFFLOAD, LOCAL, LOCAL]
    assert set(pick(20.0)) == {LOCAL}  # only local fits
    assert set(pick(5.0)) == {LOCAL}  # nothing fits: fall back to local


def test_bands_and_lockout_width() -> None:
    def band(budget: float) -> Band:
        return classify_band(local_ms=10.0, offload_ms=30.0, budget_ms=budget)

    assert band(5.0) is Band.NOTHING_FITS
    assert band(10.0) is Band.LOCAL_ONLY
    assert band(30.0) is Band.OFFLOAD_NO_ESCALATION
    assert band(39.9) is Band.OFFLOAD_NO_ESCALATION
    assert band(40.0) is Band.EVERYTHING
    assert lockout_width_ms(10.0) == 10.0
