"""Scenario-level policy outcomes for birds: one local cost and one offload cost per cell.

Unlike attempt 1's `coco.router.latency_policies`, no latency column is read per photo; every
photo shares the cell's two costs. The action pickers and metrics come from `shared.routing`.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt

from shared.routing import ACTION_DTYPE, ESCALATE, LOCAL, OFFLOAD


class Band(Enum):
    """Which policies a hard budget leaves room for, given the two costs."""

    NOTHING_FITS = "nothing fits"
    LOCAL_ONLY = "local only"
    OFFLOAD_NO_ESCALATION = "offload but no escalation"
    EVERYTHING = "everything"


def outcomes(
    actions: npt.NDArray[np.str_],
    *,
    local_ms: float,
    offload_ms: float,
    small_correct: npt.NDArray[np.float64],
    large_correct: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-photo `(latency_ms, correctness)`. LOCAL pays local and takes the small model's
    answer, OFFLOAD pays offload and takes the large one, ESCALATE pays both and takes the
    large one."""
    latency = np.select(
        [actions == LOCAL, actions == OFFLOAD], [local_ms, offload_ms], local_ms + offload_ms
    ).astype(np.float64)
    correct = np.where(actions == LOCAL, small_correct, large_correct)
    return latency, correct


def oracle_actions(
    *,
    local_ms: float,
    offload_ms: float,
    small_correct: npt.NDArray[np.float64],
    large_correct: npt.NDArray[np.float64],
    budget_ms: float,
) -> npt.NDArray[np.str_]:
    """Per photo, the most accurate option whose cost fits the budget; LOCAL when none fits.

    Ties go to the cheaper option, then to the order LOCAL, OFFLOAD, ESCALATE, as in attempt 1.
    """
    n = small_correct.shape[0]
    options = (
        (LOCAL, local_ms, small_correct),
        (OFFLOAD, offload_ms, large_correct),
        (ESCALATE, local_ms + offload_ms, large_correct),
    )
    actions = np.full(n, LOCAL, dtype=ACTION_DTYPE)
    best_accuracy = np.full(n, -np.inf)
    best_cost = np.full(n, np.inf)
    for label, cost, accuracy in options:
        if cost > budget_ms:
            continue
        better = (accuracy > best_accuracy) | ((accuracy == best_accuracy) & (cost < best_cost))
        actions[better] = label
        best_accuracy = np.where(better, accuracy, best_accuracy)
        best_cost = np.where(better, cost, best_cost)
    return actions


def classify_band(*, local_ms: float, offload_ms: float, budget_ms: float) -> Band:
    """The band a cell is in. Escalation needs local + offload, which always exceeds offload
    alone, so the escalation band sits above the offload band."""
    if budget_ms >= local_ms + offload_ms:
        return Band.EVERYTHING
    if budget_ms >= offload_ms:
        return Band.OFFLOAD_NO_ESCALATION
    if budget_ms >= local_ms:
        return Band.LOCAL_ONLY
    return Band.NOTHING_FITS


def lockout_width_ms(local_ms: float) -> float:
    """Width of the band where offload fits but escalation does not: it equals the local cost,
    so cheaper local inference widens the range of budgets in which the cascade can act."""
    return local_ms
