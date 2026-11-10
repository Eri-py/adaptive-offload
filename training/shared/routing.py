"""Per-row routing primitives shared by both attempts: the action labels, the
policies that pick an action, the metrics they are scored with under a hard
latency budget, and the bootstrap interval for comparing two policies.

Everything here is vectorised and shape-agnostic — it takes arrays of costs
and scores, never a particular attempt's DataFrame. The attempt-specific
pieces stay with their attempt: how a row's latency and accuracy are looked up
(`outcomes`) and each attempt's oracle both depend on the shape of that
attempt's data, so they live in `coco/router/latency_policies.py` and
`birds/router/policies.py` respectively.

Lives in `shared/` rather than under an attempt folder because both attempts
use it — the guidelines' rule that code reused by both attempts moves to a
shared location on first reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

LOCAL = "LOCAL"
OFFLOAD = "OFFLOAD"
ESCALATE = "ESCALATE"

# <U8 fits "ESCALATE"; np.full would otherwise size the dtype from the fill value and truncate.
ACTION_DTYPE = "<U8"


def constant_actions(n: int, action: str) -> npt.NDArray[np.str_]:
    """`n`-length action array, every entry `action` (e.g. LOCAL or OFFLOAD)."""
    return np.full(n, action, dtype=ACTION_DTYPE)


def budget_only_actions(
    predicted_offload_ms: pd.Series | npt.NDArray[np.float64], budget_ms: float
) -> npt.NDArray[np.str_]:
    """OFFLOAD where the predicted offload latency fits the budget
    (`<= budget_ms`), else LOCAL."""
    predicted = np.asarray(predicted_offload_ms, dtype=np.float64)
    actions = np.full(predicted.shape, LOCAL, dtype=ACTION_DTYPE)
    actions[predicted <= budget_ms] = OFFLOAD
    return actions


def cascade_actions(
    local_latency_ms: pd.Series | npt.NDArray[np.float64],
    predicted_offload_ms: pd.Series | npt.NDArray[np.float64],
    score: pd.Series | npt.NDArray[np.float64],
    threshold: float,
    budget_ms: float,
) -> npt.NDArray[np.str_]:
    """LOCAL where `score >= threshold` (confident enough to keep the local
    result); otherwise ESCALATE where local + predicted offload latency fits
    the budget; otherwise LOCAL (escalating wouldn't fit)."""
    local = np.asarray(local_latency_ms, dtype=np.float64)
    predicted = np.asarray(predicted_offload_ms, dtype=np.float64)
    score_arr = np.asarray(score, dtype=np.float64)

    actions = np.full(local.shape, LOCAL, dtype=ACTION_DTYPE)
    below_threshold = score_arr < threshold
    fits_if_escalated = (local + predicted) <= budget_ms
    actions[below_threshold & fits_if_escalated] = ESCALATE
    return actions


def on_time_accuracy_per_row(
    latency_ms: pd.Series | npt.NDArray[np.float64],
    accuracy: pd.Series | npt.NDArray[np.float64],
    budget_ms: float,
) -> npt.NDArray[np.float64]:
    """Per-row accuracy with results that missed the budget (`latency_ms >
    budget_ms`) counted as 0 — the bootstrap's per-row input."""
    latency = np.asarray(latency_ms, dtype=np.float64)
    acc = np.asarray(accuracy, dtype=np.float64)
    return np.where(latency <= budget_ms, acc, 0.0)


@dataclass(frozen=True)
class RouterMetrics:
    """Summary of a policy's outcomes at one budget: mean accuracy, on-time
    accuracy (late results count as 0), mean latency and the share of rows
    that missed the budget."""

    mean_accuracy: float
    on_time_accuracy: float
    mean_latency_ms: float
    over_budget_share: float


def metrics(
    latency_ms: pd.Series | npt.NDArray[np.float64],
    accuracy: pd.Series | npt.NDArray[np.float64],
    budget_ms: float,
) -> RouterMetrics:
    """`RouterMetrics` for a policy's per-row `(latency_ms, accuracy)`
    outcomes at `budget_ms`."""
    latency = np.asarray(latency_ms, dtype=np.float64)
    acc = np.asarray(accuracy, dtype=np.float64)
    on_time = on_time_accuracy_per_row(latency, acc, budget_ms)
    return RouterMetrics(
        mean_accuracy=float(acc.mean()),
        on_time_accuracy=float(on_time.mean()),
        mean_latency_ms=float(latency.mean()),
        over_budget_share=float((latency > budget_ms).mean()),
    )


@dataclass(frozen=True)
class BootstrapResult:
    """95% bootstrap CI for a mean per-row difference between two policies,
    resampling unique group ids with replacement, plus whether the first
    policy is useful (the lower bound is strictly above zero)."""

    ci_low: float
    ci_high: float
    useful: bool


def bootstrap_mean_difference(
    group_ids: pd.Series | npt.NDArray[np.str_],
    diff: pd.Series | npt.NDArray[np.float64],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> BootstrapResult:
    """95% CI for the mean of `diff`, resampling `group_ids`'s unique groups
    with replacement. A group drawn twice in a resample counts its rows twice.
    Groups are whatever unit generalisation is claimed over — a frame in
    attempt 1, a photo in attempt 2.

    Vectorised: each group's sum of per-row `diff` values and row count are
    precomputed once, then every resample sums only those precomputed values
    for its drawn groups, instead of re-touching every row per resample.
    """
    diff_array = np.asarray(diff)
    group_id_array = np.asarray(group_ids)
    unique_groups, inverse = np.unique(group_id_array, return_inverse=True)
    n_groups = len(unique_groups)

    group_sums = np.zeros(n_groups)
    np.add.at(group_sums, inverse, diff_array)
    group_counts = np.bincount(inverse, minlength=n_groups)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    resample_means = group_sums[draws].sum(axis=1) / group_counts[draws].sum(axis=1)

    ci_low, ci_high = np.percentile(resample_means, [2.5, 97.5])
    return BootstrapResult(ci_low=float(ci_low), ci_high=float(ci_high), useful=bool(ci_low > 0))
