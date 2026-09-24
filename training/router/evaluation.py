"""Utility scoring shared by every router design: per-row utilities, decide-first
and cascade scoring, the router/baseline/oracle summary with headroom share, and
the bootstrap confidence interval for router utility vs always-offload.

Every router evaluation (`two_stage.py`, `feature_experiment.py`) goes through
this module, so results stay comparable across designs — no reimplementation
of the utility formula per router.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from datagen.config import DEFAULT_LAMBDA


def compute_utility(
    accuracy: pd.Series, latency_ms: pd.Series, lambda_value: float = DEFAULT_LAMBDA
) -> pd.Series:
    """accuracy - lambda * latency_ms / 1000 — the formula labels were generated
    with. Reimplemented here (rather than imported) since `datagen.simulate.labeling`'s
    version is private, matching `router/baseline.py` and `router/analyze.py`'s
    own reimplementations.
    """
    return accuracy - lambda_value * (latency_ms / 1000.0)


def local_utility(df: pd.DataFrame) -> pd.Series:
    """Per-row utility of the local path: local accuracy and latency."""
    return compute_utility(df["local_accuracy"], df["local_latency_ms"])


def offload_utility(df: pd.DataFrame) -> pd.Series:
    """Per-row utility of the offload path: offload accuracy and latency."""
    return compute_utility(df["offload_accuracy"], df["offload_latency_ms"])


def escalated_utility(df: pd.DataFrame) -> pd.Series:
    """Per-row utility of a cascade ESCALATE: offload accuracy, charged both the
    local and the offload latency (the local pass already ran before escalating)."""
    return compute_utility(
        df["offload_accuracy"], df["local_latency_ms"] + df["offload_latency_ms"]
    )


def oracle_utility(df: pd.DataFrame) -> pd.Series:
    """Per-row ceiling: whichever of local/offload has the higher real utility."""
    return np.maximum(local_utility(df), offload_utility(df))


def cascade_ceiling_utility(df: pd.DataFrame) -> pd.Series:
    """Per-row ceiling for the cascade design: the best of ACCEPT (local utility)
    or ESCALATE (`escalated_utility`, charged both local and offload latency) a
    perfect cascade policy could pick.

    Strictly at or below `oracle_utility` row-wise, since `escalated_utility` can
    never exceed `offload_utility` (escalation pays the local pass's latency on
    top of the offload pass, on every row) — the oracle's OFFLOAD alternative is
    never available to a cascade router, which must run locally first.
    """
    return np.maximum(local_utility(df), escalated_utility(df))


def score_decide_first(df: pd.DataFrame, picks: npt.NDArray[np.str_]) -> pd.Series:
    """Per-row router utility for decide-first picks: a LOCAL pick gets local
    utility, an OFFLOAD pick gets offload utility."""
    return pd.Series(
        np.where(picks == "LOCAL", local_utility(df), offload_utility(df)), index=df.index
    )


def score_cascade(df: pd.DataFrame, picks: npt.NDArray[np.str_]) -> pd.Series:
    """Per-row router utility for cascade picks: an ACCEPT gets local utility, an
    ESCALATE gets `escalated_utility` (offload accuracy, local + offload latency)."""
    return pd.Series(
        np.where(picks == "ACCEPT", local_utility(df), escalated_utility(df)), index=df.index
    )


@dataclass(frozen=True)
class RouterSummary:
    """Average utility for the router against always-local, always-offload and
    the oracle, plus the router's share of the headroom above always-offload."""

    avg_utility_router: float
    avg_utility_always_local: float
    avg_utility_always_offload: float
    avg_utility_oracle: float
    headroom_share: float


def summarize(df: pd.DataFrame, router_utility: pd.Series) -> RouterSummary:
    """Average utility for the router, always-local, always-offload and the
    oracle, plus headroom share = (router - offload) / (oracle - offload).

    Headroom share is `nan` when the oracle and always-offload averages are
    equal — there is no headroom to share, so the ratio is undefined rather
    than zero.
    """
    avg_router = float(np.mean(router_utility))
    avg_local = float(local_utility(df).mean())
    avg_offload = float(offload_utility(df).mean())
    avg_oracle = float(oracle_utility(df).mean())
    denominator = avg_oracle - avg_offload
    headroom_share = (
        float("nan") if denominator == 0 else (avg_router - avg_offload) / denominator
    )
    return RouterSummary(
        avg_utility_router=avg_router,
        avg_utility_always_local=avg_local,
        avg_utility_always_offload=avg_offload,
        avg_utility_oracle=avg_oracle,
        headroom_share=headroom_share,
    )


@dataclass(frozen=True)
class BootstrapResult:
    """95% bootstrap CI for the mean per-row (router - offload) utility
    difference, resampling unique `frame_id`s with replacement, plus whether
    the router is useful (the lower bound is strictly above zero)."""

    ci_low: float
    ci_high: float
    useful: bool


def bootstrap_mean_difference(
    frame_ids: pd.Series | npt.NDArray[np.str_],
    diff: pd.Series | npt.NDArray[np.float64],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> BootstrapResult:
    """95% CI for the mean of `diff`, resampling `frame_ids`'s unique photos
    with replacement. A frame drawn twice in a resample counts its rows twice.

    Vectorised: each frame's sum of per-row `diff` values and row count are
    precomputed once, then every resample sums only those precomputed values
    for its drawn frames, instead of re-touching every row per resample.
    """
    diff_array = np.asarray(diff)
    frame_id_array = np.asarray(frame_ids)
    unique_frames, inverse = np.unique(frame_id_array, return_inverse=True)
    n_frames = len(unique_frames)

    frame_sums = np.zeros(n_frames)
    np.add.at(frame_sums, inverse, diff_array)
    frame_counts = np.bincount(inverse, minlength=n_frames)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_frames, size=(n_resamples, n_frames))
    resample_means = frame_sums[draws].sum(axis=1) / frame_counts[draws].sum(axis=1)

    ci_low, ci_high = np.percentile(resample_means, [2.5, 97.5])
    return BootstrapResult(ci_low=float(ci_low), ci_high=float(ci_high), useful=bool(ci_low > 0))


def bootstrap_router_vs_offload(
    df: pd.DataFrame,
    router_utility: pd.Series,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> BootstrapResult:
    """95% CI for the mean per-row (router - offload) utility difference,
    resampling `df["frame_id"]`'s unique photos with replacement. A frame drawn
    twice in a resample counts its rows twice. Delegates to
    `bootstrap_mean_difference`, the generic paired bootstrap.
    """
    diff = (router_utility - offload_utility(df)).to_numpy()
    return bootstrap_mean_difference(
        df["frame_id"], diff, n_resamples=n_resamples, seed=seed
    )
