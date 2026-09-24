"""Tests for `router.evaluation`.

Avoids `import pytest` (per `training/tests/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv). Plain
`def test_...()` functions are enough for pytest to collect and run these.

Utilities below are hand-worked with DEFAULT_LAMBDA = 0.3.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from datagen.config import DEFAULT_LAMBDA
from router.evaluation import (
    bootstrap_router_vs_offload,
    cascade_ceiling_utility,
    escalated_utility,
    local_utility,
    offload_utility,
    score_cascade,
    score_decide_first,
    summarize,
)


def test_default_lambda_is_point_three() -> None:
    # Every hand-computed value below assumes this; fail loudly if it drifts.
    assert DEFAULT_LAMBDA == 0.3


def test_local_and_offload_utility_hand_computed() -> None:
    df = pd.DataFrame(
        {
            "local_accuracy": [0.8],
            "local_latency_ms": [100.0],
            "offload_accuracy": [0.9],
            "offload_latency_ms": [200.0],
        }
    )
    # 0.8 - 0.3 * 100/1000 = 0.77
    assert math.isclose(local_utility(df).iloc[0], 0.77)
    # 0.9 - 0.3 * 200/1000 = 0.84
    assert math.isclose(offload_utility(df).iloc[0], 0.84)


def test_score_decide_first_picks_matching_path_utility() -> None:
    df = pd.DataFrame(
        {
            "local_accuracy": [0.8, 0.5],
            "local_latency_ms": [100.0, 50.0],
            "offload_accuracy": [0.9, 0.6],
            "offload_latency_ms": [200.0, 150.0],
        }
    )
    picks = np.array(["LOCAL", "OFFLOAD"])
    scored = score_decide_first(df, picks)
    assert math.isclose(scored.iloc[0], local_utility(df).iloc[0])
    assert math.isclose(scored.iloc[1], offload_utility(df).iloc[1])


def test_score_cascade_charges_escalated_rows_local_plus_offload_latency() -> None:
    df = pd.DataFrame(
        {
            "local_accuracy": [0.8, 0.8],
            "local_latency_ms": [100.0, 100.0],
            "offload_accuracy": [0.9, 0.9],
            "offload_latency_ms": [200.0, 200.0],
        }
    )
    picks = np.array(["ACCEPT", "ESCALATE"])
    scored = score_cascade(df, picks)

    # ACCEPT: local utility, 0.77.
    assert math.isclose(scored.iloc[0], 0.77)
    # ESCALATE: offload accuracy charged local + offload latency:
    # 0.9 - 0.3 * (100 + 200)/1000 = 0.81 — not plain offload_utility (0.84).
    assert math.isclose(scored.iloc[1], 0.81)
    assert not math.isclose(scored.iloc[1], offload_utility(df).iloc[1])


def test_cascade_ceiling_utility_is_max_of_local_and_escalated() -> None:
    df = pd.DataFrame(
        {
            # Row 0: local wins. local_utility = 0.9 - 0.3*50/1000 = 0.885;
            # escalated_utility = 0.85 - 0.3*(50+200)/1000 = 0.775.
            # Row 1: escalated wins. local_utility = 0.5 - 0.3*50/1000 = 0.485;
            # escalated_utility = 0.95 - 0.3*(50+100)/1000 = 0.905.
            "local_accuracy": [0.9, 0.5],
            "local_latency_ms": [50.0, 50.0],
            "offload_accuracy": [0.85, 0.95],
            "offload_latency_ms": [200.0, 100.0],
        }
    )
    ceiling = cascade_ceiling_utility(df)
    local = local_utility(df)
    escalated = escalated_utility(df)

    assert math.isclose(local.iloc[0], 0.885)
    assert math.isclose(escalated.iloc[0], 0.775)
    assert math.isclose(ceiling.iloc[0], 0.885)  # local wins row 0

    assert math.isclose(local.iloc[1], 0.485)
    assert math.isclose(escalated.iloc[1], 0.905)
    assert math.isclose(ceiling.iloc[1], 0.905)  # escalated wins row 1

    # Never below either candidate.
    assert (ceiling >= local - 1e-12).all()
    assert (ceiling >= escalated - 1e-12).all()


def _oracle_df() -> pd.DataFrame:
    # Zero latency so utility == accuracy exactly, for round numbers.
    return pd.DataFrame(
        {
            "local_accuracy": [0.5, 0.2],
            "local_latency_ms": [0.0, 0.0],
            "offload_accuracy": [0.3, 0.6],
            "offload_latency_ms": [0.0, 0.0],
        }
    )


def test_summary_headroom_share_is_correct() -> None:
    df = _oracle_df()
    # Router matches the oracle pick on every row: LOCAL on row 0, OFFLOAD on row 1.
    router_utility = pd.Series([0.5, 0.6])
    summary = summarize(df, router_utility)

    assert math.isclose(summary.avg_utility_router, 0.55)
    assert math.isclose(summary.avg_utility_always_local, 0.35)
    assert math.isclose(summary.avg_utility_always_offload, 0.45)
    assert math.isclose(summary.avg_utility_oracle, 0.55)
    # (0.55 - 0.45) / (0.55 - 0.45) = 1.0 — router captures all the headroom.
    assert math.isclose(summary.headroom_share, 1.0)


def test_summary_headroom_share_is_nan_when_denominator_is_zero() -> None:
    # Offload wins every row, so oracle == offload and there is no headroom.
    df = pd.DataFrame(
        {
            "local_accuracy": [0.1, 0.2],
            "local_latency_ms": [0.0, 0.0],
            "offload_accuracy": [0.5, 0.6],
            "offload_latency_ms": [0.0, 0.0],
        }
    )
    router_utility = pd.Series([0.5, 0.6])  # matches offload exactly
    summary = summarize(df, router_utility)

    assert math.isclose(summary.avg_utility_oracle, summary.avg_utility_always_offload)
    assert math.isnan(summary.headroom_share)


def _bootstrap_df(n_frames: int = 20, rows_per_frame: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame_ids = np.repeat([f"frame_{i}" for i in range(n_frames)], rows_per_frame)
    n_rows = len(frame_ids)
    return pd.DataFrame(
        {
            "frame_id": frame_ids,
            "local_accuracy": rng.uniform(0.3, 0.9, size=n_rows),
            "local_latency_ms": rng.uniform(10, 100, size=n_rows),
            "offload_accuracy": rng.uniform(0.3, 0.9, size=n_rows),
            "offload_latency_ms": rng.uniform(50, 300, size=n_rows),
        }
    )


def test_bootstrap_is_identical_across_two_calls_with_same_seed() -> None:
    df = _bootstrap_df()
    router_utility = offload_utility(df) + 0.1
    first = bootstrap_router_vs_offload(df, router_utility, seed=42)
    second = bootstrap_router_vs_offload(df, router_utility, seed=42)
    assert first == second


def test_bootstrap_flags_a_router_always_better_by_a_constant_as_useful() -> None:
    df = _bootstrap_df()
    router_utility = offload_utility(df) + 0.1
    result = bootstrap_router_vs_offload(df, router_utility, seed=42)
    assert result.ci_low > 0
    assert result.useful is True


def test_bootstrap_does_not_flag_a_router_identical_to_offload_as_useful() -> None:
    df = _bootstrap_df()
    router_utility = offload_utility(df)
    result = bootstrap_router_vs_offload(df, router_utility, seed=42)
    assert result.ci_low <= 0
    assert result.useful is False
