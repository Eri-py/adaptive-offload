"""Breaks down where the utility-regression router wins or loses against the
always-offload baseline, by preset condition and by the input features
themselves — the "which conditions does the router actually win" follow-up
to baseline.py's aggregate utility numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from common.db import get_engine
from dotenv import load_dotenv

from datagen.config import DEFAULT_LAMBDA
from router.baseline import predict_utility_regression_picks, train_utility_regressors
from router.dataset import FEATURE_COLUMNS, frame_level_split, load_training_data


def _utility(
    accuracy: pd.Series, latency_ms: pd.Series, lambda_value: float = DEFAULT_LAMBDA
) -> pd.Series:
    """Same formula as datagen.simulate.labeling's private `_utility` — reimplemented
    here rather than importing a leading-underscore name across packages, matching
    baseline.py's own `_utility`."""
    return accuracy - lambda_value * (latency_ms / 1000.0)


def _per_row_utilities(df: pd.DataFrame, picks: np.ndarray) -> pd.DataFrame:
    """Attach per-row local/offload/router/oracle utility columns to a copy of `df`.

    `oracle_utility` is the ceiling: per row, whichever of local/offload has the
    higher *real* (not predicted) utility. No router, however good, can beat it —
    it answers "is there any headroom above always-offload at all," independent
    of whether our specific router design is any good.
    """
    out = df.copy()
    out["local_utility"] = _utility(out["local_accuracy"], out["local_latency_ms"])
    out["offload_utility"] = _utility(out["offload_accuracy"], out["offload_latency_ms"])
    out["router_pick"] = picks
    out["router_utility"] = np.where(
        out["router_pick"] == "LOCAL", out["local_utility"], out["offload_utility"]
    )
    out["router_vs_offload"] = out["router_utility"] - out["offload_utility"]
    out["oracle_pick"] = np.where(out["local_utility"] > out["offload_utility"], "LOCAL", "OFFLOAD")
    out["oracle_utility"] = np.maximum(out["local_utility"], out["offload_utility"])
    out["oracle_vs_offload"] = out["oracle_utility"] - out["offload_utility"]
    return out


def breakdown_by_preset(df: pd.DataFrame) -> pd.DataFrame:
    """Average utility per preset for router vs oracle vs always-local vs
    always-offload, plus win/tie/loss counts of the router against
    always-offload specifically.
    """
    return (
        df.groupby("preset_name")
        .agg(
            rows=("router_utility", "size"),
            avg_utility_router=("router_utility", "mean"),
            avg_utility_oracle=("oracle_utility", "mean"),
            avg_utility_always_local=("local_utility", "mean"),
            avg_utility_always_offload=("offload_utility", "mean"),
            router_wins_vs_offload=("router_vs_offload", lambda s: int((s > 0).sum())),
            router_ties_vs_offload=("router_vs_offload", lambda s: int((s == 0).sum())),
            router_loses_vs_offload=("router_vs_offload", lambda s: int((s < 0).sum())),
            oracle_beats_offload=("oracle_vs_offload", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )


def condition_profile_by_pick(df: pd.DataFrame) -> pd.DataFrame:
    """Average feature values for rows where the router picked LOCAL vs OFFLOAD —
    shows what kind of conditions push the router away from the always-offload default.
    """
    return df.groupby("router_pick")[FEATURE_COLUMNS].mean()


def feature_margin_correlations(df: pd.DataFrame) -> pd.Series:
    """Pearson correlation of each router feature with the real per-row margin
    (local_utility - offload_utility) — how much information about "should this
    row go local" each feature alone carries. Run on the full dataset rather than
    just the held-out split: this is a diagnostic about the features themselves,
    not a trained model, so there's no leakage concern and more rows means a more
    reliable correlation estimate.
    """
    margin = _utility(df["local_accuracy"], df["local_latency_ms"]) - _utility(
        df["offload_accuracy"], df["offload_latency_ms"]
    )
    correlations: pd.Series = df[FEATURE_COLUMNS].corrwith(margin)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    engine = get_engine()

    df = load_training_data(engine, dataset="coco_val2017")

    print("--- Feature correlation with the real local-vs-offload margin (full dataset) ---")
    print(feature_margin_correlations(df).to_string())
    print()

    df_train, df_test = frame_level_split(df, test_size=0.2, seed=42)

    regressors = train_utility_regressors(df_train)
    picks = predict_utility_regression_picks(regressors, df_test)
    scored = _per_row_utilities(df_test, picks)

    print(f"Test set: {len(scored)} rows / {scored['frame_id'].nunique()} photos.\n")

    avg_router = scored["router_utility"].mean()
    avg_oracle = scored["oracle_utility"].mean()
    avg_offload = scored["offload_utility"].mean()
    headroom = avg_oracle - avg_offload
    captured = avg_router - avg_offload
    print("--- Oracle ceiling (real, not predicted, best-of-local/offload per row) ---")
    print(f"Always-offload avg utility:  {avg_offload:.4f}")
    print(
        f"Oracle avg utility:          {avg_oracle:.4f}  "
        f"(headroom over always-offload: {headroom:.4f})"
    )
    print(f"Router avg utility:          {avg_router:.4f}  (captured: {captured:.4f})")
    if headroom > 0:
        print(f"Router captured {captured / headroom:.1%} of the available headroom.")
    print(
        f"Oracle picks LOCAL on {int((scored['oracle_pick'] == 'LOCAL').sum())} of "
        f"{len(scored)} rows ({(scored['oracle_pick'] == 'LOCAL').mean():.1%})."
    )
    print()

    print("Router pick distribution (test set):")
    print(scored["router_pick"].value_counts().to_string())
    print()

    print("--- Utility by preset (router vs oracle vs always-local vs always-offload) ---")
    print(breakdown_by_preset(scored).to_string(index=False))
    print()

    print("--- Average condition values by router pick ---")
    print(condition_profile_by_pick(scored).to_string())
    print()

    local_mask = scored["router_pick"] == "LOCAL"
    n_local_picks = int(local_mask.sum())
    if n_local_picks:
        overall_avg = scored[FEATURE_COLUMNS].mean()
        local_pick_avg = scored.loc[local_mask, FEATURE_COLUMNS].mean()
        comparison = pd.DataFrame(
            {"local_picks_avg": local_pick_avg, "overall_test_avg": overall_avg}
        )
        print("--- Rows where router disagreed with always-offload (picked LOCAL) ---")
        print(f"{n_local_picks} of {len(scored)} rows ({n_local_picks / len(scored):.1%}).")
        print(comparison.to_string())
    else:
        print("Router never picked LOCAL on the test set — it always agreed with always-offload.")


if __name__ == "__main__":
    main()
