"""Wires Tasks 1-4 together: loads the frame/simulated data, trains the
offload-latency predictor and the learned confidence score, then evaluates
always-local, always-offload, budget-only, the tuned raw/learned cascades and
the within-budget oracle on the frame-level held-out split, at six latency
budgets. Prints a per-budget table, each score's tuned threshold and
bootstrap intervals, and a usefulness verdict, and writes the
accuracy-vs-latency figure.

Run as `python -m router.budget_experiment` from `training/`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from common.db import get_engine
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sqlalchemy import Engine

from router.budget_plot import BudgetPanelData, BudgetPoint, plot_budget_curves
from router.dataset import frame_level_split
from router.evaluation import BootstrapResult, bootstrap_mean_difference
from router.frame_dataset import load_frame_table, load_simulated_frame_ids, load_simulated_rows
from router.latency_policies import (
    LOCAL,
    OFFLOAD,
    RouterMetrics,
    budget_only_actions,
    cascade_actions,
    learned_score,
    learned_score_model,
    metrics,
    on_time_accuracy_per_row,
    oracle_actions,
    outcomes,
    predict_offload_latency,
    raw_score,
    train_offload_latency_model,
)
from router.two_stage import Stage1Models

DEFAULT_BUDGETS_MS: tuple[float, ...] = (100.0, 150.0, 200.0, 300.0, 400.0, 500.0)
SCORE_NAMES = ("raw", "learned")

# 0.00-1.00 in steps of 0.05 (21 values). `np.round` avoids the float
# drift `np.arange(0, 1.05, 0.05)` would otherwise accumulate.
THRESHOLDS: npt.NDArray[np.float64] = np.round(np.arange(21) * 0.05, 2)

# Sized to the longest label ("ESCALATE") for the same reason
# `latency_policies._ACTION_DTYPE` is — see that constant's comment. Never
# mutated after creation here (always-local/always-offload are constant
# arrays), but kept explicit and consistent regardless.
_ACTION_DTYPE = "<U8"


def _constant_actions(n: int, action: str) -> npt.NDArray[np.str_]:
    """`n`-length action array, every entry `action` (LOCAL or OFFLOAD) —
    always-local/always-offload's "policy"."""
    return np.full(n, action, dtype=_ACTION_DTYPE)


def _policy_metrics(
    rows: pd.DataFrame, actions: npt.NDArray[np.str_], budget_ms: float
) -> RouterMetrics:
    latency_ms, accuracy = outcomes(rows, actions)
    return metrics(latency_ms, accuracy, budget_ms)


def _on_time_per_row(
    rows: pd.DataFrame, actions: npt.NDArray[np.str_], budget_ms: float
) -> npt.NDArray[np.float64]:
    latency_ms, accuracy = outcomes(rows, actions)
    return on_time_accuracy_per_row(latency_ms, accuracy, budget_ms)


@dataclass(frozen=True)
class _SplitContext:
    """One split's (train or test) rows plus the per-row arrays every
    threshold/policy needs, computed once and reused across every budget."""

    rows: pd.DataFrame
    local_latency_ms: npt.NDArray[np.float64]
    predicted_offload_ms: npt.NDArray[np.float64]
    scores: dict[str, npt.NDArray[np.float64]]


def _make_context(
    rows: pd.DataFrame,
    latency_model: HistGradientBoostingRegressor,
    score_model: Stage1Models,
) -> _SplitContext:
    return _SplitContext(
        rows=rows,
        local_latency_ms=np.asarray(rows["local_latency_ms"], dtype=np.float64),
        predicted_offload_ms=predict_offload_latency(latency_model, rows),
        scores={"raw": raw_score(rows), "learned": learned_score(score_model, rows)},
    )


def _threshold_sweep(
    ctx: _SplitContext, score_name: str, budget_ms: float
) -> list[tuple[float, RouterMetrics]]:
    """Every threshold's `RouterMetrics` on `ctx`'s rows — the reported curve."""
    sweep = []
    for threshold in THRESHOLDS:
        actions = cascade_actions(
            ctx.local_latency_ms,
            ctx.predicted_offload_ms,
            ctx.scores[score_name],
            float(threshold),
            budget_ms,
        )
        sweep.append((float(threshold), _policy_metrics(ctx.rows, actions, budget_ms)))
    return sweep


def _tuned_threshold(ctx: _SplitContext, score_name: str, budget_ms: float) -> float:
    """The threshold with the highest on-time accuracy on `ctx`'s rows (meant
    to be called on train rows), ties won by the lowest threshold — only a
    strict improvement overwrites the running best, so the first (lowest)
    threshold to reach the maximum is the one kept."""
    best_threshold = float(THRESHOLDS[0])
    best_on_time = -np.inf
    for threshold in THRESHOLDS:
        actions = cascade_actions(
            ctx.local_latency_ms,
            ctx.predicted_offload_ms,
            ctx.scores[score_name],
            float(threshold),
            budget_ms,
        )
        on_time = _policy_metrics(ctx.rows, actions, budget_ms).on_time_accuracy
        if on_time > best_on_time:
            best_on_time = on_time
            best_threshold = float(threshold)
    return best_threshold


@dataclass(frozen=True)
class ScoreBudgetResult:
    """One confidence score's results at one budget: the full threshold
    sweep on the test rows, the threshold tuned on train rows, that tuned
    threshold's metrics on the test rows, and bootstrap 95% CIs for the
    tuned cascade's on-time accuracy minus budget-only's and minus
    always-local's (both over the test rows). `useful` mirrors
    `vs_budget_only.useful` — the spec's usefulness rule.

    `share_score_ge_tuned_threshold` is the share of test rows whose score
    is at or above the tuned threshold — the share the cascade keeps LOCAL
    on confidence grounds alone, before the budget fit check."""

    sweep: list[tuple[float, RouterMetrics]]
    tuned_threshold: float
    tuned_metrics: RouterMetrics
    vs_budget_only: BootstrapResult
    vs_always_local: BootstrapResult
    useful: bool
    share_score_ge_tuned_threshold: float


def _score_budget_result(
    score_name: str,
    train_ctx: _SplitContext,
    test_ctx: _SplitContext,
    budget_ms: float,
    budget_only_on_time: npt.NDArray[np.float64],
    always_local_on_time: npt.NDArray[np.float64],
) -> ScoreBudgetResult:
    sweep = _threshold_sweep(test_ctx, score_name, budget_ms)
    tuned_threshold = _tuned_threshold(train_ctx, score_name, budget_ms)

    tuned_actions = cascade_actions(
        test_ctx.local_latency_ms,
        test_ctx.predicted_offload_ms,
        test_ctx.scores[score_name],
        tuned_threshold,
        budget_ms,
    )
    tuned_metrics = _policy_metrics(test_ctx.rows, tuned_actions, budget_ms)
    tuned_on_time = _on_time_per_row(test_ctx.rows, tuned_actions, budget_ms)

    vs_budget_only = bootstrap_mean_difference(
        test_ctx.rows["frame_id"], tuned_on_time - budget_only_on_time, seed=42
    )
    vs_always_local = bootstrap_mean_difference(
        test_ctx.rows["frame_id"], tuned_on_time - always_local_on_time, seed=42
    )
    share_score_ge_tuned_threshold = float(
        (test_ctx.scores[score_name] >= tuned_threshold).mean()
    )

    return ScoreBudgetResult(
        sweep=sweep,
        tuned_threshold=tuned_threshold,
        tuned_metrics=tuned_metrics,
        vs_budget_only=vs_budget_only,
        vs_always_local=vs_always_local,
        useful=vs_budget_only.useful,
        share_score_ge_tuned_threshold=share_score_ge_tuned_threshold,
    )


@dataclass(frozen=True)
class BudgetResult:
    """Everything one latency budget needs: the four baseline/oracle
    policies' metrics on the test rows, each confidence score's tuned
    cascade result, and bootstrap 95% CIs for budget-only's on-time accuracy
    minus each static baseline's (both over the test rows)."""

    budget_ms: float
    always_local: RouterMetrics
    always_offload: RouterMetrics
    budget_only: RouterMetrics
    oracle: RouterMetrics
    scores: dict[str, ScoreBudgetResult]
    budget_only_vs_always_local: BootstrapResult
    budget_only_vs_always_offload: BootstrapResult


@dataclass(frozen=True)
class BudgetExperimentResult:
    """Everything the experiment prints: the held-out split's photo counts,
    the offload-dominance shares (computed once over the test rows, not per
    budget — a row's local/offload accuracy don't depend on the budget), and
    every budget's result, in budget order. Holds no `pd.DataFrame` field,
    unlike `feature_experiment.ExperimentResult` — plain `==` between two
    runs works without pandas's "truth value of a DataFrame is ambiguous"
    trap."""

    train_frame_count: int
    test_frame_count: int
    offload_dominance_share: float
    local_strictly_better_share: float
    budgets: list[BudgetResult]


def run_experiment(
    engine: Engine, dataset: str, budgets: Sequence[float] = DEFAULT_BUDGETS_MS
) -> BudgetExperimentResult:
    """Loads the data, splits by photo (seed 42), trains the offload-latency
    predictor on the train rows and the learned score on non-simulated
    frames, then evaluates every policy at each of `budgets` on the test rows.
    """
    frame_table = load_frame_table(engine, dataset)
    simulated_frame_ids = load_simulated_frame_ids(engine, dataset)
    simulated_rows = load_simulated_rows(engine, dataset)

    split_train, split_test = frame_level_split(simulated_rows, test_size=0.2, seed=42)
    train_frame_ids = set(split_train["frame_id"])
    test_frame_ids = set(split_test["frame_id"])
    # `frame_level_split`'s own DataFrames come back in a different row order
    # than `simulated_rows` (spec 02's learning) — only its `frame_id` *sets*
    # are used; both splits' rows are re-derived by filtering `simulated_rows`
    # itself, so every array built below lines up positionally with its rows.
    train_rows = simulated_rows[simulated_rows["frame_id"].isin(train_frame_ids)].reset_index(
        drop=True
    )
    test_rows = simulated_rows[simulated_rows["frame_id"].isin(test_frame_ids)].reset_index(
        drop=True
    )

    latency_model = train_offload_latency_model(train_rows)
    score_model = learned_score_model(frame_table, simulated_frame_ids)

    train_ctx = _make_context(train_rows, latency_model, score_model)
    test_ctx = _make_context(test_rows, latency_model, score_model)

    # Once over the test rows, not per budget: a row's local/offload
    # accuracy is fixed regardless of the budget being evaluated.
    test_local_accuracy = np.asarray(test_rows["local_accuracy"], dtype=np.float64)
    test_offload_accuracy = np.asarray(test_rows["offload_accuracy"], dtype=np.float64)
    offload_dominance_share = float((test_offload_accuracy >= test_local_accuracy).mean())
    local_strictly_better_share = float((test_local_accuracy > test_offload_accuracy).mean())

    budget_results = []
    for raw_budget_ms in budgets:
        budget_ms = float(raw_budget_ms)

        always_local_actions = _constant_actions(len(test_rows), LOCAL)
        always_offload_actions = _constant_actions(len(test_rows), OFFLOAD)
        budget_only_test_actions = budget_only_actions(test_ctx.predicted_offload_ms, budget_ms)
        oracle_test_actions = oracle_actions(test_rows, budget_ms)

        always_local_on_time = _on_time_per_row(test_rows, always_local_actions, budget_ms)
        always_offload_on_time = _on_time_per_row(test_rows, always_offload_actions, budget_ms)
        budget_only_on_time = _on_time_per_row(test_rows, budget_only_test_actions, budget_ms)

        scores = {
            score_name: _score_budget_result(
                score_name,
                train_ctx,
                test_ctx,
                budget_ms,
                budget_only_on_time,
                always_local_on_time,
            )
            for score_name in SCORE_NAMES
        }

        budget_results.append(
            BudgetResult(
                budget_ms=budget_ms,
                always_local=_policy_metrics(test_rows, always_local_actions, budget_ms),
                always_offload=_policy_metrics(test_rows, always_offload_actions, budget_ms),
                budget_only=_policy_metrics(test_rows, budget_only_test_actions, budget_ms),
                oracle=_policy_metrics(test_rows, oracle_test_actions, budget_ms),
                scores=scores,
                budget_only_vs_always_local=bootstrap_mean_difference(
                    test_rows["frame_id"], budget_only_on_time - always_local_on_time, seed=42
                ),
                budget_only_vs_always_offload=bootstrap_mean_difference(
                    test_rows["frame_id"], budget_only_on_time - always_offload_on_time, seed=42
                ),
            )
        )

    return BudgetExperimentResult(
        train_frame_count=len(train_frame_ids),
        test_frame_count=len(test_frame_ids),
        offload_dominance_share=offload_dominance_share,
        local_strictly_better_share=local_strictly_better_share,
        budgets=budget_results,
    )


def _sweep_points(sweep: list[tuple[float, RouterMetrics]]) -> list[BudgetPoint]:
    return [BudgetPoint(m.mean_latency_ms, m.on_time_accuracy) for _, m in sweep]


def _single_point(m: RouterMetrics) -> BudgetPoint:
    return BudgetPoint(m.mean_latency_ms, m.on_time_accuracy)


def build_budget_panel(budget: BudgetResult) -> BudgetPanelData:
    """`BudgetResult` -> `budget_plot.BudgetPanelData`, one panel per budget."""
    return BudgetPanelData(
        budget_ms=budget.budget_ms,
        sweep_curves={name: _sweep_points(budget.scores[name].sweep) for name in SCORE_NAMES},
        tuned_points={
            name: _single_point(budget.scores[name].tuned_metrics) for name in SCORE_NAMES
        },
        single_points={
            "budget-only": _single_point(budget.budget_only),
            "always-local": _single_point(budget.always_local),
            "always-offload": _single_point(budget.always_offload),
            "oracle": _single_point(budget.oracle),
        },
    )


_POLICY_ROWS = (
    "always-local",
    "always-offload",
    "budget-only",
    "cascade-raw",
    "cascade-learned",
    "oracle",
)


def _budget_policy_metrics(budget: BudgetResult) -> dict[str, RouterMetrics]:
    return {
        "always-local": budget.always_local,
        "always-offload": budget.always_offload,
        "budget-only": budget.budget_only,
        "cascade-raw": budget.scores["raw"].tuned_metrics,
        "cascade-learned": budget.scores["learned"].tuned_metrics,
        "oracle": budget.oracle,
    }


def _print_report(result: BudgetExperimentResult) -> None:
    print(
        f"Held-out split: {result.train_frame_count} train photos / "
        f"{result.test_frame_count} test photos.\n"
    )
    print(
        f"Held-out test rows: offload accuracy >= local accuracy on "
        f"{result.offload_dominance_share:.2%}; local is strictly better on "
        f"{result.local_strictly_better_share:.2%}.\n"
    )

    for budget in result.budgets:
        print(f"--- Budget: {budget.budget_ms:g} ms ---")
        policy_metrics = _budget_policy_metrics(budget)
        header = (
            f"{'policy':<17} {'mean acc':>10} {'on-time acc':>12} "
            f"{'mean lat ms':>12} {'over-budget %':>14}"
        )
        print(header)
        for name in _POLICY_ROWS:
            m = policy_metrics[name]
            print(
                f"{name:<17} {m.mean_accuracy:>10.4f} {m.on_time_accuracy:>12.4f} "
                f"{m.mean_latency_ms:>12.2f} {m.over_budget_share:>14.2%}"
            )
        print()

        bo_al, bo_ao = budget.budget_only_vs_always_local, budget.budget_only_vs_always_offload
        print(
            f"budget-only vs always-local 95% CI [{bo_al.ci_low:.4f}, {bo_al.ci_high:.4f}]  |  "
            f"budget-only vs always-offload 95% CI [{bo_ao.ci_low:.4f}, {bo_ao.ci_high:.4f}]"
        )
        print()

        for score_name in SCORE_NAMES:
            s = budget.scores[score_name]
            bo, al = s.vs_budget_only, s.vs_always_local
            print(
                f"{score_name} tuned threshold: {s.tuned_threshold:.2f}  |  "
                f"vs budget-only 95% CI [{bo.ci_low:.4f}, {bo.ci_high:.4f}] "
                f"useful={bo.useful}  |  "
                f"vs always-local 95% CI [{al.ci_low:.4f}, {al.ci_high:.4f}]"
            )
            print(
                f"{score_name} share of test rows with score >= tuned threshold: "
                f"{s.share_score_ge_tuned_threshold:.2%}"
            )
        print()

    print("--- Usefulness verdicts (tuned cascade vs. budget-only, on-time accuracy) ---")
    for budget in result.budgets:
        verdicts = ", ".join(
            f"{score_name}={budget.scores[score_name].useful}" for score_name in SCORE_NAMES
        )
        print(f"{budget.budget_ms:g} ms: {verdicts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latency-budget router evaluation: budget-only, cascade (raw/learned "
        "confidence), always-local, always-offload and the within-budget oracle."
    )
    parser.add_argument(
        "--dataset", default="coco_val2017", help="Dataset name to evaluate (default: coco_val2017)"
    )
    parser.add_argument(
        "--figure",
        default=str(Path(__file__).resolve().parent / "latency_budget_curves.png"),
        help="Path to write the accuracy-vs-latency figure to.",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    engine = get_engine()

    result = run_experiment(engine, args.dataset)
    _print_report(result)

    panels = [build_budget_panel(budget) for budget in result.budgets]
    plot_budget_curves(panels, args.figure)


if __name__ == "__main__":
    main()
