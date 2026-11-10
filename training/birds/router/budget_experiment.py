"""Evaluates the birds cascade under a hard latency budget, one cell at a time.

A cell names a local cost, an offload cost and a budget. For each cell the cascade's
threshold is tuned on the validation records alone, then applied to the test records, and all
five policies are scored. Everything is computed from stored predictions; no model is loaded.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from common.db import get_engine
from dotenv import load_dotenv
from sqlalchemy import Engine

from birds.config import ENV_PATH, EXPERIMENT_NAME, MODEL_NAMES
from birds.evaluation.records import DATASET_NAME
from birds.evaluation.store import read_predictions
from birds.evaluation.summary import CONFIDENCE_MEASURES, align_models
from birds.router.boundary_plot import (
    CASCADE,
    BoundaryMarks,
    BoundaryPanelData,
    plot_boundary_maps,
)
from birds.router.conditions import (
    LOCAL_SWEEP_MS,
    OFFLOAD_SWEEP_MS,
    PAYLOAD_KB,
    SMALL_DESKTOP_CPU_MS,
    named_offload_costs,
)
from birds.router.policies import Band, classify_band, lockout_width_ms, oracle_actions, outcomes
from shared import tracking
from shared.routing import (
    LOCAL,
    OFFLOAD,
    BootstrapResult,
    RouterMetrics,
    bootstrap_mean_difference,
    budget_only_actions,
    cascade_actions,
    constant_actions,
    metrics,
    on_time_accuracy_per_row,
)

RECORD_COMMAND = "python -m birds.evaluation.predictions"
VALIDATION_SPLIT, TEST_SPLIT = "val", "test"
# Reported budgets, tight to loose. Checked against the named conditions at run time, because
# the conditions' offload costs depend on the supplied server compute.
DEFAULT_BUDGETS_MS = (50.0, 100.0, 200.0, 400.0)
# Local costs at which tables are printed: the desktop measurement, then two slower stand-ins.
MARKED_LOCAL_MS = (SMALL_DESKTOP_CPU_MS, 20.0, 50.0)
# The confidence measure drawn in the figure: the margin recovers more of the gap at every level.
FIGURE_MEASURE = "top2_margin"
DEFAULT_FIGURE = Path(__file__).resolve().parent / "latency_budget_boundary.png"


@dataclass(frozen=True)
class Cell:
    """One evaluated scenario: what each path costs and the budget the result must meet."""

    local_ms: float
    offload_ms: float
    budget_ms: float


@dataclass(frozen=True)
class PhotoRecords:
    """One split's stored predictions, aligned per photo (ordered by numeric photo id).
    `confidence` holds the small model's confidence per measure."""

    photo_ids: npt.NDArray[np.str_]
    small_correct: npt.NDArray[np.float64]
    large_correct: npt.NDArray[np.float64]
    confidence: dict[str, npt.NDArray[np.float64]]


def records_from_frames(small: pd.DataFrame, large: pd.DataFrame) -> PhotoRecords:
    """Align the two models' stored rows per photo."""
    joined = align_models(small, large)
    return PhotoRecords(
        photo_ids=joined["photo_id"].to_numpy(dtype=np.str_),
        small_correct=joined["is_correct_small"].to_numpy(dtype=np.float64),
        large_correct=joined["is_correct_large"].to_numpy(dtype=np.float64),
        confidence={
            m: joined[f"{m}_small"].to_numpy(dtype=np.float64) for m in CONFIDENCE_MEASURES
        },
    )


def load_records(engine: Engine, split: str) -> PhotoRecords:
    """Both models' stored predictions for `split`, aligned per photo. Raises if none are
    stored, naming the command that records them."""
    small, large = (read_predictions(engine, DATASET_NAME, split, m) for m in MODEL_NAMES)
    if small.empty or large.empty:
        raise ValueError(
            f"no stored predictions for the {split!r} split. Run `{RECORD_COMMAND}` first."
        )
    return records_from_frames(small, large)


def _candidate_thresholds(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Every distinct score (the first escalates nothing), plus one just above the maximum
    (escalates everything), ascending so ties resolve to the lowest threshold."""
    return np.append(np.unique(scores), np.nextafter(scores.max(), np.inf))


def _policy_rows(
    actions: npt.NDArray[np.str_], records: PhotoRecords, cell: Cell
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-photo latency, correctness and on-time correctness. Correctness is returned rather
    than recomputed by callers, so the action-to-correctness mapping lives only in `outcomes`."""
    latency, correct = outcomes(
        actions,
        local_ms=cell.local_ms,
        offload_ms=cell.offload_ms,
        small_correct=records.small_correct,
        large_correct=records.large_correct,
    )
    return latency, correct, on_time_accuracy_per_row(latency, correct, cell.budget_ms)


def _cascade_actions(
    records: PhotoRecords, measure: str, threshold: float, cell: Cell
) -> npt.NDArray[np.str_]:
    n = len(records.photo_ids)
    return cascade_actions(
        np.full(n, cell.local_ms),
        np.full(n, cell.offload_ms),
        records.confidence[measure],
        threshold,
        cell.budget_ms,
    )


def tune_accuracy_threshold(validation: PhotoRecords, measure: str) -> float:
    """The threshold with the highest plain accuracy on the validation records, the lowest
    among ties: `mean(score < threshold ? large_correct : small_correct)`.

    This is the cascade's tuned threshold in every cell where escalation fits the budget. There
    every result arrives within budget, so on-time accuracy equals plain accuracy and depends on
    the threshold alone, never on the local cost, offload cost or budget. Compute it once per
    measure and reuse it across cells. Takes validation only, so no test photo can influence it."""
    scores = validation.confidence[measure]
    best_threshold, best_score = 0.0, -1.0
    for threshold in _candidate_thresholds(scores):
        accuracy = float(
            np.where(scores < threshold, validation.large_correct, validation.small_correct).mean()
        )
        if accuracy > best_score:
            best_threshold, best_score = float(threshold), accuracy
    return best_threshold


def escalation_fits(cell: Cell) -> bool:
    """Whether a kept result and an escalated one both arrive within the budget. Costs are
    non-negative, so this is the only band in which the threshold affects on-time accuracy:
    with local over budget everything is late, and with local + offload over budget nothing
    escalates, so the cascade is always-local for every threshold."""
    return cell.local_ms <= cell.budget_ms and cell.local_ms + cell.offload_ms <= cell.budget_ms


def tune_threshold(
    validation: PhotoRecords,
    measure: str,
    cell: Cell,
    tuned: dict[str, float] | None = None,
) -> float:
    """The cascade threshold for `cell`, tuned on validation. Where escalation fits it is the
    plain-accuracy optimum (`tuned[measure]` if already computed, to avoid repeating the pass);
    elsewhere every threshold ties, so the lowest candidate is returned."""
    if escalation_fits(cell):
        if tuned is not None:
            return tuned[measure]
        return tune_accuracy_threshold(validation, measure)
    return float(validation.confidence[measure].min())


def tuned_thresholds(validation: PhotoRecords) -> dict[str, float]:
    """`tune_accuracy_threshold` for every confidence measure."""
    return {m: tune_accuracy_threshold(validation, m) for m in CONFIDENCE_MEASURES}


def fewest_escalations_to_match(records: PhotoRecords, measure: str, cell: Cell) -> int | None:
    """Fewest least-confident photos sent to the large model for the cascade to reach
    always-offload's accuracy, or None if it never does. Ties on confidence fall back to the
    numeric photo id (records are in that order and the sort is stable). Where escalation does
    not fit the budget no photo can be escalated, so only zero escalations are possible."""
    small, large = records.small_correct, records.large_correct
    order = np.argsort(records.confidence[measure], kind="stable")
    gained = np.concatenate([[0.0], np.cumsum(large[order] - small[order])])
    if not escalation_fits(cell):
        gained = gained[:1]
    reaches = np.flatnonzero(small.sum() + gained >= large.sum())
    return int(reaches[0]) if reaches.size else None


@dataclass(frozen=True)
class CascadeResult:
    """The tuned cascade under one confidence measure, scored on the test records. `useful`
    is the rule: the interval against budget-only lies entirely above zero."""

    threshold: float
    metrics: RouterMetrics
    escalated_share: float
    vs_budget_only: BootstrapResult
    vs_always_local: BootstrapResult
    useful: bool
    fewest_escalations_to_match: int | None


@dataclass(frozen=True)
class CellResult:
    """All five policies in one cell. Holds no DataFrame, so two runs compare with plain `==`
    (attempt 1's result type documents the trap)."""

    cell: Cell
    band: Band
    lockout_width_ms: float
    always_local: RouterMetrics
    always_offload: RouterMetrics
    budget_only: RouterMetrics
    oracle: RouterMetrics
    cascades: dict[str, CascadeResult]


def _metrics(actions: npt.NDArray[np.str_], records: PhotoRecords, cell: Cell) -> RouterMetrics:
    latency, correct, _ = _policy_rows(actions, records, cell)
    return metrics(latency, correct, cell.budget_ms)


def evaluate_cell(
    validation: PhotoRecords,
    test: PhotoRecords,
    cell: Cell,
    tuned: dict[str, float] | None = None,
) -> CellResult:
    """Score all five policies on `test`, with the cascade threshold tuned on `validation`.
    `tuned` (from `tuned_thresholds`) skips repeating the per-measure tuning pass."""
    n = len(test.photo_ids)
    local = constant_actions(n, LOCAL)
    offload = constant_actions(n, OFFLOAD)
    budget_only = budget_only_actions(np.full(n, cell.offload_ms), cell.budget_ms)
    oracle = oracle_actions(
        local_ms=cell.local_ms,
        offload_ms=cell.offload_ms,
        small_correct=test.small_correct,
        large_correct=test.large_correct,
        budget_ms=cell.budget_ms,
    )
    *_, local_on_time = _policy_rows(local, test, cell)
    *_, budget_only_on_time = _policy_rows(budget_only, test, cell)

    cascades = {}
    for measure in CONFIDENCE_MEASURES:
        threshold = tune_threshold(validation, measure, cell, tuned)
        actions = _cascade_actions(test, measure, threshold, cell)
        *_, on_time = _policy_rows(actions, test, cell)
        vs_budget_only = bootstrap_mean_difference(test.photo_ids, on_time - budget_only_on_time)
        cascades[measure] = CascadeResult(
            threshold=threshold,
            metrics=_metrics(actions, test, cell),
            escalated_share=float((actions != LOCAL).mean()),
            vs_budget_only=vs_budget_only,
            vs_always_local=bootstrap_mean_difference(test.photo_ids, on_time - local_on_time),
            useful=vs_budget_only.useful,
            fewest_escalations_to_match=fewest_escalations_to_match(test, measure, cell),
        )

    return CellResult(
        cell=cell,
        band=classify_band(
            local_ms=cell.local_ms, offload_ms=cell.offload_ms, budget_ms=cell.budget_ms
        ),
        lockout_width_ms=lockout_width_ms(cell.local_ms),
        always_local=_metrics(local, test, cell),
        always_offload=_metrics(offload, test, cell),
        budget_only=_metrics(budget_only, test, cell),
        oracle=_metrics(oracle, test, cell),
        cascades=cascades,
    )


def validate_budgets(
    budgets: Sequence[float],
    *,
    local_ms: float,
    good_offload_ms: float,
    poor_offload_ms: float,
) -> None:
    """Raise unless at least one budget admits no offload under the poor condition and at
    least one admits escalation (local + offload) under the good one."""
    if not any(b < poor_offload_ms for b in budgets):
        raise ValueError("no budget is below the poor condition's offload cost")
    if not any(b >= local_ms + good_offload_ms for b in budgets):
        raise ValueError("no budget admits escalation under the good condition")


def grid_accuracy(
    test: PhotoRecords,
    *,
    budget_ms: float,
    local_ms: Sequence[float],
    offload_ms: Sequence[float],
    threshold: float,
    measure: str,
) -> dict[str, npt.NDArray[np.float64]]:
    """On-time accuracy of the four deployable policies over the whole grid, shaped
    `(len(offload_ms), len(local_ms))`, using the closed form behind `escalation_fits`: each
    static policy is all-on-time or all-late, and the cascade is plain accuracy at `threshold`
    where escalation fits and always-local elsewhere. `threshold` is tuned once, not per cell."""
    local, offload = np.meshgrid(np.asarray(local_ms), np.asarray(offload_ms))
    small_accuracy = float(test.small_correct.mean())
    large_accuracy = float(test.large_correct.mean())
    cascade_accuracy = float(
        np.where(
            test.confidence[measure] < threshold, test.large_correct, test.small_correct
        ).mean()
    )
    always_local = np.where(local <= budget_ms, small_accuracy, 0.0)
    offload_fits = offload <= budget_ms
    return {
        "always-local": always_local,
        "always-offload": np.where(offload_fits, large_accuracy, 0.0),
        "budget-only": np.where(offload_fits, large_accuracy, always_local),
        CASCADE: np.where(
            (local <= budget_ms) & (local + offload <= budget_ms), cascade_accuracy, always_local
        ),
    }


def _policy_table(result: CellResult) -> list[tuple[str, RouterMetrics]]:
    rows = [
        ("always-local", result.always_local),
        ("always-offload", result.always_offload),
        ("budget-only", result.budget_only),
    ]
    rows += [(f"cascade ({m})", c.metrics) for m, c in result.cascades.items()]
    return [*rows, ("oracle", result.oracle)]


def format_report(
    results: dict[float, dict[str, list[CellResult]]], offload_costs: dict[str, float]
) -> str:
    """Per-budget tables at the marked cells, then the band summary."""
    lines: list[str] = []
    for budget, by_condition in results.items():
        lines.append(f"--- Budget: {budget:g} ms ---")
        for condition, cells in by_condition.items():
            for result in cells:
                lines += [
                    "",
                    f"{condition} network (offload {result.cell.offload_ms:.1f} ms), "
                    f"local {result.cell.local_ms:g} ms: band '{result.band.value}'",
                    f"  {'policy':<30}{'mean acc':>10}{'on-time acc':>13}"
                    f"{'mean lat ms':>13}{'over-budget %':>15}",
                ]
                lines += [
                    f"  {name:<30}{m.mean_accuracy:>10.4f}{m.on_time_accuracy:>13.4f}"
                    f"{m.mean_latency_ms:>13.2f}{m.over_budget_share:>15.2%}"
                    for name, m in _policy_table(result)
                ]
                for measure, c in result.cascades.items():
                    lo, hi = c.vs_budget_only.ci_low, c.vs_budget_only.ci_high
                    local_lo, local_hi = c.vs_always_local.ci_low, c.vs_always_local.ci_high
                    # Both intervals are reported, per the spec; only the budget-only one
                    # decides usefulness.
                    lines.append(
                        f"  cascade ({measure}): threshold {c.threshold:.4f}, escalates "
                        f"{c.escalated_share:.2%}, vs budget-only 95% CI [{lo:+.4f}, {hi:+.4f}] "
                        f"-> {'useful' if c.useful else 'not useful'}; vs always-local 95% CI "
                        f"[{local_lo:+.4f}, {local_hi:+.4f}]; fewest escalations to "
                        f"match always-offload: {c.fewest_escalations_to_match}"
                    )
        lines.append("")
    lines.append("Band summary (lockout width = local cost)")
    lines.append(f"  {'budget':>8}  {'network':<8}{'local':>8}{'offload':>10}  band")
    for budget, by_condition in results.items():
        for condition, cells in by_condition.items():
            lines += [
                f"  {budget:>8g}  {condition:<8}{r.cell.local_ms:>8g}"
                f"{offload_costs[condition]:>10.1f}"
                f"  {r.band.value} (lockout {r.lockout_width_ms:g} ms)"
                for r in cells
            ]
    return "\n".join(lines)


def run_experiment(
    validation: PhotoRecords,
    test: PhotoRecords,
    *,
    budgets: Sequence[float],
    server_compute_ms: float,
    payload_kb: float,
) -> tuple[dict[float, dict[str, list[CellResult]]], dict[str, float], list[BoundaryPanelData]]:
    """Every marked cell's result, the named offload costs, and the boundary grid per budget."""
    offload_costs = named_offload_costs(payload_kb=payload_kb, server_compute_ms=server_compute_ms)
    validate_budgets(
        budgets,
        local_ms=min(MARKED_LOCAL_MS),
        good_offload_ms=offload_costs["good"],
        poor_offload_ms=offload_costs["poor"],
    )
    tuned = tuned_thresholds(validation)
    results = {
        budget: {
            condition: [
                evaluate_cell(validation, test, Cell(local, offload, budget), tuned)
                for local in MARKED_LOCAL_MS
            ]
            for condition, offload in offload_costs.items()
        }
        for budget in budgets
    }
    panels = [
        BoundaryPanelData(
            budget_ms=budget,
            local_ms=LOCAL_SWEEP_MS.tolist(),
            offload_ms=OFFLOAD_SWEEP_MS.tolist(),
            accuracy=grid_accuracy(
                test,
                budget_ms=budget,
                local_ms=LOCAL_SWEEP_MS.tolist(),
                offload_ms=OFFLOAD_SWEEP_MS.tolist(),
                threshold=tuned[FIGURE_MEASURE],
                measure=FIGURE_MEASURE,
            ),
        )
        for budget in budgets
    ]
    return results, offload_costs, panels


def _metric_key(budget: float, condition: str, local: float, policy: str) -> str:
    slug = policy.replace(" (", "_").replace(")", "").replace("-", "_")
    return f"on_time_b{budget:g}_{condition}_l{local:g}_{slug}"


def tracked_metrics(results: dict[float, dict[str, list[CellResult]]]) -> dict[str, float]:
    """Each marked cell's on-time accuracy per policy, keyed by budget, condition and local cost."""
    out: dict[str, float] = {}
    for budget, by_condition in results.items():
        for condition, cells in by_condition.items():
            for r in cells:
                for name, m in _policy_table(r):
                    key = _metric_key(budget, condition, r.cell.local_ms, name)
                    out[key] = m.on_time_accuracy
    return {k: v for k, v in out.items() if math.isfinite(v)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Birds cascade under a hard latency budget, from stored predictions."
    )
    parser.add_argument(
        "--server-compute-ms",
        type=float,
        required=True,
        help="The large model's measured per-photo latency: its mean_latency_ms as logged by "
        "birds.evaluation.evaluate (a desktop GPU standing in for a server one). No default.",
    )
    parser.add_argument("--payload-kb", type=float, default=PAYLOAD_KB, help="Request payload.")
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGETS_MS),
        help="Budgets in ms (default: %(default)s).",
    )
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE), help="Boundary figure path.")
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    run_on_engine(
        get_engine(),
        budgets=args.budgets,
        server_compute_ms=args.server_compute_ms,
        payload_kb=args.payload_kb,
        figure_path=args.figure,
    )


def run_on_engine(
    engine: Engine,
    *,
    budgets: Sequence[float],
    server_compute_ms: float,
    payload_kb: float,
    figure_path: str | Path,
) -> None:
    """Load both splits, print the report, write the figure and record one tracking run. Both
    splits are loaded (and budgets validated) before anything is written."""
    validation = load_records(engine, VALIDATION_SPLIT)
    test = load_records(engine, TEST_SPLIT)

    results, offload_costs, panels = run_experiment(
        validation,
        test,
        budgets=budgets,
        server_compute_ms=server_compute_ms,
        payload_kb=payload_kb,
    )
    print(format_report(results, offload_costs))
    plot_boundary_maps(panels, BoundaryMarks(condition_offload_ms=offload_costs), figure_path)
    print(f"\nWrote {figure_path}")

    params: dict[str, str | float | int] = {
        "dataset": DATASET_NAME,
        "validation_photos": len(validation.photo_ids),
        "test_photos": len(test.photo_ids),
        "server_compute_ms": server_compute_ms,
        "payload_kb": payload_kb,
        "budgets_ms": ",".join(f"{b:g}" for b in budgets),
    }
    with tracking.run("budget-experiment", params, experiment=EXPERIMENT_NAME) as handle:
        tracking.log_metrics(handle, tracked_metrics(results))


if __name__ == "__main__":
    main()
