"""Evaluates the birds cascade under a hard latency budget, one cell at a time.

A cell names a local cost, an offload cost and a budget. For each cell the cascade's
threshold is tuned on the validation records alone, then applied to the test records, and all
five policies are scored. Everything is computed from stored predictions; no model is loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sqlalchemy import Engine

from birds.config import MODEL_NAMES
from birds.evaluation.records import DATASET_NAME
from birds.evaluation.store import read_predictions
from birds.evaluation.summary import CONFIDENCE_MEASURES, align_models
from birds.router.policies import Band, classify_band, lockout_width_ms, oracle_actions, outcomes
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
    """Both models' stored predictions for `split`, aligned per photo."""
    small, large = (read_predictions(engine, DATASET_NAME, split, m) for m in MODEL_NAMES)
    return records_from_frames(small, large)


def _candidate_thresholds(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Every distinct score (the first escalates nothing), plus one just above the maximum
    (escalates everything), ascending so ties resolve to the lowest threshold."""
    return np.append(np.unique(scores), np.nextafter(scores.max(), np.inf))


def _policy_rows(
    actions: npt.NDArray[np.str_], records: PhotoRecords, cell: Cell
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    latency, correct = outcomes(
        actions,
        local_ms=cell.local_ms,
        offload_ms=cell.offload_ms,
        small_correct=records.small_correct,
        large_correct=records.large_correct,
    )
    return latency, on_time_accuracy_per_row(latency, correct, cell.budget_ms)


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


def tune_threshold(validation: PhotoRecords, measure: str, cell: Cell) -> float:
    """The threshold with the highest on-time accuracy on the validation records, the lowest
    among ties. Takes validation only, so no test photo can influence it."""
    best_threshold, best_score = 0.0, -1.0
    for threshold in _candidate_thresholds(validation.confidence[measure]):
        actions = _cascade_actions(validation, measure, float(threshold), cell)
        _, on_time = _policy_rows(actions, validation, cell)
        if on_time.mean() > best_score:
            best_threshold, best_score = float(threshold), float(on_time.mean())
    return best_threshold


def fewest_escalations_to_match(records: PhotoRecords, measure: str, cell: Cell) -> int | None:
    """Fewest least-confident photos sent to the large model for the cascade to reach
    always-offload's accuracy, or None if it never does. Ties on confidence fall back to the
    numeric photo id (records are in that order and the sort is stable). Where escalation does
    not fit the budget no photo can be escalated, so only zero escalations are possible."""
    small, large = records.small_correct, records.large_correct
    order = np.argsort(records.confidence[measure], kind="stable")
    gained = np.concatenate([[0.0], np.cumsum(large[order] - small[order])])
    if cell.local_ms + cell.offload_ms > cell.budget_ms:
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
    latency, _ = _policy_rows(actions, records, cell)
    correct = np.where(actions == LOCAL, records.small_correct, records.large_correct)
    return metrics(latency, correct, cell.budget_ms)


def evaluate_cell(validation: PhotoRecords, test: PhotoRecords, cell: Cell) -> CellResult:
    """Score all five policies on `test`, with the cascade threshold tuned on `validation`."""
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
    _, local_on_time = _policy_rows(local, test, cell)
    _, budget_only_on_time = _policy_rows(budget_only, test, cell)

    cascades = {}
    for measure in CONFIDENCE_MEASURES:
        threshold = tune_threshold(validation, measure, cell)
        actions = _cascade_actions(test, measure, threshold, cell)
        _, on_time = _policy_rows(actions, test, cell)
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
