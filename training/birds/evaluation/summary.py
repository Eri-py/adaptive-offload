"""Summary tables over the stored test-split predictions: agreement, confidence separation, cascade.

Run as `python -m birds.evaluation.summary`.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from common.db import get_engine
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sqlalchemy import Engine

from birds.config import EXPERIMENT_NAME, ModelName
from birds.evaluation.predictions import DATASET_NAME, ENV_PATH
from birds.evaluation.store import read_predictions
from shared import tracking

SPLIT = "test"
CONFIDENCE_MEASURES = ("max_probability", "top2_margin")
# Share of photos (by percent) sent to the large model in the threshold sweep.
ESCALATION_PERCENTS = (0, 5, 10, 15, 20, 30, 50, 100)
_MODELS: tuple[ModelName, ...] = ("small", "large")


@dataclass(frozen=True)
class Agreement:
    """Accuracies and the four right/wrong combinations, as shares of all photos."""

    small_accuracy: float
    large_accuracy: float
    gap: float
    both_right: float
    only_large_right: float
    only_small_right: float
    both_wrong: float
    oracle: float


@dataclass(frozen=True)
class CascadeRow:
    """One threshold-sweep point: the least-confident `percent` of photos go to the large model."""

    percent: int
    escalated: int
    accuracy: float
    gap_recovered: float


@dataclass(frozen=True)
class Summary:
    """Every reported figure; `separation` is AUC keyed by (model, measure)."""

    photos: int
    agreement: Agreement
    separation: dict[tuple[ModelName, str], float]
    cascade: dict[str, list[CascadeRow]]


def align_models(small: pd.DataFrame, large: pd.DataFrame) -> pd.DataFrame:
    """One row per photo with `<column>_small`/`<column>_large`, ordered by numeric photo id."""
    if set(small["photo_id"]) != set(large["photo_id"]):
        raise ValueError("small and large models cover different photos; re-record both")
    if small["photo_id"].duplicated().any() or large["photo_id"].duplicated().any():
        raise ValueError("duplicate photo ids in stored predictions")
    joined = small.merge(large, on="photo_id", suffixes=("_small", "_large"))
    return joined.sort_values("photo_id", key=lambda ids: ids.astype(int), ignore_index=True)


def _agreement(joined: pd.DataFrame) -> Agreement:
    small_ok = joined["is_correct_small"].to_numpy(dtype=bool)
    large_ok = joined["is_correct_large"].to_numpy(dtype=bool)
    small_acc = float(small_ok.mean())
    large_acc = float(large_ok.mean())
    return Agreement(
        small_accuracy=small_acc,
        large_accuracy=large_acc,
        gap=large_acc - small_acc,
        both_right=float((small_ok & large_ok).mean()),
        only_large_right=float((~small_ok & large_ok).mean()),
        only_small_right=float((small_ok & ~large_ok).mean()),
        both_wrong=float((~small_ok & ~large_ok).mean()),
        oracle=float((small_ok | large_ok).mean()),
    )


def _separation(joined: pd.DataFrame, model: ModelName, measure: str) -> float:
    """AUC of the confidence for telling correct from incorrect; NaN if only one outcome occurs."""
    correct = joined[f"is_correct_{model}"].astype(bool)
    if correct.nunique() < 2:
        return math.nan
    return float(roc_auc_score(correct, joined[f"{measure}_{model}"]))


def _cascade(joined: pd.DataFrame, measure: str, agreement: Agreement) -> list[CascadeRow]:
    """Threshold sweep: escalate the least-confident small-model photos to the large model."""
    # Ties on confidence fall back to the numeric photo id, so the order never varies.
    ordered = joined.assign(_id=joined["photo_id"].astype(int)).sort_values(
        [f"{measure}_small", "_id"], kind="stable", ignore_index=True
    )
    small_ok = ordered["is_correct_small"].to_numpy(dtype=bool)
    large_ok = ordered["is_correct_large"].to_numpy(dtype=bool)
    total = len(ordered)
    rows = []
    for percent in ESCALATION_PERCENTS:
        escalated = total * percent // 100
        correct = np.concatenate([large_ok[:escalated], small_ok[escalated:]])
        accuracy = float(correct.mean())
        # Not clamped: a cascade that beats the large model alone legitimately exceeds 1.
        recovered = (
            (accuracy - agreement.small_accuracy) / agreement.gap if agreement.gap else math.nan
        )
        rows.append(CascadeRow(percent, escalated, accuracy, recovered))
    return rows


def summarize(small: pd.DataFrame, large: pd.DataFrame) -> Summary:
    """Compute every table from the two models' stored rows for one split."""
    joined = align_models(small, large)
    if joined.empty:
        raise ValueError("no stored predictions to summarize")
    agreement = _agreement(joined)
    return Summary(
        photos=len(joined),
        agreement=agreement,
        separation={
            (model, measure): _separation(joined, model, measure)
            for model in _MODELS
            for measure in CONFIDENCE_MEASURES
        },
        cascade={measure: _cascade(joined, measure, agreement) for measure in CONFIDENCE_MEASURES},
    )


def load_summary(engine: Engine) -> Summary:
    """Summarize the test split as currently stored for both models."""
    small, large = (read_predictions(engine, DATASET_NAME, SPLIT, model) for model in _MODELS)
    return summarize(small, large)


def format_summary(summary: Summary) -> str:
    """The tables as text."""
    a = summary.agreement
    lines = [
        f"Test split: {summary.photos} photos",
        "",
        "Agreement",
        f"  small accuracy    {a.small_accuracy:.4f}",
        f"  large accuracy    {a.large_accuracy:.4f}",
        f"  gap               {a.gap:.4f}",
        f"  both right        {a.both_right:.4f}",
        f"  only large right  {a.only_large_right:.4f}",
        f"  only small right  {a.only_small_right:.4f}",
        f"  both wrong        {a.both_wrong:.4f}",
        f"  oracle            {a.oracle:.4f}",
        "",
        "Confidence separation (AUC, correct vs incorrect)",
        f"  {'model':<8}" + "".join(f"{m:>18}" for m in CONFIDENCE_MEASURES),
    ]
    for model in _MODELS:
        cells = "".join(f"{summary.separation[(model, m)]:>18.4f}" for m in CONFIDENCE_MEASURES)
        lines.append(f"  {model:<8}{cells}")
    for measure in CONFIDENCE_MEASURES:
        lines += [
            "",
            f"Cascade by small-model {measure}",
            f"  {'escalate %':>10}{'photos':>8}{'accuracy':>10}{'gap recovered':>15}",
        ]
        lines += [
            f"  {r.percent:>10}{r.escalated:>8}{r.accuracy:>10.4f}{r.gap_recovered:>15.4f}"
            for r in summary.cascade[measure]
        ]
    return "\n".join(lines)


def _metrics(summary: Summary) -> dict[str, float]:
    a = summary.agreement
    metrics = {
        "small_accuracy": a.small_accuracy,
        "large_accuracy": a.large_accuracy,
        "gap": a.gap,
        "both_right": a.both_right,
        "only_large_right": a.only_large_right,
        "only_small_right": a.only_small_right,
        "both_wrong": a.both_wrong,
        "oracle": a.oracle,
    }
    for (model, measure), auc in summary.separation.items():
        metrics[f"auc_{model}_{measure}"] = auc
    for measure, rows in summary.cascade.items():
        for row in rows:
            metrics[f"cascade_{measure}_{row.percent}_accuracy"] = row.accuracy
            metrics[f"cascade_{measure}_{row.percent}_gap_recovered"] = row.gap_recovered
    return {key: value for key, value in metrics.items() if math.isfinite(value)}


def main() -> None:
    """Print the summary tables and log them as one standalone MLflow run."""
    load_dotenv(ENV_PATH)
    summary = load_summary(get_engine())
    print(format_summary(summary))
    with tracking.run(
        "prediction-summary",
        {"dataset": DATASET_NAME, "split": SPLIT, "photos": summary.photos},
        experiment=EXPERIMENT_NAME,
    ) as handle:
        tracking.log_metrics(handle, _metrics(summary))


if __name__ == "__main__":
    main()
