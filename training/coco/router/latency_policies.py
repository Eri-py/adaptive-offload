"""Attempt 1's COCO-shaped latency-budget pieces: how a simulated row's
latency and accuracy follow from an action, the within-budget oracle over
those rows, the offload-latency predictor, and the two cascade confidence
scores.

The action labels and the shape-agnostic policies and metrics live in
`shared.routing`, since attempt 2 uses them too. What stays here depends on
the shape of attempt 1's data — see
`coco.router.frame_dataset.load_simulated_rows` for the row columns
(`local_latency_ms`, `local_accuracy`, `offload_latency_ms`,
`offload_accuracy`). No inference is re-run; a policy only returns each row's
action, and outcomes follow mechanically from that action.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from coco.router.frame_dataset import CONFIDENCE_COLUMNS
from coco.router.two_stage import Stage1Models, predict_stage1, train_stage1
from shared.routing import ACTION_DTYPE, ESCALATE, LOCAL, OFFLOAD

# The router decides using predicted conditions only — it never sees the true latency in advance.
_OFFLOAD_LATENCY_FEATURE_COLUMNS = [
    "network_bandwidth_mbps",
    "network_latency_ms",
    "network_packet_loss_pct",
    "device_load_pct",
]


def train_offload_latency_model(train_rows: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Fit a `HistGradientBoostingRegressor(random_state=42)` predicting
    `offload_latency_ms` from `_OFFLOAD_LATENCY_FEATURE_COLUMNS` alone.
    Gradient-boosted because the simulated offload latency scales with
    1/bandwidth, which a linear model misfits.

    Unlike stage 2's GBT (`two_stage.STAGE2_MIN_LEAF_PHOTOS`), this model's
    inputs are only condition columns, never a per-photo stage-1 output —
    there's nothing here a leaf could split on to isolate one training
    photo, so the photo-memorisation guard that fix needed doesn't apply.
    """
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(train_rows[_OFFLOAD_LATENCY_FEATURE_COLUMNS], train_rows["offload_latency_ms"])
    return model


def predict_offload_latency(
    model: HistGradientBoostingRegressor, rows: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Predicted offload latency (ms) for each row of `rows`."""
    return np.asarray(model.predict(rows[_OFFLOAD_LATENCY_FEATURE_COLUMNS]), dtype=np.float64)


def raw_score(rows: pd.DataFrame) -> npt.NDArray[np.float64]:
    """The raw cascade confidence score: YOLOv8n's stored mean detection
    confidence for the frame (0 when there are no detections, so those
    frames always escalate if the budget allows)."""
    return np.asarray(rows["mean_confidence"], dtype=np.float64)


def learned_score_model(
    frame_table: pd.DataFrame, simulated_frame_ids: set[str]
) -> Stage1Models:
    """Fit the learned cascade score: stage 1's gap classifier
    (`coco.router.two_stage.train_stage1`) on `CONFIDENCE_COLUMNS`, family
    `"gbt"`, trained on every frame in `frame_table` except the simulated
    ones — it never sees a simulated photo, train or test
    (`Stage1Models.trained_frame_ids` is what a test checks this against).
    """
    return train_stage1(frame_table, CONFIDENCE_COLUMNS, "gbt", simulated_frame_ids)


def learned_score(model: Stage1Models, rows: pd.DataFrame) -> npt.NDArray[np.float64]:
    """The learned cascade confidence score for each row of `rows`: the
    predicted probability that the local result is at least as accurate as
    the offload result (`p_local_good_enough`, from
    `coco.router.two_stage.predict_stage1`)."""
    predicted = predict_stage1(model, rows)
    return np.asarray(predicted["p_local_good_enough"], dtype=np.float64)


def oracle_actions(rows: pd.DataFrame, budget_ms: float) -> npt.NDArray[np.str_]:
    """Among LOCAL/OFFLOAD/ESCALATE's *true* latencies, the option whose
    latency fits the budget with the highest accuracy; ties broken by lower
    latency, then by the order LOCAL, OFFLOAD, ESCALATE. Falls back to LOCAL
    when nothing fits.
    """
    local_latency = np.asarray(rows["local_latency_ms"], dtype=np.float64)
    local_accuracy = np.asarray(rows["local_accuracy"], dtype=np.float64)
    offload_latency = np.asarray(rows["offload_latency_ms"], dtype=np.float64)
    offload_accuracy = np.asarray(rows["offload_accuracy"], dtype=np.float64)
    escalate_latency = local_latency + offload_latency
    escalate_accuracy = offload_accuracy

    # (label, latency, accuracy) in tie-break order — LOCAL, OFFLOAD, ESCALATE.
    options = (
        (LOCAL, local_latency, local_accuracy),
        (OFFLOAD, offload_latency, offload_accuracy),
        (ESCALATE, escalate_latency, escalate_accuracy),
    )

    n = local_latency.shape[0]
    actions = np.full(n, LOCAL, dtype=ACTION_DTYPE)
    best_accuracy = np.full(n, -np.inf)
    best_latency = np.full(n, np.inf)

    for label, latency, accuracy in options:
        fits = latency <= budget_ms
        higher_accuracy = accuracy > best_accuracy
        tied_but_faster = (accuracy == best_accuracy) & (latency < best_latency)
        # Strict improvement only, so among ties the earlier option in
        # `options` (the required LOCAL/OFFLOAD/ESCALATE order) keeps it.
        better = fits & (higher_accuracy | tied_but_faster)
        actions[better] = label
        best_accuracy = np.where(better, accuracy, best_accuracy)
        best_latency = np.where(better, latency, best_latency)

    # Rows where nothing fits never trigger `better`, so `actions` keeps its
    # LOCAL default — that is the fallback the spec asks for.
    return actions


def outcomes(
    rows: pd.DataFrame, actions: npt.NDArray[np.str_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-row (latency_ms, accuracy) implied by `actions`: LOCAL -> local;
    OFFLOAD -> offload; ESCALATE -> local + offload latency, offload
    accuracy."""
    local_latency = np.asarray(rows["local_latency_ms"], dtype=np.float64)
    local_accuracy = np.asarray(rows["local_accuracy"], dtype=np.float64)
    offload_latency = np.asarray(rows["offload_latency_ms"], dtype=np.float64)
    offload_accuracy = np.asarray(rows["offload_accuracy"], dtype=np.float64)

    conditions = [actions == LOCAL, actions == OFFLOAD, actions == ESCALATE]
    latency_ms = np.select(
        conditions, [local_latency, offload_latency, local_latency + offload_latency]
    )
    accuracy = np.select(conditions, [local_accuracy, offload_accuracy, offload_accuracy])
    return latency_ms.astype(np.float64), accuracy.astype(np.float64)

