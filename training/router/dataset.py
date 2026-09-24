"""Loads simulation_results (+ scene_complexity) into a training DataFrame,
and splits it by photo (frame_id), not by row, so no photo appears in both
train and test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sqlalchemy import Engine, text

FEATURE_COLUMNS = [
    "network_bandwidth_mbps",
    "network_latency_ms",
    "network_packet_loss_pct",
    "device_load_pct",
    "scene_complexity",
]

_QUERY = text(
    """
    SELECT
        sr.frame_id,
        run.preset_name,
        sr.network_bandwidth_mbps,
        sr.network_latency_ms,
        sr.network_packet_loss_pct,
        sr.device_load_pct,
        sc.scene_complexity,
        sr.local_latency_ms,
        sr.local_accuracy,
        sr.offload_latency_ms,
        sr.offload_accuracy,
        sr.label
    FROM simulation_results sr
    JOIN simulation_runs run ON sr.run_id = run.run_id
    JOIN scene_complexity sc ON sc.dataset = run.dataset AND sc.file_name = sr.frame_id
    WHERE run.dataset = :dataset
    """
)


def load_training_data(engine: Engine, dataset: str) -> pd.DataFrame:
    """Load every simulation_results row for `dataset`, joined with its scene_complexity score."""
    return pd.read_sql(_QUERY, engine, params={"dataset": dataset})


def frame_level_split(
    df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (train, test) by `frame_id` — a photo's rows never cross the split."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["frame_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
