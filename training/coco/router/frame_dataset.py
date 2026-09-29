"""Loads the per-frame feature table, the simulated-frame set, and the
simulated rows joined with per-frame features — the data the feature-router
experiment needs beyond `coco.router.dataset.load_training_data`.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

from coco.router.dataset import load_training_data

IMAGE_STAT_COLUMNS = ["sharpness", "brightness", "contrast", "colorfulness", "entropy"]

# The six image-side columns: scene complexity plus the five image statistics.
IMAGE_COLUMNS = ["scene_complexity", *IMAGE_STAT_COLUMNS]

# The six confidence columns, from the local model's own detections.
CONFIDENCE_COLUMNS = [
    "detection_count",
    "max_confidence",
    "mean_confidence",
    "min_confidence",
    "mean_box_area",
    "min_box_area",
]

# The 11 `frame_features` columns (image statistics + confidence), excluding
# `scene_complexity`, which lives in `scene_complexity` and is already
# present on the simulated rows via `load_training_data`.
_FRAME_FEATURE_COLUMNS = [*IMAGE_STAT_COLUMNS, *CONFIDENCE_COLUMNS]

_FRAME_TABLE_QUERY = text(
    f"""
    SELECT
        ff.file_name AS file_name,
        sc.scene_complexity AS scene_complexity,
        {", ".join(f"ff.{col}" for col in _FRAME_FEATURE_COLUMNS)},
        local_mi.accuracy AS local_accuracy,
        offload_mi.accuracy AS offload_accuracy,
        offload_mi.accuracy - local_mi.accuracy AS gap
    FROM frame_features ff
    JOIN scene_complexity sc
        ON sc.dataset = ff.dataset AND sc.file_name = ff.file_name
    JOIN model_inference local_mi
        ON local_mi.dataset = ff.dataset AND local_mi.file_name = ff.file_name
        AND local_mi.model_path = 'LOCAL'
    JOIN model_inference offload_mi
        ON offload_mi.dataset = ff.dataset AND offload_mi.file_name = ff.file_name
        AND offload_mi.model_path = 'OFFLOAD'
    WHERE ff.dataset = :dataset
    ORDER BY ff.file_name
    """
)

_SIMULATED_FRAME_IDS_QUERY = text(
    """
    SELECT DISTINCT sr.frame_id
    FROM simulation_results sr
    JOIN simulation_runs run ON run.run_id = sr.run_id
    WHERE run.dataset = :dataset
    """
)


def load_frame_table(engine: Engine, dataset: str) -> pd.DataFrame:
    """One row per frame with a `frame_features` row: features, scene complexity,
    both models' accuracies, and their gap. Frames missing any of
    `frame_features`/`scene_complexity`/`model_inference` (either path) are
    excluded by the inner joins. Sorted by `file_name`.
    """
    return pd.read_sql(_FRAME_TABLE_QUERY, engine, params={"dataset": dataset})


def load_simulated_frame_ids(engine: Engine, dataset: str) -> set[str]:
    """File names that appear in `simulation_results` (as `frame_id`) for `dataset`."""
    df = pd.read_sql(_SIMULATED_FRAME_IDS_QUERY, engine, params={"dataset": dataset})
    return set(df["frame_id"])


def load_simulated_rows(engine: Engine, dataset: str) -> pd.DataFrame:
    """`coco.router.dataset.load_training_data`'s rows, joined with the per-frame
    table's feature columns on `frame_id = file_name`. `scene_complexity` is
    left as `load_training_data` already provides it, so it isn't duplicated.
    """
    simulated = load_training_data(engine, dataset)
    frame_table = load_frame_table(engine, dataset)
    features = frame_table[["file_name", *_FRAME_FEATURE_COLUMNS]]
    merged = simulated.merge(features, left_on="frame_id", right_on="file_name", how="inner")
    return merged.drop(columns="file_name")
