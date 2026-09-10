"""Standalone stratified-sample preview, reading already-persisted complexity scores.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.cli.preview_sample --dataset coco_val2017 --frame-count 500 \\
        --bucket-count 5 --seed 42

Composes `persistence.get_known_complexity` (a read) with
`sampling.stratified_sample` (pure, no I/O) to report which frames a
stratified sample would select without running the full simulator
pipeline. Read-only: never calls any `persistence` write function, never
runs condition sampling, stub inference, or labeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.db import get_engine
from dotenv import load_dotenv
from sqlalchemy import Engine

from datagen import config
from datagen.persistence import get_known_complexity
from datagen.sampling import stratified_sample


def preview_sample(
    engine: Engine,
    dataset: str,
    frame_count: int,
    bucket_count: int,
    seed: int,
) -> list[tuple[str, float]]:
    """Report which frames a stratified sample of `dataset` would select.

    Reads already-persisted `scene_complexity` rows for `dataset` via
    `persistence.get_known_complexity`, then feeds them straight into
    `sampling.stratified_sample` with the given `frame_count`/`bucket_count`/
    `seed`. Returns `(file_name, complexity_score)` pairs, in the same order
    `stratified_sample` selected them, so a caller (or `main` below) has the
    score for context without a second lookup.

    Raises `ValueError` if `dataset` has no persisted complexity scores yet
    — sampling from an empty pool would otherwise silently return an empty
    list, which reads as "zero frames selected" rather than "no scores exist
    to sample from."
    """
    known_complexity = get_known_complexity(engine, dataset)
    if not known_complexity:
        raise ValueError(
            f"No complexity scores found for dataset {dataset!r}. "
            "Run the complexity-scoring step (or the full simulator pipeline) first."
        )
    selected = stratified_sample(known_complexity, frame_count, bucket_count, seed)
    return [(file_name, known_complexity[file_name]) for file_name in selected]


def main() -> None:
    # CLI-only convenience: load DATABASE_URL from training/.env if it isn't
    # already in the environment (never overrides an explicit `export`).
    # The core function stays free of this side effect.
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Preview which frames a stratified sample would select."
    )
    parser.add_argument(
        "--dataset",
        default=config.DATASET_NAME,
        help=f"Dataset name to read complexity scores for (default: {config.DATASET_NAME}).",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=config.FRAME_COUNT,
        help=f"Target number of frames to sample (default: {config.FRAME_COUNT}).",
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=config.STRATIFICATION_BUCKET_COUNT,
        help=(
            "Number of complexity-based strata to sample evenly across "
            f"(default: {config.STRATIFICATION_BUCKET_COUNT})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help=f"Random seed for sampling (default: {config.SEED}).",
    )
    args = parser.parse_args()

    engine = get_engine()
    selected = preview_sample(
        engine, args.dataset, args.frame_count, args.bucket_count, args.seed
    )

    for file_name, score in selected:
        print(f"{file_name}\t{score:.4f}")


if __name__ == "__main__":
    main()
