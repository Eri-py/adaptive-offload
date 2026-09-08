"""Stratified frame sampling by scene-complexity score.

Pure function: given a `{file_name: complexity_score}` mapping (as returned
by `persistence.get_known_complexity`), selects a target-sized sample by
splitting the score-sorted pool into equal-frequency quantile buckets and
drawing an even, seeded share from each — so the sample spans the full
complexity range instead of clustering wherever the pool happens to be
densest. No I/O, no database access; config values (frame count, bucket
count, seed) are passed in by the caller rather than imported from
`datagen.config`.
"""

from __future__ import annotations

import numpy as np


def stratified_sample(
    scores: dict[str, float],
    target_count: int,
    bucket_count: int,
    seed: int,
) -> list[str]:
    """Select `target_count` file names from `scores` via quantile bucketing.

    Sorts `(file_name, score)` pairs by score and splits them into
    `bucket_count` equal-frequency buckets (`numpy.array_split` on the
    sorted pool), then draws an even share of `target_count` from each
    bucket — any remainder from uneven division is distributed one-per-
    bucket across the first buckets — using a `numpy.random.default_rng
    (seed)` generator, so the same mapping and seed always yield the same
    selection.

    If a bucket has fewer items than its target share, every item in that
    bucket is taken instead of raising; the returned list may then be
    shorter than `target_count` (this only happens if the pool is small
    relative to the target, not for the real ~5,000-image COCO pool).
    """
    sorted_file_names = [
        file_name for file_name, _ in sorted(scores.items(), key=lambda item: item[1])
    ]
    buckets = np.array_split(np.array(sorted_file_names, dtype=object), bucket_count)

    base_share, remainder = divmod(target_count, bucket_count)
    rng = np.random.default_rng(seed)

    selected: list[str] = []
    for index, bucket in enumerate(buckets):
        share = base_share + (1 if index < remainder else 0)
        take = min(share, len(bucket))
        if take == 0:
            continue
        chosen = rng.choice(bucket, size=take, replace=False)
        selected.extend(str(file_name) for file_name in chosen)

    return selected
