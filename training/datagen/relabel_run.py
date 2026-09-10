"""Standalone re-labeling CLI, reading an existing run's already-persisted results.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.relabel_run --run-id <run-id> --lambda 0.5

Composes `persistence.get_run_results` (a read) with `labeling.compute_label`
(pure, no I/O) to report which label each row would have under a different
λ, using only that run's already-persisted latency/accuracy values.
Read-only: never calls `persistence.create_run` or `persistence.store_results`
— it never re-runs scene-complexity scoring, frame sampling, condition
sampling, or stub inference, and it never persists the recomputed labels
anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.db import get_engine
from common.models import Label
from dotenv import load_dotenv
from sqlalchemy import Engine

from datagen import labeling
from datagen.persistence import get_run_results


def relabel_run(engine: Engine, run_id: str, lambda_value: float) -> list[tuple[str, Label, Label]]:
    """Recompute each of `run_id`'s stored results' labels under `lambda_value`.

    Returns `(frame_id, stored_label, recomputed_label)` triples, ordered by
    `frame_id` (per `get_run_results`). Raises `ValueError` if `run_id` has
    no persisted results — either it doesn't exist or its run produced zero
    rows, and either way there's nothing to relabel; reporting an empty list
    silently would read as "zero rows flipped" rather than "no such run."
    """
    results = get_run_results(engine, run_id)
    if not results:
        raise ValueError(
            f"No results found for run {run_id!r}. Check the run id, or that the run "
            "actually produced result rows."
        )
    return [
        (
            row.frame_id,
            row.label,
            labeling.compute_label(
                row.local_latency_ms,
                row.local_accuracy,
                row.offload_latency_ms,
                row.offload_accuracy,
                lambda_value,
            ),
        )
        for row in results
    ]


def main() -> None:
    # CLI-only convenience: load DATABASE_URL from training/.env if it isn't
    # already in the environment (never overrides an explicit `export`).
    # The core function stays free of this side effect.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Recompute win/loss labels for an existing run under a different lambda."
    )
    parser.add_argument("--run-id", required=True, help="Run id to re-label results for.")
    parser.add_argument(
        "--lambda",
        dest="lambda_value",
        type=float,
        required=True,
        help="Utility-function weight to recompute labels under.",
    )
    args = parser.parse_args()

    engine = get_engine()
    relabeled = relabel_run(engine, args.run_id, args.lambda_value)

    flip_count = 0
    for frame_id, stored_label, recomputed_label in relabeled:
        flipped = stored_label != recomputed_label
        flip_count += flipped
        marker = " (flipped)" if flipped else ""
        print(f"{frame_id}\t{stored_label.value}\t{recomputed_label.value}{marker}")

    print(f"\n{flip_count}/{len(relabeled)} rows flipped.")


if __name__ == "__main__":
    main()
