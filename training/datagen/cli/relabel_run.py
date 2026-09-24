"""Standalone re-labeling CLI, reading an existing run's already-persisted results.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.cli.relabel_run --run-id <run-id> --lambda 0.5

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
from typing import NamedTuple

from common.db import get_engine
from common.models import Label
from dotenv import load_dotenv
from sqlalchemy import Engine

from datagen.persistence import get_run_results
from datagen.simulate import labeling


class RelabeledRow(NamedTuple):
    """One row's stored vs. recomputed label, plus the condition vector that produced it.

    Carrying the four condition values (not just `frame_id`) is what makes
    the report useful at real scale: a run has `condition_vector_count` rows
    per `frame_id` with nothing else distinguishing them, so without these a
    reader who sees a flip can't tell *which* condition flipped — which is
    the whole point of a λ sweep (per the spec's routing-boundary-shift
    question).
    """

    frame_id: str
    network_bandwidth_mbps: float
    network_latency_ms: float
    network_packet_loss_pct: float
    device_load_pct: float
    stored_label: Label
    recomputed_label: Label


def relabel_run(engine: Engine, run_id: str, lambda_value: float) -> list[RelabeledRow]:
    """Recompute each of `run_id`'s stored results' labels under `lambda_value`.

    Returns one `RelabeledRow` per persisted result, ordered by
    `(frame_id, id)` (per `get_run_results`) — a stable order across calls,
    including among the `condition_vector_count` rows that share a
    `frame_id`. Raises `ValueError` if `run_id` has no persisted results —
    either it doesn't exist or its run produced zero rows, and either way
    there's nothing to relabel; reporting an empty list silently would read
    as "zero rows flipped" rather than "no such run."
    """
    results = get_run_results(engine, run_id)
    if not results:
        raise ValueError(
            f"No results found for run {run_id!r}. Check the run id, or that the run "
            "actually produced result rows."
        )
    return [
        RelabeledRow(
            frame_id=row.frame_id,
            network_bandwidth_mbps=row.network_bandwidth_mbps,
            network_latency_ms=row.network_latency_ms,
            network_packet_loss_pct=row.network_packet_loss_pct,
            device_load_pct=row.device_load_pct,
            stored_label=row.label,
            recomputed_label=labeling.compute_label(
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
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

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
    for row in relabeled:
        flipped = row.stored_label != row.recomputed_label
        flip_count += flipped
        marker = " (flipped)" if flipped else ""
        print(
            f"{row.frame_id}\t"
            f"bw={row.network_bandwidth_mbps}\t"
            f"lat={row.network_latency_ms}\t"
            f"loss={row.network_packet_loss_pct}\t"
            f"load={row.device_load_pct}\t"
            f"{row.stored_label.value}\t{row.recomputed_label.value}{marker}"
        )

    print(f"\n{flip_count}/{len(relabeled)} rows flipped.")


if __name__ == "__main__":
    main()
