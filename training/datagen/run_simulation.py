"""Standalone data-gen simulator script — wires Tasks 3-10 together.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention — `training/`
itself is the import root, the same way `server/` is for `common`):

    python -m datagen.run_simulation --preset baseline

Builds the real Postgres engine from `DATABASE_URL`, downloads/scores any
COCO val2017 pool images not yet covered by the `scene_complexity` table,
stratified-samples frames, space-fills condition vectors for the selected
preset, crosses every frame with every condition vector (stub inference +
win/loss label), and persists one `simulation_runs` row plus one
`simulation_results` row per (frame, condition) pair.

The CLI (`main`) is a thin wrapper around `run_simulation`, the directly
callable core function — every real dependency (`image_records`,
`resolve_image`, the run-shape tunables) is injectable so tests can swap in
a small fake image pool instead of the real ~5,000-image COCO set.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from common.db import get_engine
from dotenv import load_dotenv
from sqlalchemy import Engine

from datagen import coco, config
from datagen.coco import ImageRecord
from datagen.complexity import scene_complexity
from datagen.conditions import sample_condition_vectors
from datagen.labeling import compute_label
from datagen.persistence import (
    ResultRow,
    RunConfig,
    create_run,
    get_known_complexity,
    store_complexity_scores,
    store_results,
)
from datagen.sampling import stratified_sample
from datagen.stub_inference import stub_inference

logger = logging.getLogger(__name__)

# How many newly-scored images to accumulate before flushing to Postgres in
# the complexity-scoring loop below. This is a crash-resilience/robustness
# knob, not a research-relevant tunable (it never changes what gets computed
# or persisted, only how often) — kept local here rather than in
# `config.py`, whose tunables all affect the simulation's actual behavior/
# output. 200 keeps a worst-case loss (a crash right before a flush) to a
# small fraction of the ~5,000-image val2017 pool while still batching most
# of the network/DB round-trip savings a straight per-image commit would
# give up.
COMPLEXITY_SCORE_FLUSH_BATCH_SIZE = 200


def run_simulation(
    engine: Engine,
    preset_name: str,
    *,
    image_records: list[ImageRecord] | None = None,
    resolve_image: Callable[[str], Path] | None = None,
    frame_count: int | None = None,
    condition_vector_count: int | None = None,
    bucket_count: int | None = None,
    seed: int | None = None,
    lambda_value: float | None = None,
) -> str:
    """Run one full simulator invocation for `preset_name`, return its run id.

    `image_records` and `resolve_image` default to the real COCO val2017 pool
    (`coco.load_image_index()` / `coco.resolve_image_path`) when omitted —
    tests inject a small fake pool and a resolver pointed at synthetic
    temp-directory images instead. `frame_count`/`condition_vector_count`/
    `bucket_count`/`seed`/`lambda_value` default to `datagen.config`'s
    tunables when omitted, so a reduced-scale test run doesn't have to
    exercise the real 500 x 50 defaults against a tiny fake pool.
    """
    if preset_name not in config.PRESETS:
        raise ValueError(
            f"Unknown preset {preset_name!r}. Valid presets: {sorted(config.PRESETS)}"
        )
    preset = config.PRESETS[preset_name]

    resolved_image_records = (
        image_records if image_records is not None else coco.load_image_index()
    )
    resolved_resolve_image = (
        resolve_image if resolve_image is not None else coco.resolve_image_path
    )
    resolved_frame_count = frame_count if frame_count is not None else config.FRAME_COUNT
    resolved_condition_vector_count = (
        condition_vector_count
        if condition_vector_count is not None
        else config.CONDITION_VECTOR_COUNT
    )
    resolved_bucket_count = (
        bucket_count if bucket_count is not None else config.STRATIFICATION_BUCKET_COUNT
    )
    resolved_seed = seed if seed is not None else config.SEED
    resolved_lambda = lambda_value if lambda_value is not None else config.DEFAULT_LAMBDA

    known_complexity = get_known_complexity(engine, config.DATASET_NAME)

    # Flushed in batches (not once at the end) so a network error, corrupt
    # file, or interrupted run loses at most one batch's worth of scoring
    # work instead of the whole pool — `store_complexity_scores` re-checks
    # already-persisted file names on every call, so a resumed run picks up
    # exactly where it left off rather than double-inserting.
    all_complexity = dict(known_complexity)
    pending_batch: dict[str, float] = {}
    # Total images this invocation actually needs to score (excludes ones
    # already persisted from a prior run) -- the denominator for the
    # progress log below, so resumed runs report progress against the
    # remaining work, not the full pool size.
    total_to_score = len(resolved_image_records) - len(known_complexity)
    scored_count = 0
    for record in resolved_image_records:
        if record.file_name in known_complexity:
            continue
        image_path = resolved_resolve_image(record.file_name)
        score = scene_complexity(image_path)
        all_complexity[record.file_name] = score
        pending_batch[record.file_name] = score
        scored_count += 1
        if len(pending_batch) >= COMPLEXITY_SCORE_FLUSH_BATCH_SIZE:
            store_complexity_scores(engine, config.DATASET_NAME, pending_batch)
            pending_batch = {}
            logger.info("Scored %d/%d images.", scored_count, total_to_score)
    if pending_batch:
        store_complexity_scores(engine, config.DATASET_NAME, pending_batch)
        logger.info("Scored %d/%d images.", scored_count, total_to_score)

    sampled_frames = stratified_sample(
        all_complexity, resolved_frame_count, resolved_bucket_count, resolved_seed
    )
    if len(sampled_frames) < resolved_frame_count:
        logger.warning(
            "Stratified sample came back short: got %d frames, requested %d. "
            "The image pool is too small (relative to frame_count/bucket_count) "
            "to fill every bucket's share; check for a misconfiguration.",
            len(sampled_frames),
            resolved_frame_count,
        )
    condition_vectors = sample_condition_vectors(
        preset, resolved_condition_vector_count, resolved_seed
    )

    # `preset.items()` types values as `object` under mypy (TypedDict doesn't
    # statically guarantee homogeneous value types), so build the dict from
    # the known literal keys directly rather than iterating `.items()`.
    condition_ranges: dict[str, list[float]] = {
        "bandwidth_mbps": list(preset["bandwidth_mbps"]),
        "network_latency_ms": list(preset["network_latency_ms"]),
        "packet_loss_pct": list(preset["packet_loss_pct"]),
        "device_load_pct": list(preset["device_load_pct"]),
    }
    run_config = RunConfig(
        dataset=config.DATASET_NAME,
        preset_name=preset_name,
        frame_count=len(sampled_frames),
        condition_vector_count=len(condition_vectors),
        condition_ranges=condition_ranges,
        seed=resolved_seed,
        lambda_value=resolved_lambda,
    )
    run_id = create_run(engine, run_config)

    rows: list[ResultRow] = []
    for frame_index, frame_id in enumerate(sampled_frames):
        frame_complexity = all_complexity[frame_id]
        for condition_index, condition in enumerate(condition_vectors):
            # A distinct-but-deterministic seed per (frame, condition) pair —
            # stub_inference's own contract only guarantees determinism for a
            # fixed seed, so reusing one seed for every row would make every
            # row draw identical noise. Index-derived offsets from the run's
            # seed keep this reproducible across two runs of the same
            # preset/config/seed/pool without needing per-row random state.
            row_seed = resolved_seed + frame_index * len(condition_vectors) + condition_index
            local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy = stub_inference(
                condition, frame_complexity, row_seed
            )
            label = compute_label(
                local_latency_ms,
                local_accuracy,
                offload_latency_ms,
                offload_accuracy,
                resolved_lambda,
            )
            rows.append(
                ResultRow(
                    frame_id=frame_id,
                    network_bandwidth_mbps=condition[0],
                    network_latency_ms=condition[1],
                    network_packet_loss_pct=condition[2],
                    device_load_pct=condition[3],
                    local_latency_ms=local_latency_ms,
                    local_accuracy=local_accuracy,
                    offload_latency_ms=offload_latency_ms,
                    offload_accuracy=offload_accuracy,
                    label=label,
                )
            )

    store_results(engine, run_id, rows)
    return run_id


def main() -> None:
    # CLI-only convenience: load DATABASE_URL from training/.env if it isn't
    # already in the environment (never overrides an explicit `export`).
    # `run_simulation` itself stays free of this side effect — only the CLI
    # entry point needs it, not the core function or a plain import of this
    # module.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Run the data-gen simulator for one preset.")
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(config.PRESETS),
        help="Named condition-scenario preset to sample condition vectors from.",
    )
    args = parser.parse_args()

    engine = get_engine()
    run_id = run_simulation(engine, args.preset)
    print(f"Created run {run_id}")


if __name__ == "__main__":
    main()
