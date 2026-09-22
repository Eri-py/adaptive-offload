"""Standalone data-gen simulator script — wires Tasks 3-10 together.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention — `training/`
itself is the import root, the same way `server/` is for `common`):

    run-simulation --preset baseline --annotations <path> --images <folder> \\
        --dataset <name>

`--annotations` is a COCO-format annotations file (JSON with an `"images"`
list plus `"annotations"`/`"categories"` for ground truth), `--images` is
the local folder to resolve those images' file names from, and `--dataset`
names the dataset scene-complexity/model-inference scores get stored under
(so different image pools never share/collide on the same
`scene_complexity`/`model_inference` rows). This tool never fetches
anything over the network — both the annotations file and every image must
already be present at the given paths; it runs the simulation against
exactly what it's given, rather than trying to top up missing data. Builds
the real Postgres engine from `DATABASE_URL`, scores any pool images not
yet covered by the `scene_complexity` table for that dataset, runs real
local/offload YOLO inference (cached per file name in `model_inference`,
IoU-scored against real ground truth) for any pool images not yet fully
covered, stratified-samples frames, space-fills condition vectors for the
selected preset, crosses every frame with every condition vector (cached
real inference plus synthetic condition-driven latency overhead + win/loss
label), and persists one `simulation_runs` row plus one `simulation_results`
row per (frame, condition) pair.

The CLI (`main`) is a thin wrapper around `run_simulation`, the directly
callable core function — every real dependency (`image_records`,
`resolve_image`, `dataset`, the run-shape tunables) is injectable so tests
can swap in a small fake image pool instead of the real ~5,000-image COCO
set.
"""

from __future__ import annotations

import argparse
import functools
import logging
from collections.abc import Callable
from pathlib import Path

from common.db import get_engine
from common.models import Label
from dotenv import load_dotenv
from sqlalchemy import Engine

from datagen import config, presets
from datagen.persistence import (
    ResultRow,
    RunConfig,
    create_run,
    get_known_complexity,
    get_known_model_inference,
    store_complexity_scores,
    store_model_inference,
    store_results,
)
from datagen.sampling.complexity import scene_complexity
from datagen.sampling.conditions import sample_condition_vectors
from datagen.sampling.sampling import stratified_sample
from datagen.simulate import ground_truth, inference
from datagen.simulate.inference import DetectionResult
from datagen.simulate.labeling import compute_label
from datagen.sourcing import image_source
from datagen.sourcing.image_source import ImageRecord

logger = logging.getLogger(__name__)

# How many newly-scored images/newly-computed model-inference results to
# accumulate before flushing to Postgres in the complexity-scoring and
# model-inference loops below. This is a crash-resilience/robustness knob,
# not a research-relevant tunable (it never changes what gets computed or
# persisted, only how often) — kept local here rather than in `config.py`,
# whose tunables all affect the simulation's actual behavior/output. 200
# keeps a worst-case loss (a crash right before a flush) to a small fraction
# of a real-sized (thousands-of-images) image pool while still batching most
# of the network/DB round-trip savings a straight per-image commit would
# give up. Reused (rather than duplicated) for the model-inference loop —
# same reasoning, same value.
COMPLEXITY_SCORE_FLUSH_BATCH_SIZE = 200


def run_simulation(
    engine: Engine,
    preset_name: str,
    *,
    image_records: list[ImageRecord],
    resolve_image: Callable[[str], Path],
    run_local_inference: inference.RunInferenceFn,
    run_offload_inference: inference.RunInferenceFn,
    ground_truth: dict[int, list[ground_truth.Box]],
    dataset: str | None = None,
    frame_count: int | None = None,
    condition_vector_count: int | None = None,
    bucket_count: int | None = None,
    seed: int | None = None,
    lambda_value: float | None = None,
) -> str:
    """Run one full simulator invocation for `preset_name`, return its run id.

    `image_records` and `resolve_image` are required — there is no default
    image pool; the real CLI (`main`) always builds them from the caller's
    `--annotations`/`--images` (`image_source.load_image_index()` /
    `image_source.resolve_image_path`), and tests inject a small fake pool
    and a resolver pointed at synthetic temp-directory images instead.
    `run_local_inference`/`run_offload_inference` are likewise required —
    the real CLI builds them once per invocation via
    `inference.build_local_inference_fn()`/`build_offload_inference_fn()`
    (each loads its YOLO model exactly once), and tests inject small, fast,
    deterministic fakes instead of loading a real model. `ground_truth` maps
    `image_id` to that image's ground-truth boxes (`ground_truth.load_ground_truth()`
    on the same annotations file `image_records` came from); an image with no
    entry is treated as having zero ground-truth boxes. `dataset`/`frame_count`/
    `condition_vector_count`/`bucket_count`/`seed`/`lambda_value` default to
    `datagen.config`'s tunables when omitted, so a reduced-scale test run
    doesn't have to exercise the real 500 x 50 defaults against a tiny fake
    pool.
    """
    if preset_name not in presets.PRESETS:
        raise ValueError(
            f"Unknown preset {preset_name!r}. Valid presets: {sorted(presets.PRESETS)}"
        )
    preset = presets.PRESETS[preset_name]

    resolved_frame_count = frame_count if frame_count is not None else config.FRAME_COUNT
    resolved_condition_vector_count = (
        condition_vector_count
        if condition_vector_count is not None
        else config.CONDITION_VECTOR_COUNT
    )
    resolved_bucket_count = (
        bucket_count if bucket_count is not None else config.STRATIFICATION_BUCKET_COUNT
    )
    resolved_dataset = dataset if dataset is not None else config.DATASET_NAME
    resolved_seed = seed if seed is not None else config.SEED
    resolved_lambda = lambda_value if lambda_value is not None else config.DEFAULT_LAMBDA

    known_complexity = get_known_complexity(engine, resolved_dataset)

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
    total_to_score = len(image_records) - len(known_complexity)
    scored_count = 0
    for record in image_records:
        if record.file_name in known_complexity:
            continue
        image_path = resolve_image(record.file_name)
        score = scene_complexity(image_path)
        all_complexity[record.file_name] = score
        pending_batch[record.file_name] = score
        scored_count += 1
        if len(pending_batch) >= COMPLEXITY_SCORE_FLUSH_BATCH_SIZE:
            store_complexity_scores(engine, resolved_dataset, pending_batch)
            pending_batch = {}
            logger.info("Scored %d/%d images.", scored_count, total_to_score)
    if pending_batch:
        store_complexity_scores(engine, resolved_dataset, pending_batch)
        logger.info("Scored %d/%d images.", scored_count, total_to_score)

    all_model_inference = _compute_missing_model_inference(
        engine,
        resolved_dataset,
        image_records,
        resolve_image,
        ground_truth,
        run_local_inference,
        run_offload_inference,
    )

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
        dataset=resolved_dataset,
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
        local_base = all_model_inference[frame_id][Label.LOCAL]
        offload_base = all_model_inference[frame_id][Label.OFFLOAD]
        for condition_index, condition in enumerate(condition_vectors):
            # A distinct-but-deterministic seed per (frame, condition) pair —
            # `apply_condition_overhead`'s own contract only guarantees
            # determinism for a fixed seed, so reusing one seed for every row
            # would make every row draw identical noise. Index-derived
            # offsets from the run's seed keep this reproducible across two
            # runs of the same preset/config/seed/pool without needing
            # per-row random state.
            row_seed = resolved_seed + frame_index * len(condition_vectors) + condition_index
            (
                local_latency_ms,
                local_accuracy,
                offload_latency_ms,
                offload_accuracy,
            ) = inference.apply_condition_overhead(condition, local_base, offload_base, row_seed)
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


def _compute_missing_model_inference(
    engine: Engine,
    dataset: str,
    image_records: list[ImageRecord],
    resolve_image: Callable[[str], Path],
    ground_truth: dict[int, list[ground_truth.Box]],
    run_local_inference: inference.RunInferenceFn,
    run_offload_inference: inference.RunInferenceFn,
) -> dict[str, dict[Label, DetectionResult]]:
    """Compute and persist any `model_inference` rows missing for `image_records`.

    Reads the known-cache via `persistence.get_known_model_inference`, then
    for every `ImageRecord` not yet covered by *both* `Label.LOCAL` and
    `Label.OFFLOAD` (a file may already have just one of the two paths
    cached, per `store_model_inference`'s per-pair dedup), resolves its
    image path, looks up its ground-truth boxes (`ground_truth.get(record.image_id,
    [])` — an image with no annotations entries simply scores against zero
    ground-truth boxes), and runs both `run_local_inference`/
    `run_offload_inference` on it. New results are flushed to Postgres in
    batches of `COMPLEXITY_SCORE_FLUSH_BATCH_SIZE` (same crash-resilience
    reasoning as the complexity-scoring loop in `run_simulation` above),
    with the same per-batch progress logging. Returns the full per-file-name
    inference dict (known plus newly computed), keyed the same way
    `get_known_model_inference` is.
    """
    known_model_inference = get_known_model_inference(engine, dataset)
    all_model_inference = dict(known_model_inference)

    missing_records = [
        record
        for record in image_records
        if Label.LOCAL not in known_model_inference.get(record.file_name, {})
        or Label.OFFLOAD not in known_model_inference.get(record.file_name, {})
    ]

    pending_batch: dict[str, dict[Label, DetectionResult]] = {}
    total_to_compute = len(missing_records)
    computed_count = 0
    for record in missing_records:
        image_path = resolve_image(record.file_name)
        frame_ground_truth = ground_truth.get(record.image_id, [])
        by_model = {
            Label.LOCAL: run_local_inference(image_path, frame_ground_truth),
            Label.OFFLOAD: run_offload_inference(image_path, frame_ground_truth),
        }
        all_model_inference[record.file_name] = by_model
        pending_batch[record.file_name] = by_model
        computed_count += 1
        if len(pending_batch) >= COMPLEXITY_SCORE_FLUSH_BATCH_SIZE:
            store_model_inference(engine, dataset, pending_batch)
            pending_batch = {}
            logger.info(
                "Computed inference for %d/%d images.", computed_count, total_to_compute
            )
    if pending_batch:
        store_model_inference(engine, dataset, pending_batch)
        logger.info("Computed inference for %d/%d images.", computed_count, total_to_compute)

    return all_model_inference


def _require_images_dir(images_dir: Path) -> None:
    """Fail fast with a clear error if `images_dir` isn't an existing folder.

    Kept as a standalone function (rather than inlined in `main()`) so it's
    directly unit-testable without going through argparse or a database
    connection.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No such images folder: {images_dir}")


def main() -> None:
    # CLI-only convenience: load DATABASE_URL from training/.env if it isn't
    # already in the environment (never overrides an explicit `export`).
    # `run_simulation` itself stays free of this side effect — only the CLI
    # entry point needs it, not the core function or a plain import of this
    # module.
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Run the data-gen simulator for one preset against a COCO-format "
            "annotations file and images folder, storing scene-complexity "
            "scores under the given dataset name."
        )
    )
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(presets.PRESETS),
        help="Named condition-scenario preset to sample condition vectors from.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to a COCO-format annotations file (JSON with an 'images' list).",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Folder to resolve image files from.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name scene-complexity scores are stored under.",
    )
    args = parser.parse_args()

    _require_images_dir(args.images)
    image_records = image_source.load_image_index(args.annotations)
    # Same annotations file as `image_records`, different array within it —
    # both loaded once here rather than re-reading the file per frame.
    frame_ground_truth = ground_truth.load_ground_truth(args.annotations)
    resolve_image = functools.partial(image_source.resolve_image_path, images_dir=args.images)

    # Each builder loads its YOLO model exactly once per CLI invocation.
    run_local_inference = inference.build_local_inference_fn()
    run_offload_inference = inference.build_offload_inference_fn()

    engine = get_engine()
    run_id = run_simulation(
        engine,
        args.preset,
        image_records=image_records,
        resolve_image=resolve_image,
        run_local_inference=run_local_inference,
        run_offload_inference=run_offload_inference,
        ground_truth=frame_ground_truth,
        dataset=args.dataset,
    )
    print(f"Created run {run_id}")


if __name__ == "__main__":
    main()
