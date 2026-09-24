"""Standalone condition-vector preview, no database access at all.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.cli.preview_conditions --preset baseline --count 50 --seed 42

Composes `presets.PRESETS` with `conditions.sample_condition_vectors` (both
pure, no I/O) to report which condition vectors a preset/count/seed would
sample without running the full simulator pipeline. Pure function: never
touches Postgres, never runs frame sampling, stub inference, or labeling.
"""

from __future__ import annotations

import argparse

from datagen import config, presets
from datagen.sampling.conditions import sample_condition_vectors


def preview_conditions(
    preset_name: str,
    count: int,
    seed: int,
) -> list[tuple[float, float, float, float]]:
    """Report which condition vectors `preset_name`/`count`/`seed` would sample.

    Looks `preset_name` up in `presets.PRESETS` and feeds its ranges straight
    into `conditions.sample_condition_vectors` with the given `count`/`seed`.
    Returns the same list `sample_condition_vectors` would return directly
    for that preset's ranges.

    Raises `ValueError` if `preset_name` isn't a known preset — a typo'd
    name would otherwise surface as a bare `KeyError` with no indication of
    what names are valid.
    """
    try:
        preset = presets.PRESETS[preset_name]
    except KeyError:
        raise ValueError(
            f"Unknown preset {preset_name!r}. "
            f"Valid presets: {sorted(presets.PRESETS)}"
        ) from None
    return sample_condition_vectors(preset, count, seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview which condition vectors a preset/count/seed would sample."
    )
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(presets.PRESETS),
        help="Condition-scenario preset to sample from.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=config.CONDITION_VECTOR_COUNT,
        help=f"Number of condition vectors to sample (default: {config.CONDITION_VECTOR_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help=f"Random seed for sampling (default: {config.SEED}).",
    )
    args = parser.parse_args()

    vectors = preview_conditions(args.preset, args.count, args.seed)

    for bandwidth_mbps, network_latency_ms, packet_loss_pct, device_load_pct in vectors:
        print(
            f"{bandwidth_mbps:.4f}\t{network_latency_ms:.4f}\t"
            f"{packet_loss_pct:.4f}\t{device_load_pct:.4f}"
        )


if __name__ == "__main__":
    main()
