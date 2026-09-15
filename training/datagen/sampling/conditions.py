"""Space-filling condition-vector sampling for a run's selected preset.

Pure function: given a preset's per-axis `(min, max)` ranges (from
`presets.PRESETS`), draws `count` 4-dimensional Latin Hypercube samples and
scales each dimension into its preset range, so the sampled conditions cover
the full configured range on every axis instead of clustering at a few
points. No I/O, no database access; the preset dict and seed are passed in by
the caller rather than imported from `datagen.presets`.
"""

from __future__ import annotations

from scipy.stats import qmc

from datagen.presets import ConditionPresetRanges

# Dimension order matches the returned tuple's field order.
_AXES = ("bandwidth_mbps", "network_latency_ms", "packet_loss_pct", "device_load_pct")


def sample_condition_vectors(
    preset: ConditionPresetRanges,
    count: int,
    seed: int,
) -> list[tuple[float, float, float, float]]:
    """Draw `count` condition vectors from `preset`'s ranges via LHS.

    Uses `scipy.stats.qmc.LatinHypercube(d=4, seed=seed)` to draw `count`
    stratified samples in the unit hypercube `[0, 1)^4`, then scales each of
    the 4 dimensions into that axis's `(min, max)` range from `preset`. Axis
    order (and therefore each returned tuple's order) is `bandwidth_mbps`,
    `network_latency_ms`, `packet_loss_pct`, `device_load_pct`.

    Same `preset` + `count` + `seed` always yields the same list.
    """
    sampler = qmc.LatinHypercube(d=len(_AXES), seed=seed)
    unit_samples = sampler.random(n=count)

    # `preset[axis]` with a loop variable trips mypy's `literal-required`
    # check (TypedDict subscripts need a literal key), so build the bounds
    # from the known literal keys directly rather than iterating `_AXES`.
    lower_bounds = [
        preset["bandwidth_mbps"][0],
        preset["network_latency_ms"][0],
        preset["packet_loss_pct"][0],
        preset["device_load_pct"][0],
    ]
    upper_bounds = [
        preset["bandwidth_mbps"][1],
        preset["network_latency_ms"][1],
        preset["packet_loss_pct"][1],
        preset["device_load_pct"][1],
    ]
    scaled = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return [(float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in scaled]
