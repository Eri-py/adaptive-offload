"""Tests for Latin Hypercube condition-vector sampling."""

from __future__ import annotations

from datagen.conditions import sample_condition_vectors
from datagen.presets import PRESETS

# Axis order matches the returned tuple's field order in conditions.py.
_AXES = ("bandwidth_mbps", "network_latency_ms", "packet_loss_pct", "device_load_pct")

# LHS stratifies the unit hypercube into `count` equal-width bins per axis and
# draws one sample per bin, so with 50 samples the extreme bins are each
# 1/50 = 2% of the range wide. Allowing the sampled extreme to land anywhere
# within 10% of the range width from its bound is a comfortable margin above
# that 2% guarantee, while still being tight enough to fail if scaling were
# broken (e.g. an axis silently left in [0, 1) or scaled against the wrong
# bounds).
_COVERAGE_TOLERANCE_FRACTION = 0.10


def test_same_preset_and_seed_produce_identical_vectors() -> None:
    preset = PRESETS["baseline"]
    first = sample_condition_vectors(preset, count=50, seed=42)
    second = sample_condition_vectors(preset, count=50, seed=42)
    assert first == second


def test_different_seeds_can_produce_different_vectors() -> None:
    preset = PRESETS["baseline"]
    first = sample_condition_vectors(preset, count=50, seed=1)
    second = sample_condition_vectors(preset, count=50, seed=2)
    assert first != second


def test_returns_exact_count() -> None:
    preset = PRESETS["baseline"]
    vectors = sample_condition_vectors(preset, count=50, seed=42)
    assert len(vectors) == 50


def test_every_axis_value_falls_within_configured_bounds() -> None:
    for preset in PRESETS.values():
        vectors = sample_condition_vectors(preset, count=50, seed=42)
        for axis_index, axis in enumerate(_AXES):
            lower, upper = preset[axis]  # type: ignore[literal-required]
            values = [vector[axis_index] for vector in vectors]
            assert min(values) >= lower
            assert max(values) <= upper


def test_each_axis_spans_close_to_the_full_configured_range() -> None:
    preset = PRESETS["baseline"]
    vectors = sample_condition_vectors(preset, count=50, seed=42)

    for axis_index, axis in enumerate(_AXES):
        lower, upper = preset[axis]  # type: ignore[literal-required]
        width = upper - lower
        tolerance = width * _COVERAGE_TOLERANCE_FRACTION

        values = [vector[axis_index] for vector in vectors]
        assert min(values) <= lower + tolerance, (
            f"{axis}: min {min(values)} not within tolerance of lower bound {lower}"
        )
        assert max(values) >= upper - tolerance, (
            f"{axis}: max {max(values)} not within tolerance of upper bound {upper}"
        )
