"""Validates the shape of `datagen.config`'s tunables (no I/O)."""

from datagen import config

_REQUIRED_AXES = ("bandwidth_mbps", "network_latency_ms", "packet_loss_pct", "device_load_pct")


def test_frame_and_condition_counts_are_positive_ints() -> None:
    assert isinstance(config.FRAME_COUNT, int)
    assert config.FRAME_COUNT > 0
    assert isinstance(config.CONDITION_VECTOR_COUNT, int)
    assert config.CONDITION_VECTOR_COUNT > 0


def test_every_preset_defines_all_four_ranges() -> None:
    assert len(config.PRESETS) > 0
    for preset_name, ranges in config.PRESETS.items():
        for axis in _REQUIRED_AXES:
            assert axis in ranges, f"preset {preset_name!r} is missing axis {axis!r}"
            low, high = ranges[axis]  # type: ignore[literal-required]
            assert low < high, f"preset {preset_name!r} axis {axis!r} has an empty range"
