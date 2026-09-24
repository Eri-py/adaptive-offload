"""Validates the shape of `datagen.presets`' named condition-scenario ranges
(no I/O)."""

from datagen import presets

_REQUIRED_AXES = ("bandwidth_mbps", "network_latency_ms", "packet_loss_pct", "device_load_pct")


def test_every_preset_defines_all_four_ranges() -> None:
    assert len(presets.PRESETS) > 0
    for preset_name, ranges in presets.PRESETS.items():
        for axis in _REQUIRED_AXES:
            assert axis in ranges, f"preset {preset_name!r} is missing axis {axis!r}"
            low, high = ranges[axis]  # type: ignore[literal-required]
            assert low < high, f"preset {preset_name!r} axis {axis!r} has an empty range"
