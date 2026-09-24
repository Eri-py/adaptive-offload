"""Named condition-scenario presets for the data-gen simulator.

Each invocation selects one preset by name; the resolved ranges are
snapshotted onto that run's `simulation_runs` record. Kept separate from
`config.py`'s run-shape/stub-model tunables since a preset is a named
scenario definition (data), not a single tunable knob.
"""

from typing import TypedDict


class ConditionPresetRanges(TypedDict):
    """Per-axis (min, max) sampling ranges for one condition-scenario preset."""

    bandwidth_mbps: tuple[float, float]
    network_latency_ms: tuple[float, float]
    packet_loss_pct: tuple[float, float]
    device_load_pct: tuple[float, float]


PRESETS: dict[str, ConditionPresetRanges] = {
    "baseline": {
        "bandwidth_mbps": (0.5, 100.0),
        "network_latency_ms": (10.0, 400.0),
        "packet_loss_pct": (0.0, 10.0),
        "device_load_pct": (0.0, 100.0),
    },
    "network-stress": {
        "bandwidth_mbps": (0.2, 5.0),
        "network_latency_ms": (150.0, 400.0),
        "packet_loss_pct": (5.0, 20.0),
        "device_load_pct": (0.0, 100.0),
    },
    "device-stress": {
        "bandwidth_mbps": (0.5, 100.0),
        "network_latency_ms": (10.0, 400.0),
        "packet_loss_pct": (0.0, 10.0),
        "device_load_pct": (60.0, 100.0),
    },
    # Isolates the network axis from device load, per the spec's reasoning
    # for the crossed frame x condition design.
    "degraded-network-idle-device": {
        "bandwidth_mbps": (0.2, 5.0),
        "network_latency_ms": (150.0, 400.0),
        "packet_loss_pct": (5.0, 20.0),
        "device_load_pct": (0.0, 20.0),
    },
}
