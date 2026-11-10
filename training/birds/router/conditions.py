"""The birds experiment's cost model: what the local and offload paths cost, in milliseconds.

Latency is scenario-level for birds (no per-photo latency exists), so every cell of the
experiment fixes one local cost and one offload cost shared by all photos. Offload cost is
composed as server compute + payload transfer + round trip.

One input has no default on purpose, because its real value is a measurement this
module must not invent:
- `server_compute_ms`: the large model's measured per-photo GPU latency. It must be
  supplied from the large model's `mean_latency_ms` as logged by `birds.evaluation.evaluate`
  (a desktop GPU standing in for a server one).
Payload is measured (`PAYLOAD_KB`); only server compute is still a required input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Fixed request payload, the measured figure the experiment uses by default. Measured on 300 CUB
# test photos with Resize(232) + CenterCrop(224) (both models' pretrained transforms) and JPEG
# quality 85: mean 11.39 KB, median 10.97 KB, range 3.97-26.71 KB. The spread is why the spec
# treats payload as fixed rather than per-photo. The cost functions still take `payload_kb`.
PAYLOAD_KB = 11.39

# 1 KB = 1000 bytes and 1 Mbps = 10^6 bits/s, so KB * 8 / Mbps is milliseconds.
_BITS_PER_KB = 8_000
_BITS_PER_MBIT = 1_000_000
_MS_PER_S = 1_000

# The only real birds latency measurement: the small model's CPU time on the desktop.
# A marked reference point on the local sweep, not an on-device anchor. Attempt 1's phone
# figures measure YOLOv8n on COCO frames and must never stand in for a birds model.
SMALL_DESKTOP_CPU_MS = 5.877

# Local cost sweep: 1-150 ms in 1 ms steps, wide enough for any plausible on-device cost.
LOCAL_SWEEP_MS: npt.NDArray[np.float64] = np.arange(1.0, 151.0, 1.0)
# Offload cost sweep: 5-1000 ms in 5 ms steps, spanning a fast LAN to a congested cellular link.
OFFLOAD_SWEEP_MS: npt.NDArray[np.float64] = np.arange(5.0, 1005.0, 5.0)


@dataclass(frozen=True)
class NetworkCondition:
    """A named network condition: available bandwidth and round-trip time."""

    name: str
    bandwidth_mbps: float
    round_trip_ms: float


# The project's chosen named points (good / medium / poor), approved by the user. Attempt 1's
# data-gen presets (`coco/datagen/presets.py`) sample ranges rather than naming points
# (baseline 0.5-100 Mbps / 10-400 ms, network-stress 0.2-5 Mbps / 150-400 ms); all three points
# fall inside those ranges, so the two attempts stay comparable.
NETWORK_CONDITIONS: tuple[NetworkCondition, ...] = (
    NetworkCondition("good", bandwidth_mbps=50.0, round_trip_ms=20.0),
    NetworkCondition("medium", bandwidth_mbps=10.0, round_trip_ms=60.0),
    NetworkCondition("poor", bandwidth_mbps=2.0, round_trip_ms=150.0),
)


def transfer_ms(payload_kb: float, bandwidth_mbps: float) -> float:
    """Milliseconds to send `payload_kb` over `bandwidth_mbps` (20 KB at 10 Mbps is 16 ms)."""
    return payload_kb * _BITS_PER_KB / (bandwidth_mbps * _BITS_PER_MBIT) * _MS_PER_S


def offload_cost_ms(
    *, payload_kb: float, bandwidth_mbps: float, round_trip_ms: float, server_compute_ms: float
) -> float:
    """Offload cost: server compute + payload transfer + round trip."""
    return server_compute_ms + transfer_ms(payload_kb, bandwidth_mbps) + round_trip_ms


def condition_offload_ms(
    condition: NetworkCondition, *, payload_kb: float, server_compute_ms: float
) -> float:
    """Offload cost under one named condition."""
    return offload_cost_ms(
        payload_kb=payload_kb,
        bandwidth_mbps=condition.bandwidth_mbps,
        round_trip_ms=condition.round_trip_ms,
        server_compute_ms=server_compute_ms,
    )


def named_offload_costs(*, payload_kb: float, server_compute_ms: float) -> dict[str, float]:
    """Offload cost of every named condition, in `NETWORK_CONDITIONS` order."""
    return {
        c.name: condition_offload_ms(c, payload_kb=payload_kb, server_compute_ms=server_compute_ms)
        for c in NETWORK_CONDITIONS
    }
