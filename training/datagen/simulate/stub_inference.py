"""Condition-driven synthetic latency/accuracy model for both inference paths.

Fully stubbed per the spec's Out of Scope — no real local/offload models
exist yet. This module only needs to produce plausible-shaped, deterministic
latency/accuracy numbers so the win/loss labels computed downstream (Task 10)
are meaningful. Pure function: no I/O, no database access. All coefficients
come from `datagen.config`.
"""

from __future__ import annotations

import numpy as np

from datagen.config import (
    LOCAL_ACCURACY_DEVICE_LOAD_PENALTY_COEFFICIENT,
    LOCAL_ACCURACY_NOISE_STD,
    LOCAL_BASE_ACCURACY,
    LOCAL_BASE_LATENCY_MS,
    LOCAL_LATENCY_DEVICE_LOAD_COEFFICIENT_MS,
    LOCAL_LATENCY_NOISE_STD_MS,
    OFFLOAD_ACCURACY_NOISE_STD,
    OFFLOAD_ACCURACY_PACKET_LOSS_PENALTY_COEFFICIENT,
    OFFLOAD_BASE_ACCURACY,
    OFFLOAD_BASE_LATENCY_MS,
    OFFLOAD_LATENCY_BANDWIDTH_COEFFICIENT_MS,
    OFFLOAD_LATENCY_NETWORK_LATENCY_COEFFICIENT,
    OFFLOAD_LATENCY_NOISE_STD_MS,
    OFFLOAD_LATENCY_PACKET_LOSS_PENALTY_COEFFICIENT_MS,
    SCENE_COMPLEXITY_ACCURACY_PENALTY_COEFFICIENT,
)

# Condition-vector field order, matching `conditions.sample_condition_vectors`.
_BANDWIDTH_INDEX = 0
_NETWORK_LATENCY_INDEX = 1
_PACKET_LOSS_INDEX = 2
_DEVICE_LOAD_INDEX = 3


def stub_inference(
    condition: tuple[float, float, float, float],
    scene_complexity: float,
    seed: int,
) -> tuple[float, float, float, float]:
    """Compute stubbed (local_latency_ms, local_accuracy, offload_latency_ms,
    offload_accuracy) for one (condition, frame) pair.

    `condition` is `(bandwidth_mbps, network_latency_ms, packet_loss_pct,
    device_load_pct)`, matching `conditions.sample_condition_vectors`'s
    output order. Noise for each of the 4 outputs is independent Gaussian
    noise drawn from a `seed`-derived RNG, in a fixed draw order, so the same
    `(condition, scene_complexity, seed)` always returns identical values and
    varying `seed` alone changes only the noise draws. Accuracy outputs are
    clipped to `[0, 1]` after noise/penalties are applied.
    """
    bandwidth_mbps = condition[_BANDWIDTH_INDEX]
    network_latency_ms = condition[_NETWORK_LATENCY_INDEX]
    packet_loss_pct = condition[_PACKET_LOSS_INDEX]
    device_load_pct = condition[_DEVICE_LOAD_INDEX]

    rng = np.random.default_rng(seed)
    local_latency_noise = rng.normal(0.0, LOCAL_LATENCY_NOISE_STD_MS)
    offload_latency_noise = rng.normal(0.0, OFFLOAD_LATENCY_NOISE_STD_MS)
    local_accuracy_noise = rng.normal(0.0, LOCAL_ACCURACY_NOISE_STD)
    offload_accuracy_noise = rng.normal(0.0, OFFLOAD_ACCURACY_NOISE_STD)

    local_latency_ms = (
        LOCAL_BASE_LATENCY_MS
        + LOCAL_LATENCY_DEVICE_LOAD_COEFFICIENT_MS * device_load_pct
        + local_latency_noise
    )

    offload_latency_ms = (
        OFFLOAD_BASE_LATENCY_MS
        + OFFLOAD_LATENCY_BANDWIDTH_COEFFICIENT_MS / bandwidth_mbps
        + OFFLOAD_LATENCY_NETWORK_LATENCY_COEFFICIENT * network_latency_ms
        + OFFLOAD_LATENCY_PACKET_LOSS_PENALTY_COEFFICIENT_MS * packet_loss_pct
        + offload_latency_noise
    )

    local_accuracy = float(
        np.clip(
            LOCAL_BASE_ACCURACY
            - SCENE_COMPLEXITY_ACCURACY_PENALTY_COEFFICIENT * scene_complexity
            - LOCAL_ACCURACY_DEVICE_LOAD_PENALTY_COEFFICIENT * device_load_pct
            + local_accuracy_noise,
            0.0,
            1.0,
        )
    )

    offload_accuracy = float(
        np.clip(
            OFFLOAD_BASE_ACCURACY
            - SCENE_COMPLEXITY_ACCURACY_PENALTY_COEFFICIENT * scene_complexity
            - OFFLOAD_ACCURACY_PACKET_LOSS_PENALTY_COEFFICIENT * packet_loss_pct
            + offload_accuracy_noise,
            0.0,
            1.0,
        )
    )

    return (float(local_latency_ms), local_accuracy, float(offload_latency_ms), offload_accuracy)
