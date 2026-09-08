"""Single tunable config module for the data-gen simulator.

Every simulator invocation reads its frame count, condition-sampling
parameters, preset ranges, seed, and utility-function weighting from here —
no hardcoded tunables elsewhere in `training/datagen/`, per the feature spec.
"""

from typing import TypedDict

# --- Sampling / run-shape tunables -----------------------------------------

# Frames sampled (stratified by scene complexity) from the val2017 pool.
FRAME_COUNT = 500

# Space-filling condition vectors sampled per run; crossed with every frame.
CONDITION_VECTOR_COUNT = 50

# Number of complexity-based strata the frame sample is drawn evenly across.
STRATIFICATION_BUCKET_COUNT = 5

# Seed for both frame stratified-sampling and condition-vector sampling.
SEED = 42

# Default utility-function weight: score = accuracy - DEFAULT_LAMBDA * latency.
# Latency is in milliseconds and accuracy is a 0-1 rate, so this is scaled
# down to keep the latency term comparable in magnitude to the accuracy term.
DEFAULT_LAMBDA = 0.005

DATASET_NAME = "coco_val2017"

# --- Stub-model coefficients -------------------------------------------------
# Illustrative only (no real models yet, per the spec's Out of Scope) — picked
# to produce plausible-shaped latency/accuracy curves, not measured from a
# real device or network.

# Local path: latency scales with device load; accuracy has a smaller base
# rate than the offload path (a smaller on-device model).
LOCAL_BASE_LATENCY_MS = 45.0
LOCAL_LATENCY_DEVICE_LOAD_COEFFICIENT_MS = 1.2  # added ms per device-load pct point
LOCAL_LATENCY_NOISE_STD_MS = 5.0
LOCAL_BASE_ACCURACY = 0.78
LOCAL_ACCURACY_NOISE_STD = 0.03

# Offload path: latency scales with bandwidth (inverse), round-trip network
# latency (~1:1 passthrough), and packet loss (retransmit penalty); accuracy
# has a higher base rate (a larger server-side model).
OFFLOAD_BASE_LATENCY_MS = 60.0
OFFLOAD_LATENCY_BANDWIDTH_COEFFICIENT_MS = 40.0  # scales as 1/bandwidth_mbps
OFFLOAD_LATENCY_NETWORK_LATENCY_COEFFICIENT = 1.0  # ms added per ms of RTT
OFFLOAD_LATENCY_PACKET_LOSS_PENALTY_COEFFICIENT_MS = 3.0  # ms added per pct point
OFFLOAD_LATENCY_NOISE_STD_MS = 8.0
OFFLOAD_BASE_ACCURACY = 0.85
OFFLOAD_ACCURACY_NOISE_STD = 0.03

# Both paths' accuracy drops as the frame's scene-complexity proxy rises.
SCENE_COMPLEXITY_ACCURACY_PENALTY_COEFFICIENT = 0.15

# Small additional per-path accuracy penalties (Task 9): a heavily-loaded
# device is modeled as falling back to a lighter/faster on-device model, and
# a lossy link is modeled as losing detail to dropped/retransmitted frames —
# both illustrative, same as every other stub-model coefficient above.
LOCAL_ACCURACY_DEVICE_LOAD_PENALTY_COEFFICIENT = 0.001  # per device-load pct point
OFFLOAD_ACCURACY_PACKET_LOSS_PENALTY_COEFFICIENT = 0.005  # per packet-loss pct point


class ConditionPresetRanges(TypedDict):
    """Per-axis (min, max) sampling ranges for one condition-scenario preset."""

    bandwidth_mbps: tuple[float, float]
    network_latency_ms: tuple[float, float]
    packet_loss_pct: tuple[float, float]
    device_load_pct: tuple[float, float]


# Named condition-scenario presets. Each invocation selects one by name; the
# resolved ranges are snapshotted onto that run's `simulation_runs` record.
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
