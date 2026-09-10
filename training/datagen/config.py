"""Single tunable config module for the data-gen simulator.

Every simulator invocation reads its frame count, condition-sampling
parameters, seed, and utility-function weighting from here — no hardcoded
tunables elsewhere in `training/datagen/`, per the feature spec. Named
condition-scenario presets live separately in `datagen.presets`, since a
preset is a scenario definition rather than a single tunable knob.
"""

# --- Sampling / run-shape tunables -----------------------------------------

# Frames sampled (stratified by scene complexity) from the val2017 pool.
FRAME_COUNT = 500

# Space-filling condition vectors sampled per run; crossed with every frame.
CONDITION_VECTOR_COUNT = 50

# Number of complexity-based strata the frame sample is drawn evenly across.
STRATIFICATION_BUCKET_COUNT = 5

# Seed for both frame stratified-sampling and condition-vector sampling.
SEED = 42

# Default utility-function weight: score = accuracy - DEFAULT_LAMBDA * (latency_ms / 1000),
# i.e. per second of latency (see labeling.py's `_utility`). At 0.3, a 300ms
# latency gap moves utility by 0.09 — comparable in magnitude to a realistic
# accuracy gap between the local/offload stub paths (~0.07-0.1) — so neither
# term structurally dominates the label.
DEFAULT_LAMBDA = 0.3

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
