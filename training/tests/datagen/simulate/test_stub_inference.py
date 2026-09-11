"""Tests for the condition-driven stub inference model."""

from __future__ import annotations

from datagen.simulate.stub_inference import stub_inference

# Field order matches conditions.sample_condition_vectors's output.
_BASE_CONDITION = (10.0, 100.0, 2.0, 30.0)  # bandwidth, net latency, loss, device load
_SCENE_COMPLEXITY = 0.15
_SEED = 42

# Noise stds are small relative to the deterministic coefficient deltas used
# below, but not negligible, so monotonicity comparisons average over many
# seeds instead of relying on a single seed's noise draw to not mask the
# trend.
_SEED_SAMPLE_COUNT = 200


def test_same_inputs_and_seed_produce_identical_outputs() -> None:
    first = stub_inference(_BASE_CONDITION, _SCENE_COMPLEXITY, _SEED)
    second = stub_inference(_BASE_CONDITION, _SCENE_COMPLEXITY, _SEED)
    assert first == second


def test_different_seed_changes_noise_draws() -> None:
    first = stub_inference(_BASE_CONDITION, _SCENE_COMPLEXITY, seed=1)
    second = stub_inference(_BASE_CONDITION, _SCENE_COMPLEXITY, seed=2)
    assert first != second


def _mean_outputs(
    condition: tuple[float, float, float, float], scene_complexity: float
) -> tuple[float, float, float, float]:
    """Average the 4 outputs over many seeds to smooth out noise draws."""
    totals = [0.0, 0.0, 0.0, 0.0]
    for seed in range(_SEED_SAMPLE_COUNT):
        outputs = stub_inference(condition, scene_complexity, seed)
        for index, value in enumerate(outputs):
            totals[index] += value
    count = float(_SEED_SAMPLE_COUNT)
    return (totals[0] / count, totals[1] / count, totals[2] / count, totals[3] / count)


def test_increasing_device_load_strictly_increases_local_latency() -> None:
    low_load = (10.0, 100.0, 2.0, 10.0)
    high_load = (10.0, 100.0, 2.0, 90.0)

    low_local_latency, _, _, _ = _mean_outputs(low_load, _SCENE_COMPLEXITY)
    high_local_latency, _, _, _ = _mean_outputs(high_load, _SCENE_COMPLEXITY)

    assert high_local_latency > low_local_latency


def test_decreasing_bandwidth_strictly_increases_offload_latency() -> None:
    high_bandwidth = (50.0, 100.0, 2.0, 30.0)
    low_bandwidth = (5.0, 100.0, 2.0, 30.0)

    _, _, high_bandwidth_offload_latency, _ = _mean_outputs(high_bandwidth, _SCENE_COMPLEXITY)
    _, _, low_bandwidth_offload_latency, _ = _mean_outputs(low_bandwidth, _SCENE_COMPLEXITY)

    assert low_bandwidth_offload_latency > high_bandwidth_offload_latency


def test_increasing_packet_loss_strictly_decreases_offload_accuracy() -> None:
    low_loss = (10.0, 100.0, 0.0, 30.0)
    high_loss = (10.0, 100.0, 20.0, 30.0)

    _, _, _, low_loss_offload_accuracy = _mean_outputs(low_loss, _SCENE_COMPLEXITY)
    _, _, _, high_loss_offload_accuracy = _mean_outputs(high_loss, _SCENE_COMPLEXITY)

    assert high_loss_offload_accuracy < low_loss_offload_accuracy


def test_accuracy_outputs_stay_within_zero_one_under_extreme_inputs() -> None:
    extreme_conditions = [
        (0.2, 400.0, 20.0, 100.0),  # worst-case network-stress + full device load
        (100.0, 10.0, 0.0, 0.0),  # best-case everything
        (0.2, 400.0, 100.0, 100.0),  # beyond-preset packet loss, still must clip
    ]
    extreme_scene_complexities = [0.0, 1.0, 10.0]

    for condition in extreme_conditions:
        for scene_complexity in extreme_scene_complexities:
            for seed in range(10):
                _, local_accuracy, _, offload_accuracy = stub_inference(
                    condition, scene_complexity, seed
                )
                assert 0.0 <= local_accuracy <= 1.0
                assert 0.0 <= offload_accuracy <= 1.0
