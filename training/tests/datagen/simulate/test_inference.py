"""Tests for `datagen.simulate.inference`'s pure accuracy-scoring and
condition-driven latency-overhead pieces.
"""

from datagen.simulate.ground_truth import Box
from datagen.simulate.inference import DetectionResult, apply_condition_overhead, score_accuracy

# Field order matches conditions.sample_condition_vectors's output.
_BASE_CONDITION = (10.0, 100.0, 2.0, 30.0)  # bandwidth, net latency, loss, device load
_LOCAL_BASE = DetectionResult(latency_ms=45.0, accuracy=0.78)
_OFFLOAD_BASE = DetectionResult(latency_ms=60.0, accuracy=0.85)
_SEED = 42

# Noise stds are small relative to the deterministic coefficient deltas used
# below, but not negligible, so monotonicity comparisons average over many
# seeds instead of relying on a single seed's noise draw to not mask the
# trend.
_SEED_SAMPLE_COUNT = 200


def test_apply_condition_overhead_same_inputs_and_seed_produce_identical_outputs() -> None:
    first = apply_condition_overhead(_BASE_CONDITION, _LOCAL_BASE, _OFFLOAD_BASE, _SEED)
    second = apply_condition_overhead(_BASE_CONDITION, _LOCAL_BASE, _OFFLOAD_BASE, _SEED)
    assert first == second


def test_apply_condition_overhead_different_seed_changes_noise_draws() -> None:
    first = apply_condition_overhead(_BASE_CONDITION, _LOCAL_BASE, _OFFLOAD_BASE, seed=1)
    second = apply_condition_overhead(_BASE_CONDITION, _LOCAL_BASE, _OFFLOAD_BASE, seed=2)
    assert first != second


def test_apply_condition_overhead_passes_accuracy_through_unchanged() -> None:
    for seed in range(5):
        _, local_accuracy, _, offload_accuracy = apply_condition_overhead(
            _BASE_CONDITION, _LOCAL_BASE, _OFFLOAD_BASE, seed
        )
        assert local_accuracy == _LOCAL_BASE.accuracy
        assert offload_accuracy == _OFFLOAD_BASE.accuracy


def _mean_latencies(condition: tuple[float, float, float, float]) -> tuple[float, float]:
    """Average local/offload latency over many seeds to smooth out noise draws."""
    local_total = 0.0
    offload_total = 0.0
    for seed in range(_SEED_SAMPLE_COUNT):
        local_latency_ms, _, offload_latency_ms, _ = apply_condition_overhead(
            condition, _LOCAL_BASE, _OFFLOAD_BASE, seed
        )
        local_total += local_latency_ms
        offload_total += offload_latency_ms
    count = float(_SEED_SAMPLE_COUNT)
    return (local_total / count, offload_total / count)


def test_apply_condition_overhead_increasing_device_load_increases_local_latency() -> None:
    low_load = (10.0, 100.0, 2.0, 10.0)
    high_load = (10.0, 100.0, 2.0, 90.0)

    low_local_latency, _ = _mean_latencies(low_load)
    high_local_latency, _ = _mean_latencies(high_load)

    assert high_local_latency > low_local_latency


def test_apply_condition_overhead_decreasing_bandwidth_increases_offload_latency() -> None:
    high_bandwidth = (50.0, 100.0, 2.0, 30.0)
    low_bandwidth = (5.0, 100.0, 2.0, 30.0)

    _, high_bandwidth_offload_latency = _mean_latencies(high_bandwidth)
    _, low_bandwidth_offload_latency = _mean_latencies(low_bandwidth)

    assert low_bandwidth_offload_latency > high_bandwidth_offload_latency


def test_apply_condition_overhead_adds_on_top_of_real_base_latency() -> None:
    # With every condition-driven term at zero (except bandwidth, which
    # divides), the only latency contribution is the base latency plus the
    # bandwidth term plus that seed's noise draw — swapping in a different
    # base latency shifts the result by exactly the base delta.
    zero_condition = (100.0, 0.0, 0.0, 0.0)
    higher_local_base = DetectionResult(latency_ms=_LOCAL_BASE.latency_ms + 20.0, accuracy=0.78)

    base_local_latency_ms, _, _, _ = apply_condition_overhead(
        zero_condition, _LOCAL_BASE, _OFFLOAD_BASE, _SEED
    )
    higher_local_latency_ms, _, _, _ = apply_condition_overhead(
        zero_condition, higher_local_base, _OFFLOAD_BASE, _SEED
    )

    assert higher_local_latency_ms - base_local_latency_ms == 20.0


_CAT_BOX = Box(category_name="cat", x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
_CAT_BOX_HIGH_IOU = Box(category_name="cat", x_min=1.0, y_min=1.0, x_max=11.0, y_max=11.0)
_CAR_BOX_SAME_POSITION = Box(category_name="car", x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
_DOG_BOX = Box(category_name="dog", x_min=100.0, y_min=100.0, x_max=110.0, y_max=110.0)
_DOG_BOX_HIGH_IOU = Box(
    category_name="dog", x_min=101.0, y_min=101.0, x_max=111.0, y_max=111.0
)


def test_score_accuracy_perfect_match_scores_one() -> None:
    ground_truth_boxes = [_CAT_BOX, _DOG_BOX]
    predicted_boxes = [_CAT_BOX_HIGH_IOU, _DOG_BOX_HIGH_IOU]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 1.0


def test_score_accuracy_no_matching_predictions_scores_zero() -> None:
    ground_truth_boxes = [_CAT_BOX, _DOG_BOX]
    predicted_boxes: list[Box] = []

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 0.0


def test_score_accuracy_partial_match_scores_correct_fraction() -> None:
    ground_truth_boxes = [_CAT_BOX, _DOG_BOX]
    predicted_boxes = [_CAT_BOX_HIGH_IOU]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 0.5


def test_score_accuracy_same_position_different_category_does_not_match() -> None:
    ground_truth_boxes = [_CAT_BOX]
    predicted_boxes = [_CAR_BOX_SAME_POSITION]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 0.0


def test_score_accuracy_zero_ground_truth_boxes_scores_one() -> None:
    ground_truth_boxes: list[Box] = []
    predicted_boxes = [_CAT_BOX]

    assert score_accuracy(predicted_boxes, ground_truth_boxes) == 1.0


def test_score_accuracy_below_iou_threshold_does_not_match() -> None:
    # IoU here is 25 / 175 ≈ 0.14 — below the default 0.5 threshold.
    ground_truth_boxes = [_CAT_BOX]
    slightly_overlapping = Box(category_name="cat", x_min=5.0, y_min=5.0, x_max=15.0, y_max=15.0)

    assert score_accuracy([slightly_overlapping], ground_truth_boxes) == 0.0


def test_score_accuracy_respects_custom_iou_threshold() -> None:
    # Same ~0.14 IoU box as above, but now the threshold is low enough to match.
    ground_truth_boxes = [_CAT_BOX]
    slightly_overlapping = Box(category_name="cat", x_min=5.0, y_min=5.0, x_max=15.0, y_max=15.0)

    assert (
        score_accuracy([slightly_overlapping], ground_truth_boxes, iou_threshold=0.1) == 1.0
    )
