"""Tests for stratified frame sampling by scene-complexity score."""

from __future__ import annotations

from datagen.sampling import stratified_sample


def _synthetic_scores(count: int = 200) -> dict[str, float]:
    """A wide, roughly-uniform spread of scores across [0.0, 1.0)."""
    return {f"frame_{i:04d}.jpg": i / count for i in range(count)}


def test_same_seed_and_input_produce_identical_sample() -> None:
    scores = _synthetic_scores()
    first = stratified_sample(scores, target_count=50, bucket_count=5, seed=42)
    second = stratified_sample(scores, target_count=50, bucket_count=5, seed=42)
    assert first == second


def test_different_seeds_can_produce_different_samples() -> None:
    scores = _synthetic_scores()
    first = stratified_sample(scores, target_count=50, bucket_count=5, seed=1)
    second = stratified_sample(scores, target_count=50, bucket_count=5, seed=2)
    assert first != second


def test_sample_has_exact_target_count_with_no_duplicates() -> None:
    scores = _synthetic_scores()
    sample = stratified_sample(scores, target_count=50, bucket_count=5, seed=42)
    assert len(sample) == 50
    assert len(set(sample)) == 50


def test_sample_spans_low_mid_and_high_complexity_scores() -> None:
    scores = _synthetic_scores()
    sample = stratified_sample(scores, target_count=50, bucket_count=5, seed=42)
    sampled_scores = [scores[file_name] for file_name in sample]
    assert min(sampled_scores) < 0.2
    assert max(sampled_scores) > 0.8
    assert max(sampled_scores) - min(sampled_scores) > 0.6


def test_each_bucket_contributes_at_least_one_sample() -> None:
    scores = _synthetic_scores()
    bucket_count = 5
    sample = stratified_sample(scores, target_count=50, bucket_count=bucket_count, seed=42)
    sampled_scores = sorted(scores[file_name] for file_name in sample)

    # With 200 evenly-spread scores split into 5 equal-frequency buckets, each
    # bucket spans a 0.2-wide slice of [0.0, 1.0) — assert every slice has at
    # least one representative in the sample.
    bucket_width = 1.0 / bucket_count
    for bucket_index in range(bucket_count):
        lower = bucket_index * bucket_width
        upper = lower + bucket_width
        assert any(
            lower <= score < upper for score in sampled_scores
        ), f"no sampled score in [{lower}, {upper})"


def test_handles_bucket_with_fewer_items_than_target_share() -> None:
    # A tiny pool where every bucket has far fewer items than an even share of
    # a much larger target — should return everything available, not raise.
    scores = {f"frame_{i}.jpg": float(i) for i in range(10)}
    sample = stratified_sample(scores, target_count=500, bucket_count=5, seed=42)
    assert len(sample) == 10
    assert set(sample) == set(scores)
