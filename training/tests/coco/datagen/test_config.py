"""Validates the shape of `coco.datagen.config`'s tunables (no I/O)."""

from coco.datagen import config


def test_frame_and_condition_counts_are_positive_ints() -> None:
    assert isinstance(config.FRAME_COUNT, int)
    assert config.FRAME_COUNT > 0
    assert isinstance(config.CONDITION_VECTOR_COUNT, int)
    assert config.CONDITION_VECTOR_COUNT > 0
