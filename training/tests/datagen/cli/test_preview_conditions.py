"""Tests for the standalone condition-vector preview CLI.

Pure-function core, no database access — unlike `test_preview_sample.py`,
this doesn't need the `postgres_engine` fixture at all.
"""

from __future__ import annotations

import pytest

from datagen.cli.preview_conditions import preview_conditions
from datagen.presets import PRESETS
from datagen.sampling.conditions import sample_condition_vectors

COUNT = 50
SEED = 42


def test_preview_conditions_matches_sample_condition_vectors_directly() -> None:
    expected = sample_condition_vectors(PRESETS["baseline"], COUNT, SEED)

    actual = preview_conditions("baseline", COUNT, SEED)

    assert actual == expected


def test_preview_conditions_matches_for_every_preset() -> None:
    for name, preset in PRESETS.items():
        expected = sample_condition_vectors(preset, COUNT, SEED)
        assert preview_conditions(name, COUNT, SEED) == expected


def test_preview_conditions_raises_clear_error_for_unknown_preset() -> None:
    with pytest.raises(ValueError, match="not-a-real-preset"):
        preview_conditions("not-a-real-preset", COUNT, SEED)
