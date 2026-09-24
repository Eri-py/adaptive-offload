"""Tests for the standalone stratified-sample preview CLI.

Runs against a fresh, disposable database per test (`postgres_engine`
fixture in `training/tests/conftest.py`), per this feature's Postgres-only
testing convention (see `learnings.md`'s Task 1/2 notes) — this CLI's core
function is a thin composition of a real `persistence` read plus
`sampling.stratified_sample`, so it needs real Postgres to seed from, not a
fake in-memory stand-in.
"""

from __future__ import annotations

import pytest
from common.models import SimulationResult, SimulationRun
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from datagen.cli.preview_sample import preview_sample
from datagen.persistence import store_complexity_scores
from datagen.sampling.sampling import stratified_sample

DATASET = "coco_val2017"

SCORES = {
    "000000000001.jpg": 0.10,
    "000000000002.jpg": 0.22,
    "000000000003.jpg": 0.35,
    "000000000004.jpg": 0.48,
    "000000000005.jpg": 0.55,
    "000000000006.jpg": 0.61,
    "000000000007.jpg": 0.74,
    "000000000008.jpg": 0.83,
    "000000000009.jpg": 0.91,
    "000000000010.jpg": 0.99,
}

FRAME_COUNT = 5
BUCKET_COUNT = 5
SEED = 7


def test_preview_sample_matches_stratified_sample_directly(postgres_engine: Engine) -> None:
    store_complexity_scores(postgres_engine, DATASET, SCORES)

    expected = stratified_sample(SCORES, FRAME_COUNT, BUCKET_COUNT, SEED)

    selected = preview_sample(postgres_engine, DATASET, FRAME_COUNT, BUCKET_COUNT, SEED)

    assert [file_name for file_name, _ in selected] == expected
    assert [score for _, score in selected] == [SCORES[file_name] for file_name in expected]


def test_preview_sample_creates_no_run_or_result_rows(postgres_engine: Engine) -> None:
    store_complexity_scores(postgres_engine, DATASET, SCORES)

    preview_sample(postgres_engine, DATASET, FRAME_COUNT, BUCKET_COUNT, SEED)

    with Session(postgres_engine) as session:
        run_count = len(session.execute(select(SimulationRun)).scalars().all())
        result_count = len(session.execute(select(SimulationResult)).scalars().all())

    assert run_count == 0
    assert result_count == 0


def test_preview_sample_raises_clear_error_for_dataset_with_no_scores(
    postgres_engine: Engine,
) -> None:
    with pytest.raises(ValueError, match="unscored_dataset"):
        preview_sample(postgres_engine, "unscored_dataset", FRAME_COUNT, BUCKET_COUNT, SEED)
