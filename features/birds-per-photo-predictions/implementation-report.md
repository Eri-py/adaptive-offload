# Birds Per-Photo Predictions — Implementation Report

**Feature:** Birds Per-Photo Predictions
**Directory:** `features/birds-per-photo-predictions/`
**Branch:** `feature/birds-per-photo-predictions`, cut from `main` at `d40a578`

## Tasks

| # | Task | Outcome |
|---|------|---------|
| 1 | Table model and migration | Completed |
| 2 | Per-photo record computation | Completed |
| 3 | Record store | Completed |
| 4 | Recording entry point | Completed |
| 5 | Summary tables | Completed |
| 6 | Migration, then the real run | Completed |
| 7 | Rewrite the findings | Completed |
| 8 | Regression test run | Completed |

## Files changed

**Added**

| Path | Purpose |
|------|---------|
| `database/migrations/versions/0004_create_bird_predictions_table.py` | Authored migration for `bird_predictions` |
| `training/birds/evaluation/predictions.py` | Per-photo record computation, `record_split`, and the recording entry point |
| `training/birds/evaluation/store.py` | Writes records with replace semantics; reads them back as a DataFrame |
| `training/birds/evaluation/summary.py` | Agreement, confidence-separation and cascade tables; printing and MLflow logging |
| `training/tests/birds/evaluation/test_predictions.py` | Tests for the above |
| `training/tests/birds/evaluation/test_store.py` | Tests for the store |
| `training/tests/birds/evaluation/test_summary.py` | Tests for the tables |
| `features/birds-per-photo-predictions/spec.md` | Feature spec |
| `features/birds-per-photo-predictions/implementation.md` | Implementation plan |

**Edited**

| Path | Change |
|------|--------|
| `database/common/models.py` | Added the `BirdPrediction` model; docstring now says six tables |
| `database/tests/common/test_models.py` | Round-trip extended to the new table |
| `training/birds/dataset/splits.py` | Added `split_labels()`, a public accessor for a split's (photo id, label) pairs |
| `training/tests/birds/dataset/test_splits.py` | Coverage for `split_labels()` |
| `findings/birds.md` | Reproducible committed-checkpoint tables added; pilot and flowers figures retained and marked unreproducible |

**Deleted**

None.

## Tests

25 new test functions; the `training/` suite went from 222 to 247 and `database/` holds at 1.

**Unit**

| File | New tests | Covers |
|------|-----------|--------|
| `training/tests/birds/evaluation/test_predictions.py` | 7 of 10 | Hand-computed predictions, correctness, max probability and top-2 margin through `nn.Identity`; batch-size invariance; misaligned and miscounted labels raising; missing dataset, missing checkpoint and absent GPU each stopping the entry point; the resolved `.env` path |
| `training/tests/birds/evaluation/test_summary.py` | 9 of 10 | Hand-computed agreement, AUC and cascade rows; 0%/100% endpoints equalling each model's accuracy; unclamped gap recovery above 1; zero gap; single-outcome model; mismatched photo sets; tie determinism under reversed and shuffled input |
| `training/tests/birds/dataset/test_splits.py` | 2 of 11 | `split_labels()` against a fake CUB tree in `tmp_path` |

**Integration**

| File | New tests | Covers |
|------|-----------|--------|
| `training/tests/birds/evaluation/test_store.py` | 3 | Write-then-read round trip; re-writing one (split, model) leaving other splits and the other model untouched; numeric photo-id ordering — all against a disposable Postgres database |
| `training/tests/birds/evaluation/test_predictions.py` | 3 of 10 | One record per model per photo across every split, via `record_split` against a disposable Postgres database |
| `training/tests/birds/evaluation/test_summary.py` | 1 of 10 | `main()` printing its tables end to end with tracking off |
| `database/tests/common/test_models.py` | 1 modified | Round-trips a row of all six tables, including `BirdPrediction` with a null `training_run_id` |

All tests pass: `training/` 247 passed, `database/` 1 passed, none skipped. `ruff check` is clean in both packages; `mypy` is clean in `database/` and reports only the nine pre-existing `ndarray` type-arg errors in `training/coco/`, unchanged in count throughout.

## Real-data outcome

Migration `0004` was applied by the user, after which the recording entry point wrote **23,576 rows** — 11,788 photos per model across 5,400 train / 594 val / 5,794 test — with every row carrying its checkpoint's training run id.

Verification performed at Task 6:

- Test accuracies derived from the records (0.7798 small, 0.8699 large) match what `python -m birds.evaluation.evaluate` reports independently.
- Two consecutive summary runs produced byte-identical output.

## Commits

`main..HEAD` — `8380539..81e59db`, ten commits (spec, plan, one per task, plus a plan amendment).

## Notable events

- **The plan was amended mid-execution, with the user's approval** (`23edb5b`). Task 2 surfaced that a split's labels had no public accessor — `split_ids()` returns ids only, and `CubDataset._records` and `_read_index()` are private — so the alignment check had no label source independent of the loader it was validating. `training/birds/dataset/splits.py` and its test file were added to Task 4's scope and `split_labels()` was introduced. The alternatives considered and rejected were dropping the alignment check, and reading the private index from the entry point.
- **Task 4's implementer returned an empty report** — the literal string "placeholder" with no content. The task was not accepted on that basis; it was verified directly from the diffs, the lint/type/test gate, and a check of each success criterion against the tests, and accepted on that evidence.
- **Task 6's first real run failed**, with `RuntimeError: DATABASE_URL is not set`. Both new `main()` functions called `get_engine()` without loading `training/.env`, where the URL lives, diverging from the convention every COCO entry point follows. The 246 tests passing at that point did not catch it because they monkeypatch `get_engine`, so the real resolution path was never exercised. Fixed under Task 6 by sharing an `ENV_PATH` constant and loading it in both entry points, with a test pinning the resolved path; the run then completed. No rows had been written before the failure.
- **Task 6 stopped and waited for the user** to apply migration `0004`, per `CLAUDE.md`'s Infrastructure rule. Its DDL was verified beforehand with `alembic upgrade 0003:0004 --sql`, which connects to no database.
- **Two design smells were recorded rather than fixed**, both tracing to the plan rather than to a task: `predictions.py` imports the private `_MODEL_NAMES` and `_load_checkpoint` from the frozen `evaluate.py`, and `predictions.py` and `store.py` import each other, worked around with a function-local import. Both are logged in `learnings.md` for the reviewer.
