# Data-Gen Simulator — Implementation Report

**Feature:** Data-Gen Simulator
**Directory:** `features/data-gen-router-loop/01-datagen-simulator/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — `server/common` DB scaffolding | completed |
| Task 1b — Redo Task 1's test against real Postgres (corrective, not in original plan) | completed |
| Task 2 — Alembic migration | completed |
| Task 3 — `training/datagen` package + config module | completed |
| Task 4 — COCO acquisition and local image caching | completed |
| Task 5 — Scene-complexity proxy | completed |
| Task 6 — Persistence layer (all three tables) | completed |
| Task 7 — Stratified frame sampling | completed |
| Task 8 — Condition-vector sampling | completed |
| Task 9 — Stub inference model | completed |
| Task 10 — Win/loss labeling | completed |
| Task 11 — Orchestration script + end-to-end integration test | completed |
| Task 12 — Regression test run | completed |

## Files changed

### Added

- `.gitignore`
- `features/data-gen-router-loop/01-datagen-simulator/spec.md`
- `features/data-gen-router-loop/01-datagen-simulator/implementation.md`
- `server/pyproject.toml`
- `server/alembic.ini`
- `server/common/__init__.py`
- `server/common/db.py`
- `server/common/models.py`
- `server/common/testing.py`
- `server/migrations/env.py`
- `server/migrations/script.py.mako`
- `server/migrations/versions/0001_create_simulation_tables.py`
- `server/tests/__init__.py`
- `server/tests/conftest.py`
- `server/tests/common/__init__.py`
- `server/tests/common/test_models.py`
- `training/pyproject.toml`
- `training/datagen/__init__.py`
- `training/datagen/config.py`
- `training/datagen/coco.py`
- `training/datagen/complexity.py`
- `training/datagen/persistence.py`
- `training/datagen/sampling.py`
- `training/datagen/conditions.py`
- `training/datagen/stub_inference.py`
- `training/datagen/labeling.py`
- `training/datagen/run_simulation.py`
- `training/tests/__init__.py`
- `training/tests/conftest.py`
- `training/tests/datagen/__init__.py`
- `training/tests/datagen/test_coco.py`
- `training/tests/datagen/test_complexity.py`
- `training/tests/datagen/test_config.py`
- `training/tests/datagen/test_persistence.py`
- `training/tests/datagen/test_sampling.py`
- `training/tests/datagen/test_conditions.py`
- `training/tests/datagen/test_stub_inference.py`
- `training/tests/datagen/test_labeling.py`
- `training/tests/datagen/test_run_simulation.py`

### Edited

- `.claude/coding-guidelines.md` — harness→simulator terminology
- `.claude/commands/generateFeatureSpec.md` — skip spec-only PR step, proceed straight to implementation
- `CLAUDE.md` — harness→simulator terminology; documented the scoped exception allowing test fixtures to create/drop ephemeral Postgres test databases
- `HANDOFF.md` — harness→simulator terminology

Not tracked in git but produced as part of this work: the COCO val2017 dataset (annotations + 5,000 images, ~1.6GB) was downloaded to `training/data/coco/` ahead of implementation, per the user's request to have it available locally — this directory is gitignored (`training/data/`).

## Tests

**Unit** (pure functions / mocked I/O, no database):
- `training/tests/datagen/test_config.py` (2 tests): every preset defines all four ranges; frame/condition counts are positive ints.
- `training/tests/datagen/test_coco.py` (4 tests): annotation loading, cached-file resolution never fetches, missing-file resolution fetches and caches exactly once.
- `training/tests/datagen/test_complexity.py` (5 tests): blank image scores near 0, checkerboard/noise images score meaningfully higher, grayscale input accepted, output always in [0, 1].
- `training/tests/datagen/test_sampling.py` (6 tests): determinism, differing seeds diverge, exact count with no duplicates, min/max spread across buckets, every bucket contributes, undersized-bucket edge case.
- `training/tests/datagen/test_conditions.py` (5 tests): determinism, differing seeds diverge, exact count, bounds sanity across all four presets, range coverage within tolerance.
- `training/tests/datagen/test_stub_inference.py` (6 tests): determinism, monotonicity (device load → local latency, bandwidth → offload latency, packet loss → offload accuracy), accuracy range sanity under extreme inputs.
- `training/tests/datagen/test_labeling.py` (5 tests): clear local win, clear offload win, exact tie, near tie, lambda sensitivity.

**Integration** (real ephemeral Postgres database, created/dropped per test via the scoped `CLAUDE.md` exception):
- `server/tests/common/test_models.py` (1 test): round-trip insert/read across all three ORM models plus the FK link.
- `training/tests/datagen/test_persistence.py` (4 tests): scene-complexity store/read round-trip and de-duplication on re-store, per-dataset scoping, run+results round-trip with FK linkage and full field verification.
- `training/tests/datagen/test_run_simulation.py` (2 tests): full pipeline against a 20-image synthetic pool — exact row count with every field populated and correctly linked to the run; reproducibility (identical results except `run_id`) and scene-complexity reuse (zero re-fetches, stable row count) across two runs.

**40 tests added · all passing** (1 in `server/`, 39 in `training/`), 0 skipped. Both packages' `ruff check` and `mypy` (strict) are clean. No stray ephemeral test database left behind after any run (verified after every task and at the final regression run).

## Commits

`main..feature/data-gen-router-loop` (21 commits): from `fcf365f` (Add feature spec) through `8818d09` (Task 12: regression test run). Includes the spec/plan authoring commits, the harness→simulator rename, the scoped-exception documentation, all 12 plan tasks, the Task 1b corrective task, and one standalone config fix (mypy `python_version`).

## Notable events

- **Task 1b (corrective, not in the original plan):** partway through execution, the user judged in-memory SQLite too dialect-divergent from Postgres (native `ENUM`, JSON, FK enforcement) to trust for these models, and asked to test against real Postgres instead. Since `CLAUDE.md` forbids creating/dropping a database on my own initiative, this required the user's explicit sign-off on a scoped exception (documented in `CLAUDE.md`), which was then granted. Task 1's test was rebuilt on a reusable ephemeral-Postgres-database fixture (`server/common/testing.py` + `server/tests/conftest.py`), which every subsequent DB-touching task (6, 11, and `server/`'s own suite) reused. This rebuild caught a real FK-ordering bug that SQLite's lack of constraint enforcement had silently let through.
- **Standalone fix (between Tasks 4 and 5):** a pre-existing `mypy` `python_version = "3.11"` vs. actual Python 3.12 venv mismatch (present in both `server/` and `training/`, inherited from `server/`'s original config) surfaced as a numpy-stub compatibility issue during Task 4. Fixed directly in both packages' `pyproject.toml` rather than carrying a workaround forward.
- **Dataset acquisition ahead of implementation:** the user asked for the COCO val2017 dataset to be downloaded to `training/data/coco/` before task execution began, given a slow home connection — this was done once, up front, and Tasks 4/11 were briefed to treat it as already present rather than downloading it themselves.
- No task failed or required a stop-and-report to the user; all 12 tasks plus the corrective task completed on the first attempt.
