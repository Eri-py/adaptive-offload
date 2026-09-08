## Task 1 — `server/common` DB scaffolding
- Status: completed
- Started: 2026-09-07 18:48:00
- Completed: 2026-09-07 18:51:31
- Notes: ruff/mypy clean, 1 test passed (in-memory SQLite round-trip across all three models). Import root is `common.*`, not `server.common.*` — see learnings.md.

## Task 1b — Redo Task 1's test against real Postgres (corrective, not in original plan)
- Status: completed
- Started: 2026-09-07 19:00:28
- Completed: 2026-09-07 19:04:34
- Notes: New `common/testing.py` ephemeral-DB context manager + `server/tests/conftest.py` fixture, reused by Task 6 later. Caught a real FK-ordering bug SQLite had silently let pass (no FK enforcement) — validates the switch to real Postgres. ruff/mypy clean, 1 test passed, no stray test DB left behind.

## Task 2 — Alembic migration
- Status: completed
- Started: 2026-09-07 18:51:45
- Completed: 2026-09-07 18:58:18
- Notes: `alembic upgrade head --sql` verified offline (no DB connection), DDL matches all three tables + FK. ruff/mypy clean, existing test still passes.

## Task 3 — `training/datagen` package + config module
- Status: completed
- Started: 2026-09-07 19:05:03
- Completed: 2026-09-07 19:10:40
- Notes: `common.*` import from training/'s venv verified working (editable install of ../server). ruff/mypy clean, 2 tests passed. Added test_config.py (not in original Files list, but required by the task's own success criteria).

## Task 4 — COCO acquisition and local image caching
- Status: completed
- Started: 2026-09-07 19:10:59
- Completed: 2026-09-07 19:15:21
- Notes: Real annotations file confirmed parsing to exactly 5,000 entries; cached-file resolution confirmed instant/no-fetch. ruff/mypy clean, 6/6 tests passed (full suite). Surfaced+fixed a pre-existing mypy python_version 3.11-vs-3.12 mismatch in both server/ and training/ (separate commit).

## Task 5 — Scene-complexity proxy
- Status: completed
- Started: 2026-09-07 19:15:35
- Completed: 2026-09-07 19:18:56
- Notes: Canny thresholds 100/200 (OpenCV default pair). Manual spot-check on 5 real val2017 images gave scores ~0.06-0.23 (plausible spread). ruff/mypy clean, 11/11 tests passed. Confirmed the earlier python_version fix resolved Task 4's pytest-import mypy workaround.

## Task 6 — Persistence layer (all three tables)
- Status: completed
- Started: 2026-09-07 19:19:25
- Completed: 2026-09-07 19:24:25
- Notes: Real ephemeral-Postgres fixture reused in training/tests/conftest.py. Added mypy_path="../server" to training/pyproject.toml (needed for static resolution of the editable common.* install; verified server/'s own checks still pass). ruff/mypy clean, 15/15 tests passed, no stray test DB.

## Task 7 — Stratified frame sampling
- Status: completed
- Started: 2026-09-07 19:24:39
- Completed: 2026-09-07 19:26:46
- Notes: Quantile bucketing via sort+array_split, seeded even-share sampling with divmod remainder distribution. ruff/mypy clean, 21/21 tests passed (6 new, no regressions).

## Task 8 — Condition-vector sampling
- Status: completed
- Started: 2026-09-07 19:27:00
- Completed: 2026-09-07 19:28:43
- Notes: scipy LHS + qmc.scale into per-axis preset ranges. ruff/mypy clean, 26/26 tests passed (5 new, no regressions).

## Task 9 — Stub inference model
- Status: completed
- Started: 2026-09-07 19:28:57
- Completed: 2026-09-07 19:31:51
- Notes: Added two small flagged config constants (device-load/packet-loss accuracy penalties). Single seeded RNG, fixed draw order, accuracy clipped to [0,1]. ruff/mypy clean, 32/32 tests passed (6 new, no regressions).

## Task 10 — Win/loss labeling
- Status: completed
- Started: 2026-09-07 19:32:06
- Completed: 2026-09-07 19:34:25
- Notes: Returns common.models.Label enum, not a string. Exact-tie breaks to LOCAL (documented). ruff/mypy clean, 37/37 tests passed (5 new, no regressions).

## Task 11 — Orchestration script + end-to-end integration test
- Status: completed
- Started: 2026-09-07 19:35:20
- Completed: 2026-09-07 19:41:57
- Notes: Real invocation is `python -m datagen.run_simulation --preset <name>` (training/ import root, not training.datagen.*) — verified via --help. Core function fully dependency-injected (image_records, resolve_image, all tunables) for testability. Reproducibility and scene_complexity-reuse both verified (row-for-row match except run_id; 0 resolve_image calls on second run). ruff/mypy clean, 39/39 tests passed, no stray test DB.

## Task 12 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
