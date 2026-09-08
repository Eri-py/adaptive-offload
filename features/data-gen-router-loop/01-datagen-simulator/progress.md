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
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 4 — COCO acquisition and local image caching
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Scene-complexity proxy
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 6 — Persistence layer (all three tables)
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 7 — Stratified frame sampling
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 8 — Condition-vector sampling
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 9 — Stub inference model
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 10 — Win/loss labeling
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 11 — Orchestration script + end-to-end integration test
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 12 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
