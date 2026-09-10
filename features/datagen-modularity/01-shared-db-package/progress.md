## Task 1 — Stand up `database/`, move the shared layer into it
- Status: completed
- Started: 2026-09-10 11:25:52
- Completed: 2026-09-10 11:32:48
- Notes: Clean git mv (100% similarity, zero content diff). ruff/mypy clean. Migration renders identical DDL offline. Test run blocked initially on Postgres not running — user started it, test passed, no stray DB left. Package renamed from `db` to `database` mid-task, before committing.

## Task 2 — Repoint `server/` at `database/`
- Status: completed
- Started: 2026-09-10 11:35:56
- Completed: 2026-09-10 11:38:31
- Notes: pyproject.toml trimmed to reality (deps, packages.find, dead psycopg mypy override). Fixed a plan gap: added server/api/__init__.py placeholder (mypy errors on zero .py files) and removed orphaned server/.env (never assigned to a task's Files list).

## Task 3 — Repoint `training/` at `database/`
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 4 — Update `.claude/coding-guidelines.md`
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
