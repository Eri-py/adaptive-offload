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
- Status: completed
- Started: 2026-09-10 11:38:45
- Completed: 2026-09-10 11:40:12
- Notes: Caught two independent `../server` references in pyproject.toml (doc comment + mypy_path setting, not just one). ruff/mypy clean, 48/48 tests passed (same count as before), no stray test DB.

## Task 4 — Update `.claude/coding-guidelines.md`
- Status: completed
- Started: 2026-09-10 11:40:22
- Completed: 2026-09-10 11:41:17
- Notes: Handled directly (doc-only). Repo layout + Database sections updated for database/; also clarified server/common/ as a distinct future server-internal-helpers location, separate from database/.

## Task 5 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
