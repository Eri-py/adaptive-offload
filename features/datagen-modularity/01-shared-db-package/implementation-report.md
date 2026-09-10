# Shared DB Package — Implementation Report

**Feature:** Shared DB Package
**Directory:** `features/datagen-modularity/01-shared-db-package/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Stand up `database/`, move the shared layer into it | completed |
| Task 2 — Repoint `server/` at `database/` | completed |
| Task 3 — Repoint `training/` at `database/` | completed |
| Task 4 — Update `.claude/coding-guidelines.md` | completed |
| Task 5 — Regression test run | completed |

## Files changed

### Added

- `database/pyproject.toml`
- `server/api/__init__.py`
- `features/datagen-modularity/01-shared-db-package/spec.md`
- `features/datagen-modularity/01-shared-db-package/implementation.md`
- `features/datagen-modularity/02-datagen-cli-tools/spec.md` (companion spec, not implemented by this plan)

### Moved (100%-similarity renames, zero content diff)

- `server/alembic.ini` → `database/alembic.ini`
- `server/common/__init__.py` → `database/common/__init__.py`
- `server/common/db.py` → `database/common/db.py`
- `server/common/models.py` → `database/common/models.py`
- `server/common/testing.py` → `database/common/testing.py`
- `server/migrations/env.py` → `database/migrations/env.py`
- `server/migrations/script.py.mako` → `database/migrations/script.py.mako`
- `server/migrations/versions/0001_create_simulation_tables.py` → `database/migrations/versions/0001_create_simulation_tables.py`
- `server/tests/__init__.py` → `database/tests/__init__.py`
- `server/tests/common/__init__.py` → `database/tests/common/__init__.py`
- `server/tests/common/test_models.py` → `database/tests/common/test_models.py`
- `server/tests/conftest.py` → `database/tests/conftest.py`

### Edited

- `.claude/coding-guidelines.md` — repo layout and Database sections updated for `database/`; `server/common/` clarified as a distinct, not-yet-created location for future server-internal (non-DB) helpers.
- `server/pyproject.toml` — dropped the four dependencies that moved (`sqlalchemy`, `psycopg[binary]`, `alembic`, `python-dotenv`), dropped `common*` from `packages.find`, dropped the now-dead `psycopg.*` mypy override, added a comment documenting the future `pip install -e ../database` dependency.
- `training/pyproject.toml` — retargeted the `-e ../server` dependency comment and the `[tool.mypy]` `mypy_path` setting to `../database`.
- `training/tests/conftest.py` — docstring updated to reference `database/tests/conftest.py` instead of `server/tests/conftest.py`.

### Removed (untracked, not part of the git diff)

- `server/.env` — orphaned once nothing in `server/` read it anymore; trivially recreatable when `server/api/` needs `DATABASE_URL` for real.

Not tracked in git but created during execution: `database/.env` (gitignored), copied from `server/.env`'s prior content, holding `POSTGRES_ADMIN_URL`/`DATABASE_URL` for `database/`'s own test fixture and migration.

## Tests

No new test files were added — the plan's purpose was a structural move, not new coverage. The one existing DB round-trip test moved along with the code it tests:

- `database/tests/common/test_models.py` (1 test, moved from `server/tests/common/test_models.py`, unchanged): round-trip insert/read across all three ORM models plus the FK link, against a real ephemeral Postgres database.

`training/`'s existing 48 tests (unchanged in content, count, or behavior) continue to pass, now resolving `common.*` imports from `database/` instead of `server/`.

**49 tests passing** (1 in `database/`, 48 in `training/`, 0 in `server/` — expected, since `server/` has no code yet), 0 skipped.

## Commits

`feature/data-gen-router-loop..feature/datagen-modularity` (9 commits): from `066a5e7` (Add feature specs) through `03b399b` (Task 5: regression test run).

## Notable events

- **Mid-task rename:** the package was originally planned and initially built as `db/`. Partway through Task 1, before anything was committed under that name, the user asked for it to be renamed to `database/`. The directory, `pyproject.toml`'s `[project] name`, and `learnings.md`'s references were all updated together so nothing landed in git history under the old name. One staging mistake during the rename (a `learnings.md` edit that didn't get included in either of the next two commits) was caught and fixed in a small follow-up commit.
- **Two plan gaps surfaced and were fixed directly by the orchestrator** (not delegated, consistent with how small justified fixes were handled in the prior feature): `server/.env` was named for removal in the plan's "Approach & Key Decisions" but never assigned to any task's `Files` list; and `mypy .` on `server/`'s now-fully-empty tree exits 2 ("no .py[i] files") rather than passing trivially, which the plan hadn't anticipated. Fixed by removing the orphaned `.env` and adding a minimal `server/api/__init__.py` placeholder (consistent with the repo layout `coding-guidelines.md` already documents for `server/api/`).
- **Task 1's live-database verification was blocked once**, waiting on the user to start their local Postgres instance (the agent correctly declined to start it itself, per `CLAUDE.md`). Once started, the test passed and no stray ephemeral database was left behind.
