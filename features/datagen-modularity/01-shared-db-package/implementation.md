# Shared DB Package — Implementation Plan

## Summary

Move `server/common/` (models, engine, the ephemeral-test-DB fixture helper)
and `server/migrations/` (Alembic config + the one existing migration) out of
`server/` into a new top-level `database/` package with its own manifest. `server/`
and `training/` both depend on `database/` the same way, via editable install —
neither owns it. This is a structural move: no schema change, no behavior
change, same tests passing afterward.

## Approach & Key Decisions

- The inner Python package stays named `common` (`database/common/models.py`,
  `database/common/db.py`, `database/common/testing.py`) — only its parent directory
  moves from `server/` to `database/`. This means every existing
  `from common.models import ...` / `from common.db import ...` /
  `from common.testing import ...` statement across `server/` and
  `training/` needs **zero changes** — only each package's editable-install
  target changes (`-e ../server` → `-e ../database`), the same low-churn pattern
  `training/pyproject.toml` already documents for its dependency on
  `server/common` today.
- `server/tests/conftest.py` and `server/tests/common/test_models.py` move
  to `database/tests/conftest.py` and `database/tests/common/test_models.py` — they
  test the models/fixture that now live in `database/`, so they belong there, not
  in `server/`. Their `.env`-loading path (`Path(__file__).resolve()
  .parent.parent / ".env"`) is computed relative to each file's own
  location, so it automatically resolves to `database/.env` after the move with
  no code change — same for `database/migrations/env.py`'s existing `.env` load.
- After this move, `server/` has no code of its own — `server/api/` (the
  future FastAPI app) doesn't exist yet, and everything `server/` currently
  contains is the shared DB layer that's moving out. `server/tests/`,
  `server/common/`, `server/migrations/`, and `server/alembic.ini` are all
  removed; `server/pyproject.toml` is trimmed to no longer declare
  dependencies (`sqlalchemy`, `psycopg`, `alembic`, `python-dotenv`) or
  `packages.find` entries (`common*`) that belonged to the code that moved.
  This isn't a regression — `server/` genuinely hasn't started yet, and this
  correctly reflects that instead of masking it with borrowed code.
- `database/.env` (gitignored, not committed — created directly by the
  orchestrator during execution, same as `server/.env`/`training/.env` were
  originally) becomes the new home for `POSTGRES_ADMIN_URL` and
  `DATABASE_URL`, since `database/`'s own test fixture and migration now resolve
  their `.env` path relative to `database/`. `training/.env` is untouched — it's
  still needed for `training/`'s own test fixture and the simulator CLI's
  real `DATABASE_URL` use. `server/.env` is removed since nothing in
  `server/` reads it anymore; trivially recreated once `server/api/` needs
  it.
- Per `CLAUDE.md`'s infrastructure rule, nothing in this plan touches the
  real/shared Postgres instance or applies the migration — the migration is
  only ever verified via `alembic upgrade head --sql` (offline rendering),
  same as every prior task involving it.
- The shared root venv (`.venv/`, per the earlier decision to use one venv
  for both packages) gets `database` added as a third editable install alongside
  `server`/`training` — this is orchestration housekeeping done during task
  verification, not a tracked file change.

## Out of Scope

Everything the spec excludes: any schema change, any change to
`server/api/` (doesn't exist) or `training/datagen/`'s simulation logic,
applying the migration or touching the real Postgres instance, and the
config/presets split and new CLI tools (the sibling 02 spec).

## Dependencies and Configuration

- New `database/pyproject.toml`: `sqlalchemy`, `psycopg[binary]`, `alembic`,
  `python-dotenv` (runtime — moved as-is from `server/pyproject.toml`);
  `ruff`, `mypy`, `pytest` (dev).
- New `database/.env` (gitignored, not committed): `POSTGRES_ADMIN_URL` and
  `DATABASE_URL`, same values currently in `server/.env`.
- `server/pyproject.toml` loses the above runtime dependencies and the
  `common*` entry in `packages.find`; gains a comment documenting the
  editable dependency on `../database` (mirroring `training/pyproject.toml`'s
  existing comment about depending on `server`, now retargeted).
- `training/pyproject.toml`'s existing comment about installing `../server`
  is updated to reference `../database` instead.
- Shared venv: reinstall with `database` added
  (`pip install -e ./database[dev] -e ./server[dev] -e ./training[dev]` from the
  repo root).
- No schema/migration changes. No live database touched.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `database/pyproject.toml` | add | `database` package manifest | new package needs its own manifest, moved from `server/pyproject.toml`'s relevant entries |
| `database/common/__init__.py` | move (from `server/common/__init__.py`) | package marker | — |
| `database/common/models.py` | move (from `server/common/models.py`) | ORM models | — |
| `database/common/db.py` | move (from `server/common/db.py`) | engine/session factory | — |
| `database/common/testing.py` | move (from `server/common/testing.py`) | ephemeral-test-DB fixture helper | — |
| `database/alembic.ini` | move (from `server/alembic.ini`) | migration config | — |
| `database/migrations/env.py` | move (from `server/migrations/env.py`) | migration environment | `.env` path auto-resolves to `database/.env` after the move, no edit needed |
| `database/migrations/script.py.mako` | move (from `server/migrations/script.py.mako`) | migration template | — |
| `database/migrations/versions/0001_create_simulation_tables.py` | move (from `server/migrations/versions/0001_create_simulation_tables.py`) | the migration | — |
| `database/tests/__init__.py` | move (from `server/tests/__init__.py`) | package marker | — |
| `database/tests/conftest.py` | move (from `server/tests/conftest.py`) | ephemeral-DB pytest fixture | `.env` path auto-resolves to `database/.env`, no edit needed |
| `database/tests/common/__init__.py` | move (from `server/tests/common/__init__.py`) | package marker | — |
| `database/tests/common/test_models.py` | move (from `server/tests/common/test_models.py`) | model round-trip test | — |
| `server/pyproject.toml` | edit | drop moved deps + `common*` from `packages.find`; add `database` editable-dependency comment | reflects `server/`'s actual (currently empty) scope |
| `training/pyproject.toml` | edit | retarget the `-e ../server` comment to `-e ../database` | dependency moved |
| `training/tests/conftest.py` | edit | update docstring reference from `server/tests/conftest.py` to `database/tests/conftest.py` | comment accuracy only, no behavior change |
| `.claude/coding-guidelines.md` | edit | document `database/`'s location and that `server/`/`training/` depend on it symmetrically | spec requirement |

## Tasks

### Task 1 — Stand up `database/`, move the shared layer into it

- **Objective:** Create the `database/` package containing everything that currently lives in `server/common/` and `server/migrations/`, and get it building/testing standalone.
- **Files:** `database/pyproject.toml`, `database/common/__init__.py`, `database/common/models.py`, `database/common/db.py`, `database/common/testing.py`, `database/alembic.ini`, `database/migrations/env.py`, `database/migrations/script.py.mako`, `database/migrations/versions/0001_create_simulation_tables.py`, `database/tests/__init__.py`, `database/tests/conftest.py`, `database/tests/common/__init__.py`, `database/tests/common/test_models.py`
- **Details:** Use `git mv` for every file (preserves history) — `server/common/` → `database/common/`, `server/migrations/` → `database/migrations/`, `server/alembic.ini` → `database/alembic.ini`, `server/tests/*` → `database/tests/*`. None of the moved files' Python content needs to change (import statements stay `common.*`; `.env`-relative-path code in `database/migrations/env.py` and `database/tests/conftest.py` resolves correctly from their new location automatically). Create `database/pyproject.toml` by adapting `server/pyproject.toml`'s current content: keep `sqlalchemy`, `psycopg[binary]`, `alembic`, `python-dotenv` as runtime deps and `ruff`/`mypy`/`pytest` as dev deps; `[tool.setuptools.packages.find]` should be `include = ["common*"]` (no `api*` — that's `server/`'s concern, not `database/`'s); keep the same `[tool.ruff]`/`[tool.mypy]` config shape as `server/pyproject.toml` currently has (including the `psycopg.*` mypy override). Create `database/.env` directly (gitignored, not committed) with the same `POSTGRES_ADMIN_URL`/`DATABASE_URL` values currently in `server/.env`.
- **Success criteria:**
  - `database/` installs cleanly into the shared venv (`pip install -e ./database[dev]` from the repo root)
  - `cd database && ruff check . && mypy .` clean
  - `cd database && pytest -v` — the moved model round-trip test passes against a real ephemeral Postgres database (same as it did in `server/` before the move), and no stray test database survives
  - `cd database && alembic upgrade head --sql` renders the same DDL as before the move, without connecting to any database

### Task 2 — Repoint `server/` at `database/`

- **Objective:** Reduce `server/` to its actual current scope (nothing yet — `server/api/` doesn't exist) now that the shared DB layer has moved out.
- **Files:** `server/pyproject.toml`
- **Details:** `server/common/`, `server/migrations/`, `server/alembic.ini`, and `server/tests/` no longer exist after Task 1's moves — nothing further to delete here. Edit `server/pyproject.toml`: remove `sqlalchemy`, `psycopg[binary]`, `alembic`, `python-dotenv` from `dependencies` (they belonged to the code that moved); remove `common*` from `[tool.setuptools.packages.find]`'s `include` (keep `api*`); add a comment (mirroring `training/pyproject.toml`'s existing style) documenting that `server/` depends on `database/common` via `pip install -e ../database`, for whenever `server/api/` is built and needs it.
- **Success criteria:**
  - `pip install -e ./server[dev]` from the repo root still succeeds (an empty-but-valid package)
  - `cd server && ruff check . && mypy .` clean (trivially — no source files yet)

### Task 3 — Repoint `training/` at `database/`

- **Objective:** Update `training/`'s dependency reference from `server` to `database`, and confirm every existing `common.*` import still resolves correctly from the new location.
- **Files:** `training/pyproject.toml`, `training/tests/conftest.py`
- **Details:** Edit `training/pyproject.toml`'s comment block (currently documenting `pip install -e ../server`) to reference `../database` instead — no dependency *list* changes, this is a comment-accuracy edit, since `server` was never a PEP 508 dependency here in the first place (same reasoning as before, now pointed at `database`). Edit `training/tests/conftest.py`'s docstring, which currently says it "Mirrors `server/tests/conftest.py` exactly" — update to reference `database/tests/conftest.py`, the fixture's new home. No functional code changes in either file.
- **Success criteria:**
  - `pip install -e ../database[dev]` then `pip install -e ./training[dev]` from `training/` (or the equivalent from the repo root) succeeds
  - `cd training && ruff check . && mypy .` clean
  - `cd training && pytest -v` — full existing suite passes unchanged (same test count as before this plan), including the Postgres-backed tests (`test_persistence.py`, `test_run_simulation.py`) which now resolve `common.*` from `database/` instead of `server/`
  - No stray ephemeral test database survives the run

### Task 4 — Update `.claude/coding-guidelines.md`

- **Objective:** Document the new `database/` package and its symmetric relationship to `server/`/`training/`.
- **Files:** `.claude/coding-guidelines.md`
- **Details:** Update the repo-layout section: add `database/` as a top-level area (models, engine, migrations, the test-DB fixture helper), and revise the existing `server/common/` description to reflect that this code no longer lives under `server/` — `server/` and `training/` both depend on `database/` the same way, via editable install, neither owning it.
- **Success criteria:**
  - The repo-layout section accurately describes `database/`'s location and contents, and no longer claims `server/common/` owns the DB layer

### Task 5 — Regression test run

- **Objective:** Run every test across `database/`, `server/`, and `training/` and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd database && ruff check . && mypy . && pytest` passes
  - `cd server && ruff check . && mypy . && pytest` passes
  - `cd training && ruff check . && mypy . && pytest` passes
  - `cd database && alembic upgrade head --sql` still renders correctly offline
