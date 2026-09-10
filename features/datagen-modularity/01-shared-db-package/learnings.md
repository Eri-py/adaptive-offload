# Learnings — 01-shared-db-package

## Task 1 — Stand up `db/`, move the shared layer into it

- `git mv` on whole directories (`server/common` → `db/common`, etc.) is
  detected as a 100%-similarity rename by `git diff --staged -M` with zero
  content diff, confirming byte-identical moves and preserved history in one
  step — no need to `git mv` file-by-file.
- `__pycache__/` directories under the moved trees were never git-tracked
  (confirmed with `git ls-files` before moving), so plain `git mv <dir>
  <dir>` on the parent directories was safe and didn't need `--force` or
  manual pycache cleanup first.
- The repo's root `.gitignore` has a bare `.env` entry (no path prefix), so
  it matches `.env` at any depth — `db/.env` is ignored automatically with
  no `.gitignore` edit needed, same as `server/.env` was.
- No Python content changes were needed anywhere: `common.models`/
  `common.testing` imports, and the `.env`-relative-path resolution in
  `db/migrations/env.py` (`Path(__file__).resolve().parent.parent / ".env"`)
  and `db/tests/conftest.py` (same pattern), all resolve correctly purely
  from the new location — confirms the plan's prediction exactly.
- `pip install -e ./db[dev]` from the repo root installed clean into the
  shared venv alongside the still-present `server`/`training` editable
  installs. Because `server/common` no longer exists after the move,
  `import common` now resolves unambiguously to `db/common` even though
  `server/pyproject.toml` still lists `common*` in its packages-find
  `include` (harmless until Task 2 updates it — nothing to find there
  anymore).
- `ruff check .` and `mypy .` were clean in `db/` with zero edits to the
  moved files' content, confirming the moved code was already compliant
  with the shared `[tool.ruff]`/`[tool.mypy]` config shape carried over from
  `server/pyproject.toml`.
- `alembic upgrade head --sql` in `db/` renders the exact same DDL as before
  the move (three `CREATE TABLE`s for `scene_complexity`, `simulation_runs`,
  `simulation_results`, plus the `label` enum type and `alembic_version`
  bookkeeping table), fully offline — no DB connection attempted, confirming
  `db/alembic.ini`'s `script_location = %(here)s/migrations` and `env.py`'s
  `.env` path both resolve correctly relative to `db/`.
- **Blocker encountered:** `pytest -v` in `db/` failed at the
  `postgres_engine` fixture with `OperationalError: connection refused` on
  `127.0.0.1:5432` — Postgres was not running in this environment (no
  listener on 5432, no `postgres` process, no `docker` daemon available in
  this WSL2 distro). Per `CLAUDE.md`'s infrastructure rule, the agent must
  never start Postgres itself, even to unblock a test run — this needs the
  user to start their Postgres instance before the moved
  `test_round_trips_all_three_tables` test (and its ephemeral-database
  create/drop) can be verified. Once Postgres is running, rerun `cd db &&
  pytest -v` and then check `SELECT datname FROM pg_database WHERE datname
  LIKE 'test_%'` via `POSTGRES_ADMIN_URL` to confirm no stray database was
  left behind.
