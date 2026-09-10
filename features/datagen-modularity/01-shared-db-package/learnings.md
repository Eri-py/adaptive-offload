# Learnings — 01-shared-db-package

## Task 1 — Stand up `database/`, move the shared layer into it

- `git mv` on whole directories (`server/common` → `database/common`, etc.)
  is detected as a 100%-similarity rename by `git diff --staged -M` with
  zero content diff, confirming byte-identical moves and preserved history
  in one step — no need to `git mv` file-by-file.
- `__pycache__/` directories under the moved trees were never git-tracked
  (confirmed with `git ls-files` before moving), so plain `git mv <dir>
  <dir>` on the parent directories was safe and didn't need `--force` or
  manual pycache cleanup first.
- The repo's root `.gitignore` has a bare `.env` entry (no path prefix), so
  it matches `.env` at any depth — `database/.env` is ignored automatically
  with no `.gitignore` edit needed, same as `server/.env` was.
- No Python content changes were needed anywhere: `common.models`/
  `common.testing` imports, and the `.env`-relative-path resolution in
  `database/migrations/env.py` (`Path(__file__).resolve().parent.parent /
  ".env"`) and `database/tests/conftest.py` (same pattern), all resolve
  correctly purely from the new location — confirms the plan's prediction
  exactly.
- `pip install -e ./database[dev]` from the repo root installed clean into
  the shared venv alongside the still-present `server`/`training` editable
  installs. Because `server/common` no longer exists after the move,
  `import common` now resolves unambiguously to `database/common` even
  though `server/pyproject.toml` still lists `common*` in its packages-find
  `include` (harmless until Task 2 updates it — nothing to find there
  anymore).
- `ruff check .` and `mypy .` were clean in `database/` with zero edits to
  the moved files' content, confirming the moved code was already compliant
  with the shared `[tool.ruff]`/`[tool.mypy]` config shape carried over from
  `server/pyproject.toml`.
- `alembic upgrade head --sql` in `database/` renders the exact same DDL as
  before the move (three `CREATE TABLE`s for `scene_complexity`,
  `simulation_runs`, `simulation_results`, plus the `label` enum type and
  `alembic_version` bookkeeping table), fully offline — no DB connection
  attempted, confirming `database/alembic.ini`'s
  `script_location = %(here)s/migrations` and `env.py`'s `.env` path both
  resolve correctly relative to `database/`.
- **Package initially named `db`, renamed to `database` mid-task** (before
  committing) — the top-level directory, `pyproject.toml`'s `[project]
  name`, and this file's own references were all updated together so
  nothing landed in git under the old name. `db.egg-info` (the stale
  editable-install artifact from before the rename) was deleted and the
  package reinstalled fresh into the shared venv under the new name;
  `import common` still resolves correctly afterward (the *importable*
  package is `common`, unaffected by the top-level directory's name).
- Postgres was not running when this task first executed —
  `postgres_engine`'s fixture failed with `OperationalError: connection
  refused` on `127.0.0.1:5432`. Per `CLAUDE.md`'s infrastructure rule, the
  agent does not start Postgres itself; the user started it, and a rerun of
  `pytest -v` then passed (`test_round_trips_all_three_tables`), with no
  stray `test_%` database left behind afterward.
