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

## Task 2 — Repoint `server/` at `database/`

- Removed `sqlalchemy`, `psycopg[binary]`, `alembic`, `python-dotenv` from
  `server/pyproject.toml`'s `dependencies` (now `[]`), removed `common*`
  from `[tool.setuptools.packages.find]`'s `include` (now just `["api*"]`),
  and added a comment mirroring `training/pyproject.toml`'s existing
  `../server` → `common` comment style, pointed at `pip install -e
  ../database` instead, for when `server/api/` is built and needs
  `common.db`/`common.models`.
- Also removed the now-dead `[[tool.mypy.overrides]] module = "psycopg.*"`
  block from `server/pyproject.toml` — it was config for a dependency that
  no longer exists in this package, so leaving it in would have
  contradicted the task's own goal of reducing `server/` to its actual
  scope. This wasn't in the task's literal enumerated edit list but is the
  same file already being edited, so it didn't expand the `Files` scope.
- `pip install -e ./server[dev]` from the repo root still succeeds cleanly
  with `dependencies = []` — pip has no problem with an empty
  editable-install dependency list.
- **`mypy .` failed "clean" in `server/` (exit 2, `There are no .py[i]
  files in directory '.'`)** — the subagent correctly flagged this as
  out of its `Files` scope rather than fixing it silently. Resolved by the
  orchestrator directly (not delegated, per the same small-justified-fix
  pattern used for the earlier mypy `python_version` fix in feature 01):
  added `server/api/__init__.py` (empty placeholder) — matches the repo
  layout `.claude/coding-guidelines.md` already documents for `server/api/`
  and gives mypy one real file to trivially pass against instead of
  erroring on an empty tree. `ruff check .` and `mypy .` both clean
  afterward (1 source file).
- Also removed `server/.env` directly (orchestrator, not a subagent task) —
  the plan's "Approach & Key Decisions" said this file should go since
  nothing in `server/` reads it anymore, but no task's `Files` list actually
  included it, a gap in the plan itself. Untracked/gitignored, trivially
  recreatable whenever `server/api/` needs `DATABASE_URL` for real.

## Task 3 — Repoint `training/` at `database/`

- `training/pyproject.toml` had two independent `../server` references, not
  one: the `pip install -e ../server` comment block (functional-adjacent
  documentation) and, separately, `[tool.mypy]`'s `mypy_path = "../server"`
  (a real static-analysis setting, not a comment) — both needed updating to
  `../database`, and both were legitimately in scope since they're the same
  file the task's `Files` list already named. Easy to fix only the doc
  comment and miss `mypy_path` since it's a short single-line setting buried
  after the `dependencies` list.
- Only doc/comment text changed in both files — `training/pyproject.toml`'s
  `dependencies` list and `[tool.setuptools.packages.find]` were already
  untouched-by-`server` (training never listed `server`/`database` as a PEP
  508 dependency, consistent with Task 1/2's note that plain pip has no
  portable relative-path syntax for this), so no functional dependency edits
  were needed here, only the `mypy_path` value and the two docstrings/
  comments.
- `pip install -e ./training[dev]` from the repo root re-registers the
  editable install cleanly (uninstalls and reinstalls training 0.1.0) with
  no changes needed beyond the doc-comment edit — confirms the task's
  framing that this reinstall step is precautionary, not required by any
  dependency-list change.
- `mypy_path = "../database"` resolved `common.*` correctly on the first
  `mypy .` run after the edit — no residual caching or stale-path issue from
  the prior `../server` value (mypy re-reads `pyproject.toml` fresh each
  invocation, no `.mypy_cache` staleness observed here).
- Full suite: 48 passed (same as pre-Task-3), including all Postgres-backed
  tests in `test_persistence.py` and `test_run_simulation.py`, confirming
  `common.*` resolves correctly at both runtime (via the shared venv's
  `database` editable install) and statically (via `mypy_path`) now that
  `training/` points at `database/` instead of `server/` everywhere.
- To query `pg_database` directly for stray `test_%` databases (rather than
  just trusting the fixture's own teardown), the admin URL in
  `training/.env` needs its driver swapped: `POSTGRES_ADMIN_URL` is a plain
  `postgresql://` URL, but the shared venv only has `psycopg` (v3) installed,
  not `psycopg2` — `create_engine()` on the raw URL fails with
  `ModuleNotFoundError: No module named 'psycopg2'` since SQLAlchemy defaults
  unprefixed `postgresql://` URLs to the psycopg2 dialect. Rewriting the
  scheme to `postgresql+psycopg://` before calling `create_engine()` fixes
  it. (`common.testing.ephemeral_postgres_database` presumably already does
  this internally, since the fixture-driven tests connect fine without any
  such rewrite.)

## Review fixes — S1, N1, N2 (stale `server/`-era text in `database/tests/`)

- Confirmed the pattern from Task 3's learning above: `POSTGRES_ADMIN_URL`
  in `database/.env` is also a bare `postgresql://` URL, so verifying "no
  stray `test_%` database" post-run needed the same
  `.replace("postgresql://", "postgresql+psycopg://")` rewrite before
  `create_engine()` — the raw URL fails with `ModuleNotFoundError: No
  module named 'psycopg2'` in this venv, same root cause as before, just
  recurring in a different feature directory (`database/` instead of
  `training/`). Worth considering whether `database/.env`'s
  `POSTGRES_ADMIN_URL` should just be written with `+psycopg` from the
  start to stop this from tripping up every ad hoc verification script.
- All three findings were pure text (docstring/error-message) edits with no
  logic changes, exactly as scoped — `ruff check .` and `mypy .` stayed
  clean with zero other changes needed, and the existing single test
  (`test_round_trips_all_three_tables`) still passed unmodified.
