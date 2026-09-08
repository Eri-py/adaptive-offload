# Learnings — data-gen-router-loop / 01-datagen-simulator

## Task 1 — `server/common` DB scaffolding

- **Import path is `common.*`, not `server.common.*`.** `server/pyproject.toml`
  declares `[tool.setuptools.packages.find] include = ["common*", "api*"]`
  with no `server/__init__.py`. That means `server/` itself is the import
  root (like an unprefixed `src/` layout) — code inside `server/` (and later
  `training/`, after `pip install -e ../server`) imports as
  `from common.models import ...` / `from common.db import ...`, never
  `from server.common...`. Get this wrong in a later task and imports will
  resolve at editor-time but fail at runtime once actually installed.
- **pytest package discovery works without extra config.** Because
  `server/tests/__init__.py` and `server/tests/common/__init__.py` exist but
  `server/__init__.py` does not, pytest's rootdir-insertion logic adds
  `server/` (the first ancestor lacking `__init__.py`) to `sys.path`. No
  `conftest.py` or `pythonpath` setting was needed for `from common.models
  import ...` to resolve when running `pytest` from `server/`.
- **`mypy --strict` is happy with SQLAlchemy 2.0's `Mapped`/`mapped_column`
  style out of the box** — no `sqlalchemy[mypy]` plugin or extra `mypy.ini`
  plugin config was needed for the three simple models in this task.
- **Ruff's `UP017` fires on `datetime.now(timezone.utc)`** on this
  target-version/Python combo — it wants `datetime.now(UTC)` via the
  `datetime.UTC` alias (`from datetime import UTC`). Worth remembering for
  any other module in this feature that stamps UTC timestamps.
- Used a Python-side `default=` callable (`_utcnow`) for `created_at`/
  `computed_at` rather than a DB `server_default=func.now()` — keeps
  timestamp generation portable across the in-memory SQLite used in tests
  and the real Postgres instance without relying on either dialect's clock
  function.
- `Label` win/loss values are a proper `enum.Enum` (`LOCAL`/`OFFLOAD`) mapped
  via SQLAlchemy's `Enum` type, not a bare `String`, so a typo'd label value
  is a type error / DB constraint violation rather than silently accepted.

## Task 2 — Alembic migration

- **`alembic init migrations` also drops a `migrations/README` file** that
  wasn't in the task's `Files` list — deleted it after init rather than
  leaving an unrequested extra file in the diff.
- **Autogenerate can build the migration for you without ever touching the
  real Postgres instance.** Wired `migrations/env.py`'s `target_metadata` to
  `common.models.Base.metadata` (imported as `from common.models import
  Base` — per Task 1's `common.*`-not-`server.common.*` import-root
  learning), then *temporarily* pointed `alembic.ini`'s `sqlalchemy.url` at a
  disposable local sqlite file in the scratchpad dir, ran
  `alembic revision --autogenerate`, and restored `alembic.ini`'s real
  placeholder afterward. This diffs the metadata against an empty schema and
  emits accurate `op.create_table` calls (correct types, PK/FK constraints)
  without connecting to — or needing — the project's actual database, and
  without ever running `upgrade head` (only `revision --autogenerate`, which
  never applies anything). Far less error-prone than hand-transcribing every
  column from `models.py`.
- **`sa.Enum(Label)` in the model renders as `sa.Enum('LOCAL', 'OFFLOAD',
  name='label')` in the migration** — SQLAlchemy's plain `Enum(SomeEnum)`
  column type stores/compares on the enum *member name*, not `.value`,
  unless `values_callable` is passed. So the Postgres `label` enum type ends
  up with `'LOCAL'`/`'OFFLOAD'` as its DB-level values, not the lowercase
  `.value` strings (`"local"`/`"offload"`) — this is intentional/consistent
  with the model, not a mismatch to fix.
- **`alembic.ini`'s placeholder `sqlalchemy.url` should already be a
  `postgresql+psycopg://...` URL, not the generic `driver://...` template
  default.** `--sql` (offline) mode only uses the URL to pick the rendering
  dialect, but a non-real dialect scheme falls back to generic/ANSI SQL
  instead of Postgres-specific DDL (e.g. `SERIAL`, proper `CREATE TYPE ...
  AS ENUM`). Kept a `postgresql+psycopg://user:pass@localhost:5432/dbname`
  placeholder plus an `env.py` override that prefers `DATABASE_URL` (mirrors
  `common/db.py`'s convention) whenever it's set, so a real invocation never
  needs to edit `alembic.ini` at all.
- `alembic upgrade head --sql` is the only alembic command run against this
  migration in this task — its log line `Generating static SQL` (vs.
  `Running upgrade` against a live connection) is the tell that no database
  connection was opened; worth checking for that line as a sanity check
  whenever verifying an offline-mode run.

## Task 1b — Redo Task 1's test against real Postgres (corrective)

- **Reusable ephemeral-DB helper lives in `server/common/testing.py`,
  independent of pytest.** `ephemeral_postgres_database(admin_url: str) ->
  Iterator[Engine]` is a plain `@contextmanager`, not a fixture — it doesn't
  import `pytest` at all. `server/tests/conftest.py` wraps it in a
  function-scoped `postgres_engine` fixture; `training/tests/conftest.py`
  (Task 6, later, once it depends on `server/common` via an editable install)
  should do the exact same wrap — `from common.testing import
  ephemeral_postgres_database`, read its own admin URL, `with
  ephemeral_postgres_database(admin_url) as engine: yield engine`. No new
  dependency duplication needed beyond `python-dotenv` in each package's own
  dev deps.
- **`CREATE DATABASE`/`DROP DATABASE` require an autocommit connection** —
  `create_engine(admin_url, isolation_level="AUTOCOMMIT")`. Without it,
  Postgres rejects both statements because they can't run inside a
  transaction block.
- **The admin engine must connect to a maintenance DB (`postgres`), never the
  ephemeral DB itself**, and the ephemeral DB's own engine must be
  `.dispose()`d *before* the `DROP DATABASE` runs — otherwise Postgres
  refuses to drop a database with open connections. Structured as nested
  `try`/`finally`: inner `finally` disposes the test engine and drops the DB;
  outer `finally` disposes the admin engine. Verified teardown actually fires
  on a failing test too (added a throwaway `assert False` test using the
  fixture, ran it, confirmed via `pg_database` that no `test_*` row survived)
  — this is the scenario a bare `yield` fixture without `try`/`finally` would
  get wrong.
- **`server/.env`'s `POSTGRES_ADMIN_URL` is a bare `postgresql://` URL**, which
  makes SQLAlchemy default to the `psycopg2` dialect/driver — not installed,
  since `server/pyproject.toml` depends on `psycopg[binary]` (v3). Added a
  small `_with_psycopg_driver()` normalizer in `testing.py` that rewrites
  `postgresql://` → `postgresql+psycopg://` before `create_engine`, rather
  than requiring `.env` to spell out the driver. Anyone adding a
  `training/.env` admin URL later will likely hit the same thing.
- **This is the real payoff of leaving SQLite:** the original SQLite-backed
  test passed with `session.add_all([run, result, complexity])` even though
  `SimulationResult` has an FK to `SimulationRun` and nothing declares an ORM
  `relationship()` between the two mapped classes — SQLite doesn't enforce FK
  constraints by default (no `PRAGMA foreign_keys=ON`), so the insert order
  never mattered. Postgres enforces the FK immediately and the same
  `add_all` call raised `IntegrityError: ForeignKeyViolation` because the
  unit of work has no relationship info to infer that `simulation_runs` must
  insert before `simulation_results`. Fix (in the test, not the models):
  `session.add(run); session.flush()` before adding `result`/`complexity`.
  Any future test that inserts rows across FK'd tables without a declared
  `relationship()` needs to either flush the parent first or add one in
  dependency order — `add_all` alone won't reorder for you.
- Confirmed no stray `test_*` database survives a normal passing run, a
  failing run, and back-to-back repeated runs — checked via a one-off script
  connecting with the admin URL and querying `pg_database WHERE datname LIKE
  'test_%'`.

## Task 3 — `training/datagen` package + config module

- **`from common.db import get_engine` and `from common.models import ...`
  resolve cleanly from `training/`'s own venv** once `pip install -e
  ../server` runs before `pip install -e ".[dev]"` in that venv — actually
  verified with a one-off `.venv/bin/python -c "from common.db import
  get_engine; from common.models import Base; from common.testing import
  ephemeral_postgres_database"` inside `training/`, not just inferred from
  the dependency line. This is the load-bearing detail Task 6 onward
  depends on, and it works with the plain two-step `pip install` sequence —
  no path hacks, no `conftest.py` sys.path tweaks needed.
- **Plain PEP 508 `dependencies` in `training/pyproject.toml` has no
  portable way to express "install `../server` too."** A relative
  `file://` URL isn't standard, and `${PROJECT_ROOT}` substitution in
  dependency strings is a `uv`-specific `tool.uv.sources` feature, not
  something plain `pip install -e .` understands — using it would silently
  fail to resolve under plain pip. Went with the spec's documented
  alternative instead: `server` is *not* listed in
  `training/pyproject.toml`'s `dependencies` at all; a comment above the
  `dependencies` list in `training/pyproject.toml` spells out the two-step
  install (`pip install -e ../server` then `pip install -e ".[dev]"`).
  Mirrors how `server/`'s own venv was set up (plain `python -m venv` +
  `pip install -e ".[dev]"`, no lockfile), so both packages' install
  stories stay consistent.
- **`training/`, `training/datagen/`, `training/router/`, and
  `training/data/` already existed as empty directories** from the initial
  scaffolding commit (untracked, not yet in git) — `training/data/` is
  already covered by a `training/data/` entry in the root `.gitignore`
  (dataset cache, per Task 4/5's future downloads), and the existing
  generic `.venv/`, `__pycache__/`, `*.egg-info/`, `.mypy_cache/`,
  `.pytest_cache/`, `.ruff_cache/` patterns already cover `training/`'s new
  venv/caches too — no `.gitignore` edit was needed for this task.
- **`config.PRESETS`'s value type is a `TypedDict`
  (`ConditionPresetRanges`)**, not a plain `dict[str, tuple[float,
  float]]`, so every preset is statically checked to define exactly the
  four required axis keys. The one wrinkle: iterating over axis names as
  plain `str` and subscripting the TypedDict with that variable
  (`ranges[axis]`) trips mypy's `literal-required` check (TypedDict
  subscripts normally need a literal key) — the test that does this
  (`test_config.py`) has a targeted `# type: ignore[literal-required]` on
  that one line rather than loosening the TypedDict itself.
- **The Task 3 plan entry's `Files` list doesn't name a `test_config.py`**
  (only the `tests/__init__.py` package markers), but the task's own
  success criteria explicitly requires "a test importing `config`" —
  added `training/tests/datagen/test_config.py` to satisfy that criterion
  directly, since it's clearly implied by the success criteria rather than
  scope creep.
- Illustrative stub-model coefficients (base latency/accuracy, noise
  std-devs, device-load/bandwidth/packet-loss coefficients) are unmeasured
  placeholders per the spec's stub-first stance — Task 9's stub-inference
  formula is what actually gives them meaning; revisit their magnitudes
  then if the resulting latency/accuracy numbers don't look plausible.
