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
