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
