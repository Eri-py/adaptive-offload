# Learnings — frame-feature-extraction

## Task 1 — `frame_features` table and migration

- `database/tests/common/test_models.py` has exactly one test function that
  round-trips every table in `models.py` in a single Postgres session/fixture
  instance. Adding a new table means extending that same test (rename it to
  keep "all N tables" accurate), not adding a new test function — there's
  only one test file in `database/tests/`, so `pytest` reporting "1 passed"
  after adding a table is expected, not a sign something was skipped.
- Offline render (`alembic upgrade 0002:0003 --sql` from `database/`, via
  `../.venv/bin/python -m alembic ...`) is a good pre-check but only
  validates DDL shape/ordering against the model — it never opens a real
  connection, so it can't catch the enum `create_type` class of bug that
  migration `0002`'s docstring describes. Not relevant here (no enum
  columns), but worth remembering for any future migration that adds one.
