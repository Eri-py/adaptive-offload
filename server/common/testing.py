"""Ephemeral-Postgres-database helper shared by `server/` and `training/` test suites.

Both packages need the identical create/drop-a-disposable-database pattern to
test the shared models against real Postgres instead of a SQLite stand-in.
`training/` depends on `server/common` via an editable install, so the helper
lives here to avoid duplicating create/drop logic in two independent
packages. This is covered by the scoped exception in the repo's `CLAUDE.md`
("Infrastructure") — fixture-driven create/drop of disposable, uniquely-named
test databases only. It never touches the real/shared database and never runs
a migration; schema setup here is plain `Base.metadata.create_all`.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text

from common.models import Base


def _with_psycopg_driver(url: str) -> str:
    """Force the `psycopg` (v3) driver for a bare `postgresql://` URL.

    `server/`'s runtime dependency is `psycopg[binary]` (v3), not the `psycopg2`
    SQLAlchemy defaults a driver-less URL to, so admin URLs from `.env` need
    this normalization to avoid depending on a driver that isn't installed.
    """
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


@contextmanager
def ephemeral_postgres_database(admin_url: str) -> Iterator[Engine]:
    """Create a uniquely-named disposable Postgres database, yield an engine for it.

    `admin_url` must point at a maintenance database (e.g. Postgres's `postgres`
    database) with a role authorized to `CREATE DATABASE`/`DROP DATABASE`. The
    new database is created, has all tables from `common.models.Base` applied
    to it via `Base.metadata.create_all`, and is dropped again on exit —
    including when the caller raises, so a failing test never leaks a database.
    """
    admin_url = _with_psycopg_driver(admin_url)
    db_name = f"test_{uuid.uuid4().hex[:12]}"
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))

        test_url = admin_url.rsplit("/", 1)[0] + f"/{db_name}"
        test_engine = create_engine(test_url)
        try:
            Base.metadata.create_all(test_engine)
            yield test_engine
        finally:
            test_engine.dispose()
            with admin_engine.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
    finally:
        admin_engine.dispose()
