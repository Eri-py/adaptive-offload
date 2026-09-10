"""Shared pytest fixtures for `server/tests/`.

Provides a fresh, disposable Postgres database per test so model/query tests
run against real Postgres rather than a SQLite stand-in (SQLite's dialect
differs enough on native `ENUM`, JSON columns, and FK enforcement to not be
trustworthy here). Covered by the scoped exception in `CLAUDE.md`'s
Infrastructure section: fixture-driven create/drop of ephemeral, uniquely
named test databases only.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import Engine

from common.testing import ephemeral_postgres_database

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    """Function-scoped engine bound to a freshly created, uniquely-named test DB.

    Function scope (a fresh database per test) keeps tests fully isolated from
    one another, which matters more at this suite's size than the extra
    setup/teardown cost of a broader scope.
    """
    admin_url = os.environ.get("POSTGRES_ADMIN_URL")
    if not admin_url:
        raise RuntimeError(
            "POSTGRES_ADMIN_URL is not set. Add it to server/.env "
            "(a Postgres admin connection string pointed at the "
            "'postgres' maintenance database)."
        )
    with ephemeral_postgres_database(admin_url) as engine:
        yield engine
