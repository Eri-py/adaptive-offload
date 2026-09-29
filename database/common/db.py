"""SQLAlchemy engine/session-factory construction for the shared Postgres instance.

`server/api/services` and `training/` both import from here rather than opening
their own connections, per `.claude/coding-guidelines.md`'s Database section.
"""

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine(database_url: str | None = None) -> Engine:
    """Build a SQLAlchemy engine from `database_url`, defaulting to `DATABASE_URL`.

    Raises `RuntimeError` if no URL is passed and `DATABASE_URL` isn't set —
    callers should never silently fall back to a default connection string.
    """
    resolved_url = database_url or os.environ.get("DATABASE_URL")
    if not resolved_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Set it to a Postgres connection string "
            "(e.g. postgresql+psycopg://user:pass@host:5432/dbname) before "
            "building the engine."
        )
    return create_engine(resolved_url)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to `engine`."""
    return sessionmaker(bind=engine)
