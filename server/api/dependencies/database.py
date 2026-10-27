"""FastAPI session provider over `database/`'s shared engine/session setup.

The engine is built lazily and cached so importing the app doesn't require
`DATABASE_URL`, and a Postgres that's down surfaces as a failing query (which
`core.app` turns into a 503) rather than an import-time crash.
"""

from collections.abc import Iterator
from functools import lru_cache

from common.db import get_engine, get_session_factory
from sqlalchemy.orm import Session, sessionmaker


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return get_session_factory(get_engine())


def get_session() -> Iterator[Session]:
    """Yield a session per request, closed when the request ends."""
    with _session_factory()() as session:
        yield session
