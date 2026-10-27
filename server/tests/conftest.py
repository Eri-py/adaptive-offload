"""Shared pytest fixtures for `server/tests/`.

Provides a fresh, disposable Postgres database per test so the frame-selection
query runs against real Postgres rather than a SQLite stand-in. Covered by the
scoped exception in `CLAUDE.md`'s Infrastructure section: fixture-driven
create/drop of ephemeral, uniquely named test databases only. Mirrors
`database/tests/conftest.py` and `training/tests/conftest.py`.
"""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from common.testing import ephemeral_postgres_database
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from api.core.app import create_app
from api.core.config import Settings, get_settings
from api.dependencies.database import get_session
from tests.support import DATASET, FrameFixture, seed_frames

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ClientFactory = Callable[[list[FrameFixture]], TestClient]


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    """Function-scoped engine bound to a freshly created, uniquely-named test DB."""
    admin_url = os.environ.get("POSTGRES_ADMIN_URL")
    if not admin_url:
        raise RuntimeError(
            "POSTGRES_ADMIN_URL is not set. Add it to server/.env "
            "(a Postgres admin connection string pointed at the "
            "'postgres' maintenance database)."
        )
    with ephemeral_postgres_database(admin_url) as engine:
        yield engine


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    """Stand-in for the dataset's image folder, so tests never read the real 5000."""
    folder = tmp_path / "val2017"
    folder.mkdir()
    return folder


@pytest.fixture
def postgres_session(postgres_engine: Engine) -> Iterator[Session]:
    """A session on the disposable database, for testing services directly."""
    with Session(postgres_engine) as session:
        yield session


@pytest.fixture
def settings(images_dir: Path) -> Settings:
    """Settings pointed at the test image folder rather than the real dataset."""
    return Settings(dataset=DATASET, images_dir=images_dir, default_frame_count=15)


@pytest.fixture
def build_client(
    postgres_engine: Engine, images_dir: Path, settings: Settings
) -> Iterator[ClientFactory]:
    """Seed the given frames, then hand back a client wired to that database."""
    opened: list[tuple[TestClient, Session]] = []

    def factory(frames: list[FrameFixture]) -> TestClient:
        with Session(postgres_engine) as session:
            seed_frames(session, images_dir, frames)

        request_session = Session(postgres_engine)
        app = create_app()
        app.dependency_overrides[get_session] = lambda: request_session
        app.dependency_overrides[get_settings] = lambda: settings
        client = TestClient(app)
        opened.append((client, request_session))
        return client

    yield factory

    for client, session in opened:
        client.close()
        session.close()
