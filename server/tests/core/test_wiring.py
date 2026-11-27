"""Pins the production wiring that every other test overrides away."""

from pathlib import Path

from api.core import config
from api.dependencies.providers import get_request_logger
from api.services.classification.request_log import log_classify_request


def test_env_paths_point_at_server_and_database_dotenv() -> None:
    """A wrong parent silently misses DATABASE_URL, and the best-effort logger hides it."""
    repo_root = Path(__file__).resolve().parents[3]
    assert repo_root / "server" == config.SERVER_ENV_PATH.parent
    assert config.SERVER_ENV_PATH == repo_root / "server" / ".env"
    assert config.DATABASE_ENV_PATH == repo_root / "database" / ".env"
    assert (repo_root / "database").is_dir()


def test_request_logger_provider_returns_the_real_writer() -> None:
    """Identity only: the shared `app` fixture no-ops this, so nothing else checks the wiring."""
    assert get_request_logger() is log_classify_request
