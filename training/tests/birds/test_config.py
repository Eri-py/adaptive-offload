from pathlib import Path

from birds.config import ENV_PATH


def test_env_path_points_at_training_dotenv() -> None:
    """The entry points load DATABASE_URL from training/.env; a wrong parent silently misses it."""
    training_dir = Path(__file__).resolve().parents[2]
    assert ENV_PATH == training_dir / ".env"
    assert (training_dir / "birds").is_dir()
