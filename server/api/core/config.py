"""Environment-driven server configuration with repo-relative defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_SERVER_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _SERVER_DIR.parent

load_dotenv(_SERVER_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Paths the server needs at startup."""

    large_checkpoint: Path
    species_labels: Path


def load_settings() -> Settings:
    """Reads settings at call time so tests and `.env` changes are honoured."""
    return Settings(
        large_checkpoint=Path(
            os.environ.get(
                "BIRDS_LARGE_CHECKPOINT", str(_REPO_ROOT / "training/models/birds/large.pt")
            )
        ),
        species_labels=Path(
            os.environ.get(
                "BIRDS_SPECIES_LABELS",
                str(_REPO_ROOT / "app/assets/models/cub200-species.json"),
            )
        ),
    )
