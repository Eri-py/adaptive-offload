"""Server settings, read from the environment with repo-local defaults.

The dataset's image folder is configuration rather than something request
handling knows about, per the frame-endpoints spec, so the controller and the
frame service both take it from here.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Frames returned when the caller doesn't ask for a count.
DEFAULT_FRAME_COUNT = 15

# Resolved relative to this file, not cwd, so running the server from anywhere
# still finds the dataset the simulator measured.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGES_DIR = _REPO_ROOT / "training" / "data" / "coco" / "val2017"


@dataclass(frozen=True)
class Settings:
    """The one dataset this server serves benchmark frames from."""

    dataset: str
    images_dir: Path
    default_frame_count: int


@lru_cache
def get_settings() -> Settings:
    """Build settings once per process; used directly as a FastAPI dependency."""
    return Settings(
        dataset=os.environ.get("BENCHMARK_DATASET", "coco_val2017"),
        images_dir=Path(os.environ.get("BENCHMARK_IMAGES_DIR", str(_DEFAULT_IMAGES_DIR))),
        default_frame_count=int(
            os.environ.get("BENCHMARK_FRAME_COUNT", str(DEFAULT_FRAME_COUNT))
        ),
    )
