"""Scene-complexity scoring for an arbitrary folder of images, persisted to Postgres.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.cli.score_complexity --folder <path> --dataset <name>

Wraps `complexity.scene_complexity()` over every image directly under a
folder — no COCO-specific logic, this works identically on the real COCO
cache or any other folder of images. Persists new scores to the
`scene_complexity` table under `--dataset`, skipping images already scored
for that dataset (same known-file-name filtering `run_simulation.py` uses),
so `score-complexity` is the actual way `scene_complexity` rows get
populated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.db import get_engine
from dotenv import load_dotenv

from datagen.persistence import get_known_complexity, store_complexity_scores
from datagen.sampling.complexity import scene_complexity

# Extensions this tool treats as images, matched case-insensitively.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def score_folder(folder: Path) -> dict[str, float]:
    """Score every image directly under `folder`, keyed by file name.

    Non-recursive: only files directly in `folder` are considered. Raises
    `FileNotFoundError` if `folder` doesn't exist and `ValueError` if it
    exists but contains no files matching `IMAGE_EXTENSIONS`.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")

    image_paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(
            f"No images (extensions: {', '.join(IMAGE_EXTENSIONS)}) found in {folder}"
        )

    return {path.name: scene_complexity(path) for path in image_paths}


def main() -> None:
    # CLI-only convenience: load DATABASE_URL from training/.env if it isn't
    # already in the environment (never overrides an explicit `export`).
    # Mirrors `run_simulation.py`'s `main()`.
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Score every image in a folder for scene complexity."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder of images to score (non-recursive).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name scene-complexity scores are stored under.",
    )
    args = parser.parse_args()

    scores = score_folder(args.folder)
    for file_name, score in scores.items():
        print(f"{file_name}\t{score:.4f}")

    engine = get_engine()
    known = get_known_complexity(engine, args.dataset)
    new_scores = {
        file_name: score for file_name, score in scores.items() if file_name not in known
    }
    store_complexity_scores(engine, args.dataset, new_scores)


if __name__ == "__main__":
    main()
