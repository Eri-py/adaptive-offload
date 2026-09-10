"""Standalone scene-complexity scoring for an arbitrary folder of images.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.score_complexity --folder <path>

Wraps `complexity.scene_complexity()` over every image directly under a
folder. No COCO-specific logic, no database access, no dependency on any
other pipeline stage — this works identically on the real COCO cache or any
other folder of images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datagen.complexity import scene_complexity

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
    parser = argparse.ArgumentParser(
        description="Score every image in a folder for scene complexity."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder of images to score (non-recursive).",
    )
    args = parser.parse_args()

    scores = score_folder(args.folder)
    for file_name, score in scores.items():
        print(f"{file_name}\t{score:.4f}")


if __name__ == "__main__":
    main()
