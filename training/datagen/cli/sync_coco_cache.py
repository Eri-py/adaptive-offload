"""Standalone COCO val2017 cache-population entry point.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.cli.sync_coco_cache

Composes `image_source.load_image_index()` (read the annotations file for the full
pool of images) with `image_source.download_missing_images()` (fetch whichever of
those aren't already cached under `training/data/coco/val2017/`). This is
the only place in the pipeline that downloads COCO images — the main
simulator pipeline and the complexity-scoring entry point only ever read the
local cache via `image_source.resolve_image_path`, and fail clearly if this hasn't
been run yet.

No database access, no `.env` loading — like `score_complexity.py` and
`preview_conditions.py`, this tool has no dependency on any other pipeline
stage.
"""

from __future__ import annotations

from pathlib import Path

from datagen import config
from datagen.sourcing import image_source


def sync_coco_cache(
    *,
    annotations_path: Path = config.ANNOTATIONS_PATH,
    images_dir: Path = config.IMAGES_DIR,
    base_url: str = config.COCO_VAL2017_BASE_URL,
    fetch: image_source.FetchFn = image_source.fetch_image_bytes,
) -> tuple[int, int]:
    """Download whichever COCO val2017 images aren't already cached locally.

    Reads the image pool from `annotations_path` via `image_source.load_image_index`,
    then downloads whatever's missing under `images_dir` via
    `image_source.download_missing_images`. `base_url`/`fetch` are forwarded
    straight through, so tests can inject a fake `fetch` and never touch the
    real network. Returns `(already_cached_count, newly_downloaded_count)`.
    """
    image_records = image_source.load_image_index(annotations_path)
    downloaded = image_source.download_missing_images(
        image_records, images_dir=images_dir, base_url=base_url, fetch=fetch
    )
    already_cached_count = len(image_records) - len(downloaded)
    return already_cached_count, len(downloaded)


def main() -> None:
    already_cached_count, newly_downloaded_count = sync_coco_cache()
    print(f"Already cached: {already_cached_count}")
    print(f"Newly downloaded: {newly_downloaded_count}")


if __name__ == "__main__":
    main()
