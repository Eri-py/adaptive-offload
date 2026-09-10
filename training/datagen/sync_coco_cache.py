"""Standalone COCO val2017 cache-population entry point.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`-not-`training.datagen.*` import-root convention):

    python -m datagen.sync_coco_cache

Composes `coco.load_image_index()` (read the annotations file for the full
pool of images) with `coco.download_missing_images()` (fetch whichever of
those aren't already cached under `training/data/coco/val2017/`). This is
the only place in the pipeline that downloads COCO images — the main
simulator pipeline and the complexity-scoring entry point only ever read the
local cache via `coco.resolve_image_path`, and fail clearly if this hasn't
been run yet.

No database access, no `.env` loading — like `score_complexity.py` and
`preview_conditions.py`, this tool has no dependency on any other pipeline
stage.
"""

from __future__ import annotations

from pathlib import Path

from datagen import coco


def sync_coco_cache(
    *,
    annotations_path: Path = coco.ANNOTATIONS_PATH,
    images_dir: Path = coco.IMAGES_DIR,
    base_url: str = coco.COCO_VAL2017_BASE_URL,
    fetch: coco.FetchFn = coco.fetch_image_bytes,
) -> tuple[int, int]:
    """Download whichever COCO val2017 images aren't already cached locally.

    Reads the image pool from `annotations_path` via `coco.load_image_index`,
    then downloads whatever's missing under `images_dir` via
    `coco.download_missing_images`. `base_url`/`fetch` are forwarded
    straight through, so tests can inject a fake `fetch` and never touch the
    real network. Returns `(already_cached_count, newly_downloaded_count)`.
    """
    image_records = coco.load_image_index(annotations_path)
    downloaded = coco.download_missing_images(
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
