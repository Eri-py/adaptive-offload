# Generalize run-simulation's Image Data Source

## Overview

`run-simulation` currently hardcodes its dataset name and image source to one
specific pre-downloaded COCO val2017 copy. This feature makes the
annotations file, image folder, and dataset name explicit, required CLI
inputs, so `run-simulation` works against any COCO-format-annotated image
folder, not only the COCO download it ships with today.

## Requirements

- `run-simulation` requires three new CLI arguments — an annotations file
  path, an image folder path, and a dataset name — provided together. There
  is no default dataset, annotations path, or images path anymore; omitting
  any of the three is an error.
- The annotations file is parsed assuming the standard COCO JSON shape (a
  top-level `images` array of `{id, file_name, ...}` entries) to build the
  pool of available images for that run.
- Images continue to be resolved from the given image folder by file name,
  exactly as today.
- The provided dataset name is used everywhere a dataset label is needed for
  the run (the `scene_complexity` table lookup/write, and the
  `simulation_runs` row) instead of the previous hardcoded default.
- If the annotations path doesn't exist, isn't valid JSON, or doesn't
  contain an `images` array in the expected shape, `run-simulation` fails
  with a clear, specific error identifying the problem before any scoring or
  database work happens. The same fail-fast behavior applies if the image
  folder doesn't exist.
- The module currently named `coco.py` (which holds image-index loading and
  path resolution) is renamed to something that doesn't imply it's
  COCO-specific, since it now serves any COCO-format dataset.
- The `sync-coco-cache` CLI tool keeps working exactly as it does today,
  with its own COCO val2017-specific defaults (annotations path, image
  folder, download URL) unchanged. It remains a separate, optional
  convenience for populating a local image folder from the real COCO API,
  unaffected by `run-simulation`'s new required arguments.

## Out of Scope

- Any change to `stub_inference` or how latency/accuracy are computed — this
  feature only changes where images/annotations come from, not what's done
  with them.
- Any change to `score_complexity`, `preview_sample`, `preview_conditions`,
  or `relabel_run` — their own default dataset name (`config.DATASET_NAME`)
  and behavior are untouched.
- Dropping or renaming `ImageRecord.image_id` — it stays exactly as-is,
  unused by this feature but reserved for a future real-accuracy feature
  that will need it as the join key into COCO-format ground-truth
  annotations.
- Removing `config.DATASET_NAME`, or the renamed module's COCO val2017
  constants (annotations path, images directory, download base URL) — these
  remain as `sync-coco-cache`'s (and other CLIs') own defaults.
- Any change to the image-download logic itself (`download_missing_images`).
- Real accuracy scoring against ground-truth bounding boxes — a separate,
  future feature.

## Acceptance Criteria

- Given `run-simulation` is invoked without `--annotations`, `--images`, or
  `--dataset`, then it exits with a clear error rather than falling back to
  any COCO default.
- Given valid `--annotations <path> --images <folder> --dataset <name>
  --preset <preset>`, then `run-simulation` scores any images under
  `<folder>` not already present in the `scene_complexity` table for
  `<name>`, persists `(<name>, file_name, score)` rows for them, and
  proceeds through frame sampling, condition sampling, stub inference, and
  labeling exactly as it does today.
- Given `--annotations` points at a file that doesn't exist, isn't valid
  JSON, or lacks an `images` array, then `run-simulation` fails with a
  clear, specific error message before any scoring or database work occurs.
- Given `--images` points at a folder that doesn't exist, then
  `run-simulation` fails with a clear error before any scoring or database
  work occurs.
- Given `sync-coco-cache` is invoked with no arguments, then it behaves
  exactly as it does today (downloads missing val2017 images using its own
  built-in COCO defaults), unaffected by this feature.
- Given the module previously named `coco.py`, then it is renamed and every
  importer (`run_simulation.py`, `sync_coco_cache.py`, and their tests) is
  updated to reference the new name.
