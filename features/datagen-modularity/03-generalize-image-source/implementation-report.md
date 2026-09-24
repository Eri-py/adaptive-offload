# Generalize run-simulation's Image Data Source — Implementation Report

**Feature:** Generalize run-simulation's Image Data Source
**Directory:** `features/datagen-modularity/03-generalize-image-source/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Generalize `image_source.py`'s validation and error messages | completed |
| Task 2 — Add required `--annotations`/`--images`/`--dataset` args to `run-simulation` | completed |
| Task 3 — Regression test run | completed |

## Files changed

(`coco.py`'s rename/move to `training/datagen/sourcing/image_source.py` happened in a separate, prior reorg commit, not part of this plan — only its content changed here.)

### Edited

- `training/datagen/sourcing/image_source.py` — module docstring and `ImageRecord`'s docstring generalized to describe COCO-format annotation loading usable by any COCO-format dataset (not just COCO val2017), with the val2017 download helpers called out as the one still-COCO-specific piece; `load_image_index` now raises `FileNotFoundError`/`ValueError` with clear, path-naming messages for a missing/malformed annotations file or a missing `"images"` key; `resolve_image_path`'s error message dropped its COCO-specific framing and `sync_coco_cache` suggestion.
- `training/tests/datagen/sourcing/test_image_source.py` — two existing error-message assertions updated to match the new wording; three new tests for `load_image_index`'s validation (missing file, invalid JSON, missing `"images"` key).
- `training/datagen/cli/run_simulation.py` — `run_simulation()` gained a `dataset: str | None = None` parameter (defaults to `config.DATASET_NAME`, same pattern as its other optional tunables), replacing 4 internal `config.DATASET_NAME` references; new `_require_images_dir` helper fails fast on a missing images folder; `main()` gained required `--annotations`/`--images`/`--dataset` arguments, wired through `image_source.load_image_index`/`functools.partial(image_source.resolve_image_path, ...)`; module docstring and argparse description updated, including fixing a stale `python -m datagen.cli.run_simulation` invocation example to the real `run-simulation` entry point.
- `training/tests/datagen/cli/test_run_simulation.py` — two new tests for `_require_images_dir`; only the import line changed otherwise, no existing test body/assertions modified.

## Tests

**Unit** (pure functions / no database):
- `training/tests/datagen/sourcing/test_image_source.py` (+3 tests): `load_image_index` raises a clear `FileNotFoundError` for a missing annotations file, a clear `ValueError` for invalid JSON, and a clear `ValueError` for JSON missing the `"images"` key.
- `training/tests/datagen/cli/test_run_simulation.py` (+2 tests): `_require_images_dir` accepts an existing directory and raises a clear `FileNotFoundError` (naming the path) for a nonexistent one.

**Integration** (real ephemeral Postgres database):
- No new integration tests — Task 2 deliberately left every existing `run_simulation()` integration test unmodified, since none of them pass `dataset=` and `resolved_dataset` still resolves to `config.DATASET_NAME` by construction, matching their existing assertions.

**77 tests passing** in `training/` (72 pre-existing at this plan's start + 5 net new), ruff and mypy both clean. 0 skipped.

## Commits

`c030c10..HEAD` on `feature/datagen-modularity`:

- `c030c10` — Update 03-generalize-image-source plan for the datagen reorg
- `ae813f9` — Task 1: generalize image_source.py's validation and error messages
- `730b0ac` — Task 2: add required --annotations/--images/--dataset args to run-simulation
- `14af204` — Task 3: regression test run

## Notable events

- Task 2's subagent could not independently verify the 7 existing DB-backed `run_simulation` integration tests mid-task because Postgres was not reachable at the time (confirmed via a raw TCP probe, not started per `CLAUDE.md`'s infrastructure rule). Flagged explicitly rather than assumed passing. The orchestrator asked the user to start Postgres; once it was up, a full suite rerun confirmed all 77 tests pass, including those 7 unmodified.
