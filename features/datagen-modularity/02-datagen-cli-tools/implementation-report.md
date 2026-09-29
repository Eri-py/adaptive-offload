# Data-Gen Config Split and CLI Tools — Implementation Report

**Feature:** Data-Gen Config Split and CLI Tools
**Directory:** `features/datagen-modularity/02-datagen-cli-tools/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Split presets out of config | completed |
| Task 2 — Complexity-scoring CLI | completed |
| Task 3 — Stratified-sample preview CLI | completed |
| Task 4 — Condition-vector preview CLI | completed |
| Task 5 — `get_run_results` + re-labeling CLI | completed |
| Task 6 — Split COCO acquisition: `coco.py` refactor + cache-population CLI | completed |
| Task 7 — Regression test run | completed |

## Files changed

### Added

- `training/datagen/presets.py` — `ConditionPresetRanges` + `PRESETS`, moved verbatim out of `config.py`
- `training/datagen/score_complexity.py` — standalone CLI: scores every image in an arbitrary folder
- `training/datagen/preview_sample.py` — standalone CLI: previews a stratified sample from already-persisted complexity scores
- `training/datagen/preview_conditions.py` — standalone CLI: previews condition vectors for a preset/count/seed
- `training/datagen/relabel_run.py` — standalone CLI: recomputes labels for an existing run under a different λ, without re-simulating
- `training/datagen/sync_coco_cache.py` — standalone CLI: downloads whichever COCO val2017 images are missing locally
- `training/tests/datagen/test_presets.py`, `test_score_complexity.py`, `test_preview_sample.py`, `test_preview_conditions.py`, `test_relabel_run.py`, `test_sync_coco_cache.py` — tests for the above

### Edited

- `training/datagen/config.py` — `PRESETS`/`ConditionPresetRanges` removed (moved to `presets.py`)
- `training/datagen/conditions.py` — imports `ConditionPresetRanges` from `datagen.presets`
- `training/datagen/run_simulation.py` — `config.PRESETS` → `presets.PRESETS` (4 sites)
- `training/datagen/persistence.py` — added `get_run_results(engine, run_id) -> list[ResultRow]`, reusing the existing dataclass
- `training/datagen/coco.py` — `resolve_image_path` is now a pure local lookup raising `FileNotFoundError` on a miss (previously silently downloaded); the download logic moved into a new `download_missing_images` function, preserving the existing atomic temp-file-then-rename write safety
- `training/tests/datagen/test_config.py`, `test_conditions.py`, `test_run_simulation.py` — updated for the presets split
- `training/tests/datagen/test_persistence.py` — added `get_run_results` tests
- `training/tests/datagen/test_coco.py` — rewritten for the `resolve_image_path`/`download_missing_images` split

Not part of this plan's scope but confirmed unaffected by inspection: `training/tests/datagen/test_run_simulation.py`'s existing tests (they inject their own fake `resolve_image` closure, never calling the real `coco.resolve_image_path`) and `run_simulation.py`'s reference to `coco.resolve_image_path` (used only as a default callable value, never invoked with a `fetch` argument at that call site) — both verified by direct inspection during Task 6, not assumed.

## Tests

**Unit** (pure functions / no database):
- `training/tests/datagen/test_presets.py` (1 test): every preset defines all four axis ranges.
- `training/tests/datagen/test_score_complexity.py` (4 tests): scores every image in a folder, ignores non-matching files, clear errors for a missing/empty folder.
- `training/tests/datagen/test_preview_conditions.py` (3 tests): matches `sample_condition_vectors` directly for every preset, clear error for an unknown preset.
- `training/tests/datagen/test_coco.py` (10 tests): `resolve_image_path`'s cache-hit and fail-clearly-on-miss behavior; `download_missing_images`'s skip-cached/download-missing/mixed/no-refetch/atomic-write/interrupted-write behavior.
- `training/tests/datagen/test_sync_coco_cache.py` (3 tests): downloads exactly the missing subset, leaves a fully-cached input untouched, forwards `base_url` correctly.

**Integration** (real ephemeral Postgres database):
- `training/tests/datagen/test_preview_sample.py` (3 tests): matches `stratified_sample` directly against seeded complexity scores, creates no run/result rows, clear error for an unscored dataset.
- `training/tests/datagen/test_persistence.py` (+3 tests for `get_run_results`): round-trip of all `ResultRow` fields, empty result for a run with no results or a nonexistent `run_id`.
- `training/tests/datagen/test_relabel_run.py` (3 tests): correct flip reporting under a changed λ, zero persistence side effects (verified via a fresh `get_run_results` read), clear error for a run with no results.

**71 tests passing** in `training/` (58 pre-existing at this plan's start + 13 net new, after accounting for the `test_coco.py` split), plus `database/` (1) and `server/` (0, expected — no code yet) unaffected. 0 skipped.

## Commits

`7b0a652..4826606` (8 commits, on `feature/datagen-modularity`): from `fd87ad0` (Add implementation plan) through `4826606` (Task 7: regression test run).

## Notable events

None — every task completed on its first attempt, no stop-and-report or resume-from-partial situations. Two small self-corrections worth noting (not stop-and-report events): Task 5's `relabel_run.py` initially omitted the established `.env`-loading-in-`main()`-only convention, caught via manual CLI smoke-testing before finalizing; Task 6's `sync_coco_cache.py` initially copied that same `load_dotenv` habit from the prior task even though this CLI needs no database access, also caught and removed before finalizing.
