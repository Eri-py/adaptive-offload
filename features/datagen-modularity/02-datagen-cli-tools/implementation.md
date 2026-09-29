# Data-Gen Config Split and CLI Tools — Implementation Plan

## Summary

Split `training/datagen/config.py`'s condition-scenario presets into their
own module, add four standalone CLI entry points (complexity scoring on any
folder, stratified-sample preview, condition-vector preview, and
re-labeling an existing run under a different λ), and split COCO's
download-on-miss behavior out of the main pipeline's read path into its own
explicit cache-population CLI. Every new entry point reuses existing pure
functions and persistence code unchanged — nothing about the simulator's
actual computation changes.

## Approach & Key Decisions

- **`presets.py` is a pure relocation** — `ConditionPresetRanges` and
  `PRESETS` move out of `config.py` verbatim, no value or shape changes.
  Every consumer (`conditions.py`, `run_simulation.py`, and their tests)
  gets repointed from `config.PRESETS`/`config.ConditionPresetRanges` to
  `presets.PRESETS`/`presets.ConditionPresetRanges`.
- **Every new CLI follows `run_simulation.py`'s established shape**: a
  directly-callable, injectable core function plus a thin
  `argparse`-based `main()`. The two CLIs that touch Postgres
  (`preview_sample.py`, `relabel_run.py`) build their engine the same way
  `run_simulation.py` does — `python-dotenv` auto-loads `training/.env` in
  `main()` only, never in the core function, so tests inject the ephemeral-
  Postgres fixture engine directly. `score_complexity.py` and
  `preview_conditions.py` need no database access at all.
- **`get_run_results` reuses the existing `ResultRow` dataclass** as its
  return type — it already has exactly the fields `relabel_run.py` needs
  (`frame_id` plus the four latency/accuracy values and the stored
  `label`), so no new type is introduced in `persistence.py`.
- **`coco.resolve_image_path` loses its download capability entirely.** It
  becomes a pure local-file lookup that raises a clear `FileNotFoundError`
  (naming the new `sync_coco_cache` CLI) when a file is missing, instead of
  silently fetching it — this is the spec's explicit requirement. The
  download logic (including the existing atomic temp-file-then-rename
  safety property) moves into a new `download_missing_images` function,
  used only by the new `sync_coco_cache.py` CLI. `run_simulation.py`'s
  default `resolve_image` callable is unaffected in practice for a real run
  against the already-fully-cached local pool (every file already exists,
  so the lookup always succeeds) — this only changes behavior for the
  genuinely-missing-file case, exactly as intended.
- **`relabel_run.py` never writes anything** — no call to `store_results`
  or `create_run` — per the spec's explicit Out of Scope. It reports the
  recomputed label per row (alongside the row's originally-stored label,
  so a flip is visible) and prints a flip-count summary.
- Nothing in `stub_inference.py`, `labeling.py`'s utility formula, or
  `complexity.py`'s algorithm changes — every new CLI is a thin wrapper
  reusing existing pure functions, per the spec's Out of Scope.

## Out of Scope

Everything the spec excludes: the shared DB package's location (covered by
the sibling, already-completed 01 spec), any change to the win/loss utility
formula, the stub-inference model, or the scene-complexity algorithm,
persisting re-labeled results anywhere, standalone entry points for
`persistence.py`'s write functions, and support for datasets other than
COCO val2017.

## Dependencies and Configuration

None. No new external packages (`argparse` is stdlib; everything else is
already a `training/` dependency). No new `.env` entries — the two
DB-touching CLIs reuse `DATABASE_URL`/`POSTGRES_ADMIN_URL` exactly as
`run_simulation.py` and the test fixtures already do. No schema changes.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/datagen/presets.py` | add | `ConditionPresetRanges` + `PRESETS`, split out of `config.py` | spec requirement |
| `training/datagen/config.py` | edit | remove `PRESETS`/`ConditionPresetRanges` (moved to `presets.py`) | spec requirement |
| `training/datagen/conditions.py` | edit | import `ConditionPresetRanges` from `datagen.presets` | follows the split |
| `training/datagen/run_simulation.py` | edit | `config.PRESETS` → `presets.PRESETS` (4 sites); later also gains the stricter missing-file failure mode from the `coco.py` refactor | follows the split; Task 6 |
| `training/tests/datagen/test_config.py` | edit | drop preset-related assertions (moved) | follows the split |
| `training/tests/datagen/test_presets.py` | add | preset-related assertions moved from `test_config.py` | follows the split |
| `training/tests/datagen/test_conditions.py` | edit | import `PRESETS` from `datagen.presets` | follows the split |
| `training/tests/datagen/test_run_simulation.py` | edit | `config.PRESETS` → `presets.PRESETS` (2 sites) | follows the split |
| `training/datagen/score_complexity.py` | add | standalone CLI: score every image in a given folder | spec requirement |
| `training/tests/datagen/test_score_complexity.py` | add | tests for the above | — |
| `training/datagen/preview_sample.py` | add | standalone CLI: preview a stratified sample for a frame count/bucket count/seed | spec requirement |
| `training/tests/datagen/test_preview_sample.py` | add | tests (Postgres-backed) | — |
| `training/datagen/preview_conditions.py` | add | standalone CLI: preview condition vectors for a preset/count/seed | spec requirement |
| `training/tests/datagen/test_preview_conditions.py` | add | tests | — |
| `training/datagen/persistence.py` | edit | add `get_run_results(engine, run_id) -> list[ResultRow]` | needed by `relabel_run.py` |
| `training/tests/datagen/test_persistence.py` | edit | tests for `get_run_results` | — |
| `training/datagen/relabel_run.py` | add | standalone CLI: recompute labels for an existing run under a new λ | spec requirement |
| `training/tests/datagen/test_relabel_run.py` | add | tests (Postgres-backed) | — |
| `training/datagen/coco.py` | edit | `resolve_image_path` fails clearly on a missing file (no fetch); new `download_missing_images` function | spec requirement |
| `training/datagen/sync_coco_cache.py` | add | standalone CLI: populate/refresh the local COCO image cache | spec requirement |
| `training/tests/datagen/test_coco.py` | edit | update tests for the `resolve_image_path`/`download_missing_images` split | follows the `coco.py` refactor |
| `training/tests/datagen/test_sync_coco_cache.py` | add | tests for the new CLI | — |

## Tasks

### Task 1 — Split presets out of config

- **Objective:** Move `ConditionPresetRanges`/`PRESETS` into their own module and repoint every consumer.
- **Files:** `training/datagen/presets.py`, `training/datagen/config.py`, `training/datagen/conditions.py`, `training/datagen/run_simulation.py`, `training/tests/datagen/test_config.py`, `training/tests/datagen/test_presets.py`, `training/tests/datagen/test_conditions.py`, `training/tests/datagen/test_run_simulation.py`
- **Details:** Create `training/datagen/presets.py` containing exactly `ConditionPresetRanges` and `PRESETS` as they exist today in `config.py` — no value changes. Remove both from `config.py`. Update `conditions.py`'s import (`from datagen.config import ConditionPresetRanges` → `from datagen.presets import ConditionPresetRanges`). Update `run_simulation.py`'s four `config.PRESETS` references to `presets.PRESETS` (add the import). Split `test_config.py`'s preset-related assertions into a new `test_presets.py`; update `test_conditions.py` and `test_run_simulation.py`'s imports the same way as the production code.
- **Success criteria:**
  - `cd training && ruff check . && mypy .` clean
  - Full `training/` suite passes with the same test count as before this task (assertions moved, not lost)

### Task 2 — Complexity-scoring CLI

- **Objective:** A standalone entry point that scores every image in an arbitrary folder for scene complexity.
- **Files:** `training/datagen/score_complexity.py`, `training/tests/datagen/test_score_complexity.py`
- **Details:** `python -m datagen.score_complexity --folder <path>` globs common image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`) directly under the given folder (non-recursive is fine), calls `complexity.scene_complexity()` on each, and prints one line per image (file name + score). No COCO-specific logic, no database access, no dependency on any other pipeline stage — this must work identically on the real COCO cache or any other folder of images. Error clearly if the folder doesn't exist or contains no matching images.
- **Success criteria:**
  - Given a `tmp_path` folder of synthetic images (not the COCO cache), running the CLI reports a complexity value for every image in it
  - `cd training && ruff check . && mypy .` clean; full suite passes

### Task 3 — Stratified-sample preview CLI

- **Objective:** A standalone entry point that previews which frames a stratified sample would select, reading already-persisted complexity scores.
- **Files:** `training/datagen/preview_sample.py`, `training/tests/datagen/test_preview_sample.py`
- **Details:** `python -m datagen.preview_sample [--dataset NAME] [--frame-count N] [--bucket-count N] [--seed N]`, defaulting each omitted flag to `config.DATASET_NAME`/`config.FRAME_COUNT`/`config.STRATIFICATION_BUCKET_COUNT`/`config.SEED`. The core function takes an `Engine` plus the same parameters, calls `persistence.get_known_complexity(engine, dataset)` then `sampling.stratified_sample(...)`, and returns/prints the selected frame file names (with their complexity scores, for context). Read-only — never calls any `persistence` write function, never runs condition sampling, stub inference, or labeling. Error clearly if no complexity scores exist yet for the given dataset.
- **Success criteria:**
  - Against the ephemeral-Postgres fixture, seeded with a small set of `scene_complexity` rows via `persistence.store_complexity_scores`, the CLI's core function reports the same frame selection `sampling.stratified_sample` would return directly for the same inputs
  - No `simulation_runs`/`simulation_results` rows are created by running this CLI
  - `cd training && ruff check . && mypy .` clean; full suite passes

### Task 4 — Condition-vector preview CLI

- **Objective:** A standalone entry point that previews the condition vectors a preset/count/seed would sample.
- **Files:** `training/datagen/preview_conditions.py`, `training/tests/datagen/test_preview_conditions.py`
- **Details:** `python -m datagen.preview_conditions --preset NAME [--count N] [--seed N]`, `--preset` required with `choices=sorted(presets.PRESETS)`, `--count`/`--seed` defaulting to `config.CONDITION_VECTOR_COUNT`/`config.SEED`. Calls `conditions.sample_condition_vectors(presets.PRESETS[name], count, seed)` and prints each vector's four values. Pure function, no database access at all.
- **Success criteria:**
  - The CLI's core function returns the same vectors `conditions.sample_condition_vectors` would return directly for the same inputs
  - `cd training && ruff check . && mypy .` clean; full suite passes

### Task 5 — `get_run_results` + re-labeling CLI

- **Objective:** Add a read function for an existing run's persisted results, and a standalone CLI that recomputes labels under a different λ without re-running any earlier pipeline stage.
- **Files:** `training/datagen/persistence.py`, `training/tests/datagen/test_persistence.py`, `training/datagen/relabel_run.py`, `training/tests/datagen/test_relabel_run.py`
- **Details:** In `persistence.py`, add `get_run_results(engine: Engine, run_id: str) -> list[ResultRow]` — reads every `SimulationResult` row for `run_id` and maps each to a `ResultRow` (reusing the existing dataclass; every field it needs is already there). Returns an empty list if the run has no rows (including a nonexistent `run_id` — this function doesn't need to distinguish the two cases, the CLI handles reporting). In `relabel_run.py`, `python -m datagen.relabel_run --run-id ID --lambda VALUE` calls `get_run_results`, then for each row calls `labeling.compute_label(row.local_latency_ms, row.local_accuracy, row.offload_latency_ms, row.offload_accuracy, VALUE)`, and prints each row's `frame_id`, stored label, and recomputed label, plus a summary count of how many rows flipped. Never calls `create_run` or `store_results`. Error clearly (not a silent empty report) if `get_run_results` returns nothing for the given `run_id`.
- **Success criteria:**
  - Against the ephemeral-Postgres fixture: create a run and store a handful of results with `create_run`/`store_results`, then confirm `get_run_results` returns them with all fields intact
  - Running `relabel_run`'s core function against that seeded run with a λ chosen to flip at least one row's label confirms the flip is reported correctly, and that no new `simulation_runs` row and no changes to the existing `simulation_results` rows occur (re-read via `get_run_results` after, confirm unchanged)
  - `cd training && ruff check . && mypy .` clean; full suite passes

### Task 6 — Split COCO acquisition: `coco.py` refactor + cache-population CLI

- **Objective:** Make `resolve_image_path` a pure local lookup that fails clearly on a missing file, and move the download logic into a new standalone cache-population CLI.
- **Files:** `training/datagen/coco.py`, `training/datagen/sync_coco_cache.py`, `training/tests/datagen/test_coco.py`, `training/tests/datagen/test_sync_coco_cache.py`
- **Details:** In `coco.py`: remove `resolve_image_path`'s `fetch`/download behavior — it should only check `images_dir / file_name` and either return that path or raise `FileNotFoundError` with a message pointing at `python -m datagen.sync_coco_cache`. Add `download_missing_images(image_records: list[ImageRecord], *, images_dir: Path = IMAGES_DIR, base_url: str = COCO_VAL2017_BASE_URL, fetch: FetchFn = fetch_image_bytes) -> list[str]`, moving the existing atomic-write download logic (temp-file-then-rename) here, returning the file names it actually downloaded (skipping ones already present). In `sync_coco_cache.py`: `python -m datagen.sync_coco_cache` calls `coco.load_image_index()` then `coco.download_missing_images(...)`, printing how many were already cached vs. newly downloaded. Update `test_coco.py`'s existing `resolve_image_path` tests to match the new fail-clearly behavior (a missing-file test now asserts `FileNotFoundError`, not a mocked download), and move the download-behavior tests to exercise `download_missing_images` instead, using the same injected-fetch-callable pattern as before so tests never touch the real network.
- **Success criteria:**
  - `resolve_image_path` on an existing file still returns immediately (no `fetch` parameter exists anymore to even call)
  - `resolve_image_path` on a missing file raises `FileNotFoundError` (not a network call)
  - `download_missing_images`, given a mix of already-present and missing file names with an injected fake fetch, downloads exactly the missing ones (atomically) and leaves already-cached ones untouched, never invoking `fetch` for files already on disk
  - `cd training && ruff check . && mypy .` clean; full suite passes, including `test_run_simulation.py`'s existing reproducibility tests unchanged (its fake pool already injects its own `resolve_image` closure, so this refactor shouldn't require changing those tests — confirm that's actually true rather than assuming it)

### Task 7 — Regression test run

- **Objective:** Run every test across `training/` (and confirm `database/`/`server/` are unaffected, since this plan doesn't touch them) and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && ruff check . && mypy . && pytest` passes, with no skips
  - `cd database && ruff check . && mypy . && pytest` still passes (unaffected by this plan, sanity check)
  - `cd server && ruff check . && mypy . && pytest` still passes (unaffected by this plan, sanity check)
