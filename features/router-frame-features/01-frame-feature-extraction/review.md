# Review — Frame Feature Extraction

## Verdict

The implementation meets the spec, and there are no blockers. The new
`frame_features` table, migration `0003`, the five pure image statistics, the
six confidence features (zero detections give zeros), resume-only-missing,
up-front missing-image and missing-weights failures, and batched persistence
all match the plan. The extractor calls YOLOv8n exactly as the simulator's local
path does: same `load_model`, same weights path, same
`model(path, device="cpu", verbose=False)`. `score_accuracy` applies no
confidence or class filter, so the features describe the same predictions the
local accuracy was scored on. Verification on this branch: `training` ruff is
clean, mypy shows only the 9 known errors (all outside new files), and 108
tests pass. `database` ruff, mypy and pytest (1 test) all pass. The real
end-to-end run and the "exactly 5,000 rows" check are still pending, because the
migration has not been applied (this is intentional). The findings below are
polish. The most useful one is S1: the timing summary does not report a single
per-frame cost for the confidence-feature group, which is the number spec 02
will want.

## Acceptance Criteria

1. **One row per frame for all 5,000 `coco_val2017` frames, every feature populated.** MET in code, not yet observed on real data. Frames come from `scene_complexity` (`training/router/feature_store.py:30-44`), and one row is built per pending frame (`training/router/extract_features.py:94-114`). Every column is `nullable=False` (`database/common/models.py:109-129`, migration `0003:30-47`). The row count on the real DB depends on `scene_complexity` holding all 5,000 frames. Confirm this after the user runs the migration and the step.
2. **Zero detections give zeros, not missing values.** MET. `training/router/frame_features/confidence.py:47-55`. Tested at the unit level (`tests/router/frame_features/test_confidence.py:32`, `:79`) and through persistence (`tests/router/test_extract_features.py:91`).
3. **Confidence features match a fresh YOLOv8n run with the simulator's weights and settings.** MET. `extract_features.py:131-145` uses `load_model(config.LOCAL_MODEL_WEIGHTS_PATH, "cpu")` and the same call as `yolo_inference.py` `run_inference`. `load_model` is a pure extraction of the old setup (`training/datagen/simulate/yolo_inference.py:50-72`). A manual check on 3 real images matched exactly (progress.md Task 3).
4. **A re-run computes only missing frames and leaves existing rows unchanged.** MET. `extract_features.py:75-81`, plus the skip-existing guard in `store_features` (`feature_store.py:57-89`). Tested in `test_extract_features.py:116` (predict is never called) and `test_feature_store.py:100` (no overwrite).
5. **Output includes the average per-frame compute time for the image features and the confidence features.** MET, with a caveat (see S1). `extract_features.py:116-127` prints `image_stats`, `model` and `confidence`. The printed `confidence` value covers only the list-to-stats derivation, not the model call.
6. **A missing image or missing weights stops the run with a message naming what is missing, and nothing is downloaded.** MET. Images: `extract_features.py:83-88` raises before any compute, listing the names, and this is tested (`test_extract_features.py:141`). Weights: `_require_weights_file` via `load_model` names the path and runs before `YOLO(...)`, so nothing is downloaded (see N1 on the message wording).
7. **The only schema change is the new table, added through a migration.** MET. `0003_create_frame_features_table.py` creates only `frame_features`, and `models.py` gains only `FrameFeatures`. The migration columns and primary key match the model exactly.

## Scope

The diff matches the plan's Files Changed table. The only addition is
`sequences_from_results` in `confidence.py`, which splits the model call from
the feature derivation so each can be timed separately. It is explained in
learnings.md and stays inside a planned file. The sibling
`02-feature-router-experiment/spec.md` and the workflow docs are out of review
scope, as instructed. No app, server or simulator behaviour changed.

## Blockers

None.

## Suggestions

#### S1 — Timing summary gives no per-frame cost for the confidence-feature group

- **File:** `training/router/extract_features.py:116-127`
- **Issue:** The spec asks for the average time "to compute each feature group". The output splits the confidence group into `model` and `confidence`, and the value labelled `confidence` (list-to-stats, microseconds) looks like the group's cost when the real cost is the model call. Spec 02 could easily quote the wrong number.
- **Fix:** Also print (and put in `TimingSummary`) a `confidence_group_seconds = model + confidence` total. Or rename the labels to `image_features=` and `confidence_features (model+derive)=`, with the breakdown in parentheses.
- **Decision:** Accepted — addressed in "Address S1: report the confidence-feature group total cost"

#### S2 — The model loads even when there is nothing to compute

- **File:** `training/router/extract_features.py:170-172`
- **Issue:** `main()` builds the real predictor (weights load plus a warm-up inference) before `extract_features` finds out nothing is pending or that images are missing. A re-run on a complete dataset pays the model load, and a wrong `--folder` is reported only after the model has loaded.
- **Fix:** Make `predict` lazy (build the model on first call). Or split the pending and missing-image check into a helper that `main()` calls before `_make_real_predictor`.
- **Decision:** Accepted — addressed in "Address S2: load the model only when frames are pending"

#### S3 — The missing-image error lists every missing name

- **File:** `training/router/extract_features.py:84-88`
- **Issue:** Point `--folder` at the wrong directory and the `FileNotFoundError` message holds all 5,000 file names. That hides the useful part, which is the folder path and the count.
- **Fix:** Show the count and folder, plus the first ~10 names and "... and N more".
- **Decision:** Accepted — addressed in "Address S3: truncate the missing-image error"

## Nitpicks

#### N1 — The missing-weights message names `run-simulation` when the extractor raises it

- **File:** `training/datagen/simulate/yolo_inference.py:114-119`
- **Issue:** `load_model` is now shared, but `_require_weights_file` still says "run-simulation does not download weights itself". The same message shows up when `router.extract_features` fails.
- **Fix:** Make it tool-neutral, e.g. "... Place the real YOLO weights file there; this pipeline never downloads weights."
- **Decision:** Accepted — addressed in "Address N1: tool-neutral missing-weights message"

#### N2 — Docstring misquotes the spec

- **File:** `training/router/extract_features.py:68-73`
- **Issue:** The docstring cites the spec's "never leaves partial/missing values" as the reason for the up-front image check. The spec says "never skipped or left with missing values" about zero-detection frames, which is unrelated. The right citation is the "fails with a clear message if an image ... is missing" requirement.
- **Fix:** Drop the misattributed quote and keep the "fails clearly on a missing image" reference.
- **Decision:** Accepted — addressed in "Address N2: fix the misquoted spec reference in the docstring"

#### N3 — Multi-line comments break the comment guideline

- **File:** `training/tests/router/test_feature_store.py:103-105`, also `training/tests/router/test_extract_features.py:126-127` and `training/router/extract_features.py:44-45`, `:48-49`
- **Issue:** `.claude/coding-guidelines.md` "Comments" allows one line, and two only when truly needed. The `stale_row` comment runs to three lines, and the others are two lines that could be one.
- **Fix:** Shorten each to one line, e.g. `# Same file name, different values: must be skipped, not overwritten or raise.`
- **Decision:** — _(pending)_

#### N4 — `test_nothing_pending_returns_none` duplicates the second-run test

- **File:** `training/tests/router/test_extract_features.py:159-167`
- **Issue:** `test_second_run_computes_nothing_new_and_leaves_existing_rows_unchanged` already asserts `summary is None` on a fully-populated re-run, so this test adds no coverage.
- **Fix:** Remove it, or turn it into a test of the printed timing summary (see Tests).
- **Decision:** — _(pending)_

#### N5 — `store_features` re-queries the known names that the caller already has

- **File:** `training/router/feature_store.py:63-69`
- **Issue:** Each batch re-selects every stored file name for the dataset (about 50 queries of up to 5,000 rows each over a full run), and duplicates `get_known_feature_file_names`'s query inline. This is harmless at this scale, but the query exists twice.
- **Fix:** Call `get_known_feature_file_names` inside `store_features` instead of repeating the select, or accept that the duplication is deliberate and mirrors `store_complexity_scores`.
- **Decision:** — _(pending)_

## Tests

Added: 6 image-stats tests, 6 confidence tests (including a synthetic `Results`
adapter test), 5 persistence tests against ephemeral Postgres, 5 orchestration
integration tests, and an extended round-trip test for all five tables.
`training` runs 108 passing (86 before), and `database` has 1 passing. The
tests cover every testable AC: zeros, resume, missing image before any write,
dataset scoping, and no overwrite.

Gaps:
- Nothing checks the printed timing line. The tests only assert that a
  `TimingSummary` is returned. `capsys` could check that both group timings
  appear, which is what AC5 is about.
- No test covers the batch boundary (more than `BATCH_SIZE` frames, or an
  exact multiple). Patching `BATCH_SIZE` to 2 with 5 frames would cover both
  the flush and the tail-flush path cheaply.
- The "matches a fresh YOLOv8n run" check was a one-off manual run, as the plan
  intended. That is fine at the prototype bar.
- Note: `test_extract_features.py` imports `router.extract_features`, which pulls
  in `ultralytics`/`torch` through `yolo_inference` and `confidence.py`. The
  claim in learnings.md that these tests "never import `ultralytics`/`torch`"
  is wrong. The only cost is slower collection, not correctness.

## Recommended Decisions

- **S1** — Accept — Spec 02 will quote these numbers, and the current `confidence` label understates the group's real cost. The fix is a one-line addition.
- **S2** — Accept — A lazy predictor avoids a pointless model load on no-op re-runs and reports a bad `--folder` sooner. The change is small and local.
- **S3** — Accept — A wrong `--folder` is the most likely mistake, and a 5,000-name message hides the useful information. Truncating is trivial.
- **N1** — Accept — A one-string change that stops the message pointing users at the wrong tool.
- **N2** — Accept — A docstring accuracy fix with no risk.
- **N3** — Accept — The guideline requires it, and the change is mechanical.
- **N4** — Accept — Removes a redundant test, or reuses it to cover the untested timing output.
- **N5** — Decline — It deliberately mirrors `store_complexity_scores`, and the extra queries don't matter at 5,000 rows.
