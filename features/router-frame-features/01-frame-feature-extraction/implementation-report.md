# Frame Feature Extraction — Implementation Report

**Feature:** Frame Feature Extraction
**Directory:** `features/router-frame-features/01-frame-feature-extraction/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — `frame_features` table and migration | completed |
| Task 2 — Shared YOLO loader | completed |
| Task 3 — Feature functions | completed |
| Task 4 — Feature persistence | completed |
| Task 5 — Extraction step (integration) | completed |
| Task 6 — Regression test run | completed |

## Files changed

### Added

- `database/migrations/versions/0003_create_frame_features_table.py`
- `training/router/extract_features.py`
- `training/router/feature_store.py`
- `training/router/frame_features/__init__.py`
- `training/router/frame_features/confidence.py`
- `training/router/frame_features/image_stats.py`
- `training/tests/router/__init__.py`
- `training/tests/router/frame_features/__init__.py`
- `training/tests/router/frame_features/test_confidence.py`
- `training/tests/router/frame_features/test_image_stats.py`
- `training/tests/router/test_extract_features.py`
- `training/tests/router/test_feature_store.py`
- `features/router-frame-features/01-frame-feature-extraction/spec.md`, `implementation.md`
- `features/router-frame-features/02-feature-router-experiment/spec.md`

### Edited

- `database/common/models.py`: new `FrameFeatures` model.
- `database/tests/common/test_models.py`: the round-trip test now covers `frame_features`.
- `training/datagen/simulate/yolo_inference.py`: public `load_model`, used by both the simulator and the extractor.

## Tests

### Unit

- `training/tests/router/frame_features/test_image_stats.py` (6 tests): each image statistic on synthetic images: a uniform grey image scores zero, black and white give the two brightness extremes, blurring lowers sharpness, and colour raises colourfulness. Also checks that the combined function agrees with the individual ones.
- `training/tests/router/frame_features/test_confidence.py` (6 tests): the confidence and box-area features from known inputs, the zero-detection rule, mismatched input lengths, and the adapter on a synthetic YOLO result.

### Integration

- `training/tests/router/test_feature_store.py` (5 tests, ephemeral Postgres): frame listing order and dataset scoping, the store and read-back round trip, skipping existing rows, and dataset scoping of stored rows.
- `training/tests/router/test_extract_features.py` (5 tests, ephemeral Postgres and a stub model): one row per frame with every column filled, zeros for a frame with no detections, a re-run computes nothing and leaves rows unchanged, a missing image fails before any write and names the file, and a run with nothing pending.
- `database/tests/common/test_models.py` (1 test, extended): every table round-trips, now including `frame_features`.

All tests pass: `database/` 1 passed, `training/` 108 passed (86 existing + 22 new), none skipped.

## Commits

`feature/real-model-inference..feature/router-frame-features`: the spec commit, the plan commit, and one commit per task for Tasks 1–5. Task 6 changed no files.

## Notable events

- Task 5 made a small change to `confidence.py`, which was outside its listed files. It split out the step that extracts raw confidences and box areas from a YOLO result, so the model call and the feature derivation are timed separately. The existing adapter's behaviour and tests are unchanged. The orchestrator allowed this change in the task prompt.
- Migration `0003` was only checked offline (the rendered SQL was compared with the model). Neither the migration nor the real extraction was run against the real database, per `CLAUDE.md`. The user applies the migration and runs `python -m router.extract_features --folder training/data/coco/val2017 --dataset coco_val2017` from `training/`.
