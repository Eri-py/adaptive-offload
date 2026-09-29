# Frame Feature Extraction — Implementation Plan

## Summary

Add a `frame_features` Postgres table and a runnable extraction step that, for
every `coco_val2017` frame Postgres knows about, computes five cheap image
statistics and six YOLOv8n confidence features and stores one row per frame.
Re-runs fill in only missing frames, and the step prints average per-frame
compute times.

## Approach & Key Decisions

- **The frame list comes from Postgres, not the folder.** The frames to
  process are the dataset's `scene_complexity` rows (all 5,000). Every image
  is checked for presence before anything is computed, so a missing image
  fails the run up front with its name (spec AC). This keeps Postgres as the
  record of which frames exist (coding-guidelines "Database").
- **Code lives in `training/router/`, run as a module, not a third registered
  CLI.** `.claude/coding-guidelines.md` caps `[project.scripts]` at
  `run-simulation` and `score-complexity`. These features are router inputs,
  so the step is `python -m router.extract_features`, like the existing
  `router.baseline`/`router.analyze` scripts. Its arguments mirror
  `score-complexity` (`--folder`, `--dataset`).
- **Same YOLOv8n settings as the stored accuracy, by sharing the loader.**
  `datagen/simulate/yolo_inference.py` gains a public `load_model(weights_path,
  device)` holding the existing weights check, load and warm-up. The
  simulator's builder and the extractor both use it. The extractor then makes
  the identical call (`model(image_path, device="cpu", verbose=False)`) with
  `config.LOCAL_MODEL_WEIGHTS_PATH`, so detections match what the local
  accuracy was scored on. The simulator's behaviour is unchanged.
- **Pure feature functions, injected model.** The image statistics and the
  confidence features are pure functions over arrays. The orchestration takes
  a prediction function as a parameter, so its tests use a fast stub, matching
  how `run_simulation`'s tests avoid loading real YOLO. The real-weights check
  runs as a one-off verification in Task 3, not as a pytest test.
- **Rows are flushed in batches of 100**, so an interrupted 5,000-frame run
  keeps its progress, and the resume behaviour makes the re-run cheap. This
  follows the earlier datagen review finding about batching the scoring pass.

## Out of Scope

- Analysing the features, training routers, or updating `findings.md`
  (spec 02).
- Recomputing edge density, or changing any stored accuracy or latency.
- Offload-model output features.
- Measuring feature cost on the phone.
- New simulation runs.
- App or server changes. `features/server-served-benchmark/` stays parked.
- Running the migration against the real database. The user does this.

## Dependencies and Configuration

- **New packages:** none. `opencv-python`, `numpy` and `ultralytics` are
  already in `training/`'s dependencies.
- **DB migration:** `database/migrations/versions/0003_create_frame_features_table.py`
  creates `frame_features`. The user applies it before running the step
  (`CLAUDE.md` "Infrastructure").
- **Config:** none new. It uses `training/.env`'s `DATABASE_URL`, like the
  other training scripts.
- **Feature definitions**, fixed here so that spec 02 reads unambiguous
  values:

  | Feature | Definition |
  |---|---|
  | `sharpness` | variance of the Laplacian of the grayscale image |
  | `brightness` | mean grayscale intensity / 255, in [0, 1] |
  | `contrast` | standard deviation of the grayscale intensity / 255 (RMS contrast) |
  | `colorfulness` | Hasler–Süsstrunk colourfulness metric on RGB |
  | `entropy` | Shannon entropy (bits) of the 256-bin grayscale histogram, in [0, 8] |
  | `detection_count` | number of YOLOv8n detections (integer) |
  | `max_confidence`, `mean_confidence`, `min_confidence` | over detection confidences |
  | `mean_box_area`, `min_box_area` | detection box area / image area (normalised `w * h`) |

  With zero detections, the count is 0 and all confidence and box-area
  features are 0.0.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `database/common/models.py` | edit | Add the `FrameFeatures` model | The shared DB layer owns every table |
| `database/migrations/versions/0003_create_frame_features_table.py` | add | Create `frame_features` | Schema changes go through Alembic |
| `database/tests/common/test_models.py` | edit | Round-trip test for `FrameFeatures` | Covers the new table like the others |
| `training/datagen/simulate/yolo_inference.py` | edit | Public `load_model` shared with the extractor | One place defines the local model's load and settings |
| `training/router/frame_features/__init__.py` | add | Package marker | New subpackage |
| `training/router/frame_features/image_stats.py` | add | The five image statistics | Pure, testable feature code |
| `training/router/frame_features/confidence.py` | add | The six confidence features, plus an adapter from YOLO `Results` | Pure, testable feature code |
| `training/router/feature_store.py` | add | Read known frames, and store feature rows | Persistence for the new table |
| `training/router/extract_features.py` | add | Orchestration plus `main()` | The runnable step |
| `training/tests/router/__init__.py` | add | Package marker | New test tree mirroring `router/` |
| `training/tests/router/frame_features/__init__.py` | add | Package marker | Mirrors source |
| `training/tests/router/frame_features/test_image_stats.py` | add | Image-statistics tests | Pipeline-output correctness |
| `training/tests/router/frame_features/test_confidence.py` | add | Confidence-feature tests | Pipeline-output correctness |
| `training/tests/router/test_feature_store.py` | add | Persistence tests (real ephemeral Postgres) | Pipeline-output correctness |
| `training/tests/router/test_extract_features.py` | add | Orchestration integration tests | Wiring, resume and failure behaviour |

## Tasks

### Task 1 — `frame_features` table and migration

- **Objective:** Add the `FrameFeatures` model and a matching Alembic
  migration `0003`.
- **Files:** `database/common/models.py`,
  `database/migrations/versions/0003_create_frame_features_table.py`,
  `database/tests/common/test_models.py`
- **Details:**
  - Table `frame_features` with primary key (`dataset` str, `file_name` str).
  - Float columns: `sharpness`, `brightness`, `contrast`, `colorfulness`,
    `entropy`, `max_confidence`, `mean_confidence`, `min_confidence`,
    `mean_box_area`, `min_box_area`.
  - Integer column: `detection_count`.
  - `computed_at` (timezone-aware, default `_utcnow`), like `SceneComplexity`.
  - All columns are non-nullable.
  - The migration uses `down_revision = "0002"`. Follow `0002`'s docstring
    and style. It uses no enums.
  - Extend the existing round-trip test, or add one beside it, so a
    `FrameFeatures` row is written and read back.
  - Do not run the migration against the real database. Check it with
    `alembic upgrade 0002:0003 --sql` (offline render) from `database/`, and
    compare the rendered DDL with the model.
- **Success criteria:**
  - `cd database && ruff check . && mypy . && pytest` passes.
  - The offline SQL render shows `CREATE TABLE frame_features` with the
    columns and primary key above.

### Task 2 — Shared YOLO loader

- **Objective:** Expose the simulator's model loading as a public
  `load_model(weights_path: Path, device: str) -> YOLO` in
  `training/datagen/simulate/yolo_inference.py`.
- **Files:** `training/datagen/simulate/yolo_inference.py`
- **Details:**
  - `load_model` does exactly what `_build_inference_fn` does today before
    defining its closure: the weights-file check (`_require_weights_file`),
    `YOLO(weights_path)`, `.to(device)`, and the discarded warm-up call.
  - `_build_inference_fn` calls `load_model` and keeps its closure unchanged.
  - Pure refactor: no behaviour change.
- **Success criteria:**
  - `cd training && ruff check . && pytest` passes.
  - `mypy` reports no new errors in this file (there are 9 known errors
    elsewhere in `training/`).

### Task 3 — Feature functions

- **Objective:** Implement the image statistics and confidence features as
  pure functions, plus a thin adapter from a YOLO `Results` object.
- **Files:** `training/router/frame_features/__init__.py`,
  `image_stats.py`, `confidence.py`,
  `training/tests/router/__init__.py`,
  `training/tests/router/frame_features/__init__.py`,
  `test_image_stats.py`, `test_confidence.py`
- **Details:**
  - `image_stats.py`: one function per statistic in the plan's feature
    definitions table, plus one that returns all five for an image path or
    BGR array (loaded with `cv2.imread`, as in
    `datagen/sampling/complexity.py`).
  - `confidence.py`: a pure function from `(confidences, normalised box
    areas)` sequences to the six features, with the zero-detection rule. An
    adapter extracts those sequences from `Results`, using
    `boxes.conf` and `boxes.xywhn` (area = w * h).
  - Tests use synthetic images and arrays:
    - A uniform grey image has contrast 0, entropy 0 and colourfulness 0.
    - A black image has brightness 0; a white image has brightness 1.
    - A blurred copy of a random-noise image has lower sharpness.
    - A saturated two-colour image has higher colourfulness than grey.
    - For confidence features: known inputs give known max, mean and min,
      and zero detections give all zeros.
  - Verification only, not a test: on 3 real images from
    `training/data/coco/val2017/`, check that the confidences the adapter
    gets from `load_model(config.LOCAL_MODEL_WEIGHTS_PATH, "cpu")` followed by
    `model(path, device="cpu", verbose=False)` equal those from a second,
    independent run with the same call. Record the result in the task's
    progress notes.
- **Success criteria:**
  - The new tests pass, and ruff and mypy are clean for the new files.
  - The real-image confidence comparison matches.

### Task 4 — Feature persistence

- **Objective:** Read frame lists and store feature rows for
  `frame_features`.
- **Files:** `training/router/feature_store.py`,
  `training/tests/router/test_feature_store.py`
- **Details:**
  - Functions to:
    - list the dataset's frame file names from `scene_complexity`, ordered
      by `file_name` (mirroring `datagen.persistence.get_known_complexity`)
    - list the file names already in `frame_features` for the dataset
    - insert new feature rows, skipping file names already present (like
      `store_complexity_scores`)
  - Tests use the existing `postgres_engine` fixture from
    `training/tests/conftest.py`. Cover round-trip, the skip-existing
    behaviour, and dataset scoping.
- **Success criteria:**
  - The new tests pass against ephemeral Postgres, and ruff and mypy are
    clean for the new files.

### Task 5 — Extraction step (integration)

- **Objective:** Wire the feature functions and the store into
  `router/extract_features.py`, runnable as
  `python -m router.extract_features --folder <images> --dataset <name>`.
- **Files:** `training/router/extract_features.py`,
  `training/tests/router/test_extract_features.py`
- **Details:**
  - An orchestration function takes `engine`, `dataset`, `folder`, and a
    `predict(image_path) -> Results`-like callable that returns the
    confidence sequences. It:
    1. Lists the dataset's frames and the frames already stored, and
       computes only the rest.
    2. Checks that every pending frame's image exists, and raises
       `FileNotFoundError` listing the missing names before computing
       anything.
    3. For each frame, times the image statistics, the model call and the
       confidence derivation separately.
    4. Stores rows in batches of 100.
    5. Returns and prints the per-frame average for each of the three
       timings, or says there was nothing to compute.
  - `main()` loads `training/.env` (as `score_complexity.main` does), builds
    the real predictor with
    `load_model(config.LOCAL_MODEL_WEIGHTS_PATH, "cpu")` and
    `model(path, device="cpu", verbose=False)`, and calls the orchestration
    function. Missing weights fail through `load_model`'s existing check.
  - Integration tests (ephemeral Postgres, a stub predictor, and a `tmp_path`
    folder of small generated JPEGs) cover:
    - one row per frame, with every column populated
    - a frame the stub gives zero detections is stored with zeros
    - a second run computes nothing new and leaves existing rows unchanged
    - a missing image raises before any row is written, naming the file
    - the timing summary is returned
- **Success criteria:**
  - The new tests pass, and ruff and mypy are clean for the new files.
  - `python -m router.extract_features --help` works.

### Task 6 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd database && pytest` and `cd training && pytest` pass (including the
    previous 86 training tests).
  - ruff is clean in both packages, and mypy shows no errors beyond
    `training/`'s 9 known ones.
