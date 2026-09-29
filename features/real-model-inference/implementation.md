# Real Model Inference for the Data-Gen Simulator — Implementation Plan

## Summary

Replaces `stub_inference`'s fully-synthetic latency/accuracy formulas with
real YOLOv8n (local, CPU) / YOLOv8x (offload, GPU) detection. Real per-frame
inference results are cached in a new Postgres table (mirroring
`scene_complexity`) and reused across runs; condition-driven overhead stays
synthetic but now adds to real measured base latency, and only ever affects
latency, never accuracy.

## Approach & Key Decisions

- **Real inference is computed once per (dataset, frame, model) and cached in
  Postgres**, structurally parallel to `scene_complexity`'s existing
  precompute-and-cache loop in `run_simulation()`. A new `_compute_missing_model_inference`
  function houses this (factored out, not inlined, to keep `run_simulation()`
  readable) — the existing complexity-scoring loop is left untouched.
- **Everything real is injected, nothing has a silent default** — `run_simulation()`
  gains required (no-default) `run_local_inference`/`run_offload_inference`/`ground_truth`
  params, matching the convention `03-generalize-image-source`'s S3 fix
  established for `image_records`/`resolve_image`. `main()` wires the real
  YOLO-backed implementations; every test injects small, fast, deterministic
  fakes. This is what keeps the test suite from ever needing to load a real
  68M-parameter model.
- **Ground-truth parsing is a new, separate module** (`simulate/ground_truth.py`),
  not folded into `image_source.py` — it parses a different array
  (`annotations`/`categories`) from the same COCO-format file for a different
  purpose (accuracy scoring, a `simulate/` concern) than `image_source.py`'s
  image-pool/path-resolution concern.
- **`stub_inference.py` is deleted, not renamed** — its role splits into two
  new pieces in a new `simulate/inference.py`: `apply_condition_overhead`
  (pure, replaces the old per-row formula, now adds to a real base latency
  instead of a constant) and the real YOLO-backed builder functions
  (`build_local_inference_fn`/`build_offload_inference_fn`). The shape
  changed enough (new signature, new real-model dependency) that this is a
  replacement, not an edit-in-place.
- **A frame with zero ground-truth boxes scores accuracy `1.0`** (resolves
  spec.md's Open Question) — vacuously nothing was missed, and this avoids a
  division-by-zero in the match-rate formula without needing to special-case
  or filter such frames out of sampling.
- **No CUDA-unavailable fallback** — the offload builder hardcodes
  `device="cuda"`. This targets a specific known dev machine (confirmed
  GPU present), not a portable deployment target; out of scope per the
  spec's existing framing of this as the offline data-gen simulator.
- **Real-model code (Task 5) is verified by hand, not by pytest** — loading a
  68M-parameter model on every test run is impractical. This mirrors how
  this repo already treats other real-dependency code paths (e.g. a real
  network download) as hand-smoke-tested rather than mocked purely for
  coverage.

## Out of Scope

Carried forward from spec.md, unchanged:

- Building `training/router/`.
- Any change to frame sampling, condition-vector sampling, presets, or
  win/loss labeling.
- Full COCO mAP via `pycocotools`.
- Fine-tuning or training either YOLO model.
- Live/on-device deployment to the actual RN app.
- Re-running or migrating existing `simulation_results` rows produced by the
  old synthetic `stub_inference`.
- Any change to how `scene_complexity` itself is computed or cached.

Further deferred by this plan:

- Refactoring the existing complexity-scoring precompute loop to match the
  new model-inference loop's factored-out shape — left as-is.
- CUDA-unavailable fallback / CPU-only mode for the offload path.

## Dependencies and Configuration

- New dependency: `ultralytics` (pulls in `torch`/`torchvision`) in
  `training/pyproject.toml`.
- First real invocation downloads pretrained weights (`yolov8n.pt`,
  `yolov8x.pt`) from Ultralytics' hosted release the first time each model is
  loaded — one-time, network-dependent, not something this plan's code needs
  to handle specially.
- New Alembic migration in `database/migrations/versions/` for the new
  `model_inference` table. Per `CLAUDE.md`, the user runs this migration
  themselves — no task in this plan runs it.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `database/common/models.py` | edit | add `ModelInference` table | new cache table for real per-(dataset, frame, model) results |
| `database/migrations/versions/0002_create_model_inference_table.py` | add | migration for the new table | schema change needs a migration |
| `database/tests/common/test_models.py` | edit | round-trip test for `ModelInference` | keeps DB-layer coverage symmetric with the other tables |
| `training/datagen/simulate/ground_truth.py` | add | parses COCO `annotations`/`categories` into per-image ground-truth boxes | new parsing concern, distinct from `image_source.py` |
| `training/tests/datagen/simulate/test_ground_truth.py` | add | tests for the above | |
| `training/datagen/simulate/inference.py` | add | `DetectionResult`, IoU accuracy scoring, condition-overhead combination, real YOLO builders | replaces `stub_inference.py` |
| `training/tests/datagen/simulate/test_inference.py` | add | tests for accuracy scoring + overhead combination (pure logic) | |
| `training/datagen/simulate/stub_inference.py` | delete | fully-synthetic model, replaced | |
| `training/tests/datagen/simulate/test_stub_inference.py` | delete | tests for the deleted module | |
| `training/datagen/config.py` | edit | remove dead accuracy-modeling coefficients and base-latency constants | per spec Requirements |
| `training/datagen/persistence.py` | edit | add `get_known_model_inference`/`store_model_inference` | mirrors `get_known_complexity`/`store_complexity_scores` |
| `training/tests/datagen/test_persistence.py` | edit | tests for the above | |
| `training/pyproject.toml` | edit | add `ultralytics` dependency | |
| `training/datagen/cli/run_simulation.py` | edit | wire real inference into the orchestration loop | core change |
| `training/tests/datagen/cli/test_run_simulation.py` | edit | update existing tests to inject fake inference; add new coverage | |

## Tasks

### Task 1 — Add the `model_inference` cache table

- **Objective:** Add the Postgres table that caches real per-(dataset,
  frame, model) inference results.
- **Files:** `database/common/models.py`, `database/migrations/versions/0002_create_model_inference_table.py`, `database/tests/common/test_models.py`.
- **Details:** Add a `ModelInference` SQLAlchemy model: `dataset` (String,
  part of composite PK), `file_name` (String, part of composite PK),
  `model_path` (reuse the existing `Label` enum — `LOCAL`/`OFFLOAD` — as the
  third part of the composite PK; do not define a new enum type), `latency_ms`
  (Float, not null), `accuracy` (Float, not null), `computed_at` (DateTime
  with timezone, defaults to `_utcnow`, not null). Add the corresponding
  Alembic migration (`upgrade`/`downgrade`, `op.create_table`/`op.drop_table`,
  following `0001_create_simulation_tables.py`'s style exactly). Critical
  detail: the `label` Postgres enum type was already created by migration
  `0001` for `simulation_results.label` — the new migration's `model_path`
  column must reference it as `sa.Enum("LOCAL", "OFFLOAD", name="label",
  create_type=False)`, not recreate it. Add a round-trip test to
  `database/tests/common/test_models.py` mirroring the existing pattern for
  the other tables (insert a row via the ephemeral-Postgres fixture, read it
  back, assert fields match).
- **Success criteria:**
  - `alembic upgrade head --sql` (offline, in `database/`) renders DDL for
    `model_inference` without attempting to recreate the `label` enum type.
  - `cd database && pytest -v` passes (new round-trip test included).
  - `cd database && ruff check . && mypy .` clean.

### Task 2 — Ground-truth parsing module

- **Objective:** Parse a COCO-format annotations file's `annotations`/`categories`
  arrays into per-image ground-truth boxes, for accuracy scoring.
- **Files:** `training/datagen/simulate/ground_truth.py`, `training/tests/datagen/simulate/test_ground_truth.py`.
- **Details:** Define `Box(NamedTuple): category_name: str, x_min: float,
  y_min: float, x_max: float, y_max: float` — used for both ground-truth and
  (in a later task) predicted boxes, since structurally they're identical (a
  labeled box). Define `load_ground_truth(annotations_path: Path) -> dict[int, list[Box]]`:
  parse the file's top-level `categories` array (`[{"id": int, "name": str}, ...]`)
  into an `id -> name` map, then its `annotations` array
  (`[{"image_id": int, "category_id": int, "bbox": [x, y, width, height]}, ...]`)
  into a `dict[image_id, list[Box]]`, converting each COCO `[x, y, width,
  height]` bbox (top-left corner + width/height) to this module's `(x_min,
  y_min, x_max, y_max)` `Box` shape (`x_max = x + width`, `y_max = y +
  height`) and resolving `category_id` to its name via the id→name map built
  above. An `image_id` with no matching `annotations` entries is simply
  absent from the returned dict — the caller treats a missing key as "zero
  ground-truth boxes for this frame" (do not insert empty-list entries
  proactively). Fail fast with clear, path-naming errors (matching
  `image_source.load_image_index`'s established convention — see
  `03-generalize-image-source`'s S1/S2 fixes for the exact style) for: the
  file not existing (`FileNotFoundError`), invalid JSON (`ValueError`), and
  either the `annotations` or `categories` top-level key missing or not a
  list (`ValueError`). Full five-case shape validation (per malformed-entry
  shape) is not required here — file-level and key-level validation is
  sufficient for this feature's scope.
- **Success criteria:**
  - Tests cover: a normal multi-image, multi-annotation fixture round-trips
    correctly (including the bbox coordinate conversion); an `image_id`
    present in `images` but absent from `annotations` is simply missing from
    the returned dict; at least one malformed-input case (missing file,
    missing `annotations` key) raises the expected clear error.
  - `cd training && pytest tests/datagen/simulate/test_ground_truth.py -v`
    passes.
  - `cd training && ruff check . && mypy .` clean.

### Task 3 — Real-inference core types and IoU-based accuracy scoring

- **Objective:** Add the pure, real-model-independent pieces of the new
  inference module: the result type and the accuracy-scoring function.
- **Files:** `training/datagen/simulate/inference.py` (new), `training/tests/datagen/simulate/test_inference.py` (new).
- **Details:** Define `DetectionResult(NamedTuple): latency_ms: float,
  accuracy: float`. Define `score_accuracy(predicted_boxes: list[ground_truth.Box],
  ground_truth_boxes: list[ground_truth.Box], *, iou_threshold: float = 0.5) -> float`:
  for each box in `ground_truth_boxes`, check whether any box in
  `predicted_boxes` with the same `category_name` has IoU >=
  `iou_threshold` against it; accuracy is the count of matched
  ground-truth boxes divided by `len(ground_truth_boxes)`. **When
  `ground_truth_boxes` is empty, return `1.0`** (resolves spec.md's Open
  Question — vacuously nothing was missed; avoids division by zero). Add a
  private `_iou(a: ground_truth.Box, b: ground_truth.Box) -> float` helper
  computing standard axis-aligned-box intersection-over-union (intersection
  area / union area; `0.0` for non-overlapping boxes).
- **Success criteria:**
  - Tests cover: a perfect match (every ground-truth box has a same-class,
    high-IoU predicted box) scores `1.0`; no matching predicted boxes scores
    `0.0`; a partial match (some but not all ground-truth boxes matched)
    scores the correct fraction; a same-position but different-category
    predicted box does not count as a match; the zero-ground-truth-boxes
    case scores `1.0`.
  - `cd training && pytest tests/datagen/simulate/test_inference.py -v` passes.
  - `cd training && ruff check . && mypy .` clean.

### Task 4 — Condition-driven latency overhead (replaces `stub_inference`)

- **Objective:** Replace `stub_inference`'s per-(frame, condition) formula
  with a function that adds synthetic condition-driven latency overhead on
  top of a real measured base latency, and remove the now-dead synthetic
  accuracy-modeling coefficients.
- **Files:** `training/datagen/simulate/inference.py` (edit), `training/tests/datagen/simulate/test_inference.py` (edit), `training/datagen/config.py` (edit); delete `training/datagen/simulate/stub_inference.py` and `training/tests/datagen/simulate/test_stub_inference.py`.
- **Details:** Add `apply_condition_overhead(condition: tuple[float, float,
  float, float], local_base: DetectionResult, offload_base: DetectionResult,
  seed: int) -> tuple[float, float, float, float]` to `inference.py` — pure,
  no I/O, same return shape `stub_inference` had
  (`local_latency_ms, local_accuracy, offload_latency_ms,
  offload_accuracy`). Reuse `stub_inference`'s exact latency-overhead
  formulas and their `seed`-derived Gaussian noise (device-load-driven term
  for local, bandwidth/network-latency/packet-loss-driven terms for
  offload), but add them to `local_base.latency_ms`/`offload_base.latency_ms`
  instead of a constant base. `local_accuracy`/`offload_accuracy` in the
  return value are `local_base.accuracy`/`offload_base.accuracy` passed
  through unchanged — no condition-driven accuracy modification at all.
  Delete `stub_inference.py` and its test file (not a rename — the
  signature and purpose changed materially). In `config.py`, remove
  `LOCAL_BASE_LATENCY_MS`, `OFFLOAD_BASE_LATENCY_MS`, `LOCAL_BASE_ACCURACY`,
  `OFFLOAD_BASE_ACCURACY`, `LOCAL_ACCURACY_NOISE_STD`,
  `OFFLOAD_ACCURACY_NOISE_STD`, `SCENE_COMPLEXITY_ACCURACY_PENALTY_COEFFICIENT`,
  `LOCAL_ACCURACY_DEVICE_LOAD_PENALTY_COEFFICIENT`,
  `OFFLOAD_ACCURACY_PACKET_LOSS_PENALTY_COEFFICIENT` (all now dead). Keep
  `LOCAL_LATENCY_DEVICE_LOAD_COEFFICIENT_MS`, `LOCAL_LATENCY_NOISE_STD_MS`,
  `OFFLOAD_LATENCY_BANDWIDTH_COEFFICIENT_MS`,
  `OFFLOAD_LATENCY_NETWORK_LATENCY_COEFFICIENT`,
  `OFFLOAD_LATENCY_PACKET_LOSS_PENALTY_COEFFICIENT_MS`,
  `OFFLOAD_LATENCY_NOISE_STD_MS` (still used by `apply_condition_overhead`).
- **Success criteria:**
  - Tests confirm: determinism (same inputs always produce the same output);
    accuracy passes through exactly unchanged from the input
    `DetectionResult`s; latency overhead direction/magnitude matches the old
    `stub_inference` formulas' shape (e.g. higher `device_load_pct` increases
    `local_latency_ms`).
  - `grep -rn "stub_inference" training/` returns nothing.
  - `cd training && pytest tests/datagen/simulate/ -v` passes.
  - `cd training && ruff check . && mypy .` clean.

### Task 5 — Real YOLO-backed inference builders

- **Objective:** Add the real, dependency-heavy pieces: functions that load
  YOLOv8n/YOLOv8x once and return a callable that runs real inference on an
  image.
- **Files:** `training/datagen/simulate/inference.py` (edit), `training/pyproject.toml` (edit).
- **Details:** Add `ultralytics` to `training/pyproject.toml`'s
  `dependencies`. Add `RunInferenceFn = Callable[[Path, list[ground_truth.Box]],
  DetectionResult]` type alias. Add `build_local_inference_fn() ->
  RunInferenceFn`: loads `YOLO("yolov8n.pt")` once via `ultralytics`, moves
  it to `device="cpu"`; returns a closure that, given an image path and that
  frame's ground-truth `Box` list, runs inference on CPU, converts each
  predicted detection to a `Box` (resolving the model's per-detection class
  index to a name via the loaded model's own class-name mapping, and box
  coordinates to `Box`'s `(x_min, y_min, x_max, y_max)` shape), measures the
  real inference latency in ms, calls `score_accuracy(predicted_boxes,
  ground_truth_boxes)`, and returns a `DetectionResult`. Add
  `build_offload_inference_fn() -> RunInferenceFn` — identical shape,
  `YOLO("yolov8x.pt")`, `device="cuda"`. Both builders are called once per
  CLI invocation (model loading happens once, not once per image) — that
  wiring happens in Task 7's `main()` changes, not here. No CUDA-availability
  fallback (see Approach & Key Decisions).
- **Success criteria:**
  - `cd training && python -c "from datagen.simulate import inference"`
    imports cleanly (no import-time errors) after `pip install -e
    ./training[dev]` picks up the new dependency.
  - Hand-smoke-test: call `build_local_inference_fn()` and
    `build_offload_inference_fn()` against 2-3 real images from
    `training/data/coco/val2017/` with that image's real ground truth (via
    `ground_truth.load_ground_truth` against the real annotations file) —
    confirm each returns a `DetectionResult` with `latency_ms > 0` and
    `accuracy` in `[0, 1]`. Not a pytest test (see Approach & Key Decisions)
    — report the actual observed values.
  - `cd training && ruff check . && mypy .` clean (add `ultralytics.*` to
    `[[tool.mypy.overrides]] ignore_missing_imports = true` in
    `pyproject.toml` if it ships without inline type stubs — check first).

### Task 6 — Persistence functions for the model-inference cache

- **Objective:** Add read/write access to the new `model_inference` table.
- **Files:** `training/datagen/persistence.py` (edit), `training/tests/datagen/test_persistence.py` (edit).
- **Details:** Add `get_known_model_inference(engine: Engine, dataset: str)
  -> dict[str, dict[Label, DetectionResult]]`: read all `ModelInference` rows
  for `dataset`, ordered by `file_name` (same determinism reasoning as
  `get_known_complexity`), return nested as `{file_name: {Label.LOCAL:
  DetectionResult(...), Label.OFFLOAD: DetectionResult(...)}}` — a
  `file_name` key is present only for whichever `model_path` values actually
  have a persisted row (may have just one of the two paths cached, not
  necessarily both). Add `store_model_inference(engine: Engine, dataset:
  str, results: dict[str, dict[Label, DetectionResult]]) -> None`: insert
  new `ModelInference` rows, skipping any `(dataset, file_name, model_path)`
  combination already present — mirror `store_complexity_scores`'s
  already-known-check shape exactly.
- **Success criteria:**
  - Tests cover: round-trip (store then get returns matching values for both
    `Label` values on a file), skip-already-known behavior (storing a
    `(file_name, model_path)` pair already present does not insert a
    duplicate or overwrite), empty dataset returns an empty dict.
  - `cd training && pytest tests/datagen/test_persistence.py -v` passes.
  - `cd training && ruff check . && mypy .` clean.

### Task 7 — Wire real inference into `run_simulation`'s orchestration

- **Objective:** Replace the per-(frame, condition) `stub_inference` call
  with real cached per-frame inference plus synthetic condition overhead,
  and update `main()` to build the real dependencies.
- **Files:** `training/datagen/cli/run_simulation.py` (edit), `training/tests/datagen/cli/test_run_simulation.py` (edit).
- **Details:**
  - Add three new required (no-default) keyword-only params to
    `run_simulation()`: `run_local_inference: inference.RunInferenceFn`,
    `run_offload_inference: inference.RunInferenceFn`, `ground_truth:
    dict[int, list[ground_truth.Box]]`.
  - Add a new function `_compute_missing_model_inference(engine, dataset,
    image_records, resolve_image, ground_truth, run_local_inference,
    run_offload_inference) -> dict[str, dict[Label, DetectionResult]]`
    (factored out, not inlined — see Approach & Key Decisions). For each
    `ImageRecord` not already covered by both `Label.LOCAL` and
    `Label.OFFLOAD` in the known-cache dict (from
    `persistence.get_known_model_inference`), resolve its path via
    `resolve_image`, look up its ground truth via
    `ground_truth.get(record.image_id, [])`, call both
    `run_local_inference(image_path, frame_ground_truth)` and
    `run_offload_inference(image_path, frame_ground_truth)`, accumulate into
    a pending batch, and flush to Postgres via
    `persistence.store_model_inference` in batches of
    `COMPLEXITY_SCORE_FLUSH_BATCH_SIZE` (reuse the existing constant — same
    crash-resilience reasoning applies here, no need for a second constant
    with the same value), logging progress the same way the existing
    complexity-scoring loop does. Called from `run_simulation()` right after
    the existing complexity-scoring loop.
  - In the per-(frame, condition) loop, replace `stub_inference(condition,
    frame_complexity, row_seed)` with a lookup of that frame's cached
    `DetectionResult`s (`all_model_inference[frame_id][Label.LOCAL]`, same
    for `Label.OFFLOAD`) and a call to
    `inference.apply_condition_overhead(condition, local_base, offload_base,
    row_seed)`.
  - In `main()`: after building `image_records`, also call `frame_ground_truth
    = ground_truth.load_ground_truth(args.annotations)` (same file,
    different array, both loaded once). Build `run_local_inference =
    inference.build_local_inference_fn()` and `run_offload_inference =
    inference.build_offload_inference_fn()` (each loads its model exactly
    once per CLI invocation). Pass `run_local_inference`,
    `run_offload_inference`, and `ground_truth=frame_ground_truth` into the
    `run_simulation(...)` call.
  - Update every existing `run_simulation(...)` call in
    `test_run_simulation.py` to inject fake `run_local_inference`/
    `run_offload_inference` (small, fast, deterministic — e.g. a closure
    returning a fixed or path-derived `DetectionResult`, no real model) and
    a `ground_truth` dict (`{}` is sufficient for tests that don't assert on
    accuracy values derived from real ground truth).
  - Add one new test confirming spec.md's condition-invariant-accuracy
    acceptance criterion: for a given frame, every `simulation_results` row
    across all sampled conditions has the same `local_accuracy`/
    `offload_accuracy` (only latency varies). Add one new test confirming a
    second `run_simulation()` call against the same dataset does not call
    the fake inference functions again for already-covered frames (mirrors
    `test_run_simulation_is_reproducible_and_reuses_known_complexity`'s
    pattern, via a call-counting fake).
- **Success criteria:**
  - `run-simulation --help` still works.
  - Every existing test in `test_run_simulation.py` passes with the fakes
    injected, assertions otherwise unchanged.
  - The two new tests pass.
  - `cd training && pytest -q && ruff check . && mypy .` clean.

### Task 8 — Regression test run

- **Objective:** Run every test that exercises code added or modified in
  this plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd database && pytest -q && ruff check . && mypy .` clean.
  - `cd training && pytest -q && ruff check . && mypy .` clean (ask the user
    to start Postgres first if the ephemeral-test-DB fixture can't connect;
    never start it yourself, per `CLAUDE.md`).
