# Learnings — real-model-inference

## Task 1

- The ephemeral test DB fixture (`database/common/testing.py`) builds schema via
  `Base.metadata.create_all`, not by running migrations. Reusing `Enum(Label)` for a
  second column (`ModelInference.model_path`) across a different table works fine there
  with no extra config: `create_all` calls `checkfirst=True` per column's enum type, so
  the second column's type creation is a no-op once the first (`SimulationResult.label`)
  has created the Postgres `label` type. Only the *migration* script needs the explicit
  `create_type=False` guard, since Alembic's `--sql` offline rendering has no DB to check
  against and would otherwise emit a second `CREATE TYPE label` and fail on a real apply.
- `alembic upgrade head --sql` from `database/` is a reliable, fully offline way to verify
  a new migration's DDL (including confirming it does *not* reissue `CREATE TYPE`) without
  touching Postgres at all — same pattern migration `0001` used.
- Composite PKs with an enum member as one of the parts round-trip fine through
  `session.get(Model, (a, b, Label.X))`.

## Task 2

- Initially over-built `load_ground_truth`'s per-entry validation (checking each
  `categories`/`annotations` entry's shape, plus a `category_id`-not-found check),
  mirroring `image_source.load_image_index`'s per-entry checks too literally. The
  plan explicitly scopes this out ("Full five-case shape validation ... is not
  required here — file-level and key-level validation is sufficient"), and adding
  it anyway means untested code paths (extra `ValueError` branches with no test
  coverage) — trimmed back to only file-existence, JSON-validity, and top-level
  `annotations`/`categories` key-presence-and-list-type checks, letting a genuinely
  malformed entry surface as a plain `KeyError`/`ValueError` from tuple-unpacking
  instead of a custom message. Worth reading a task's explicit scope-narrowing
  language carefully before pattern-matching a sibling module's convention wholesale.
- `ruff`'s line-length limit in `training/` is 100 chars (not the more common 88/120) —
  a single-line multi-kwarg `Box(...)` assertion tripped `E501`; had to wrap it.

## Task 3

- `score_accuracy`/`_iou` are pure and only depend on `ground_truth.Box`, so no DB or
  fixture setup was needed for tests — plain module-level `Box` constants sufficed.
- Picking IoU test fixtures by exact fraction (rather than "clearly high"/"clearly
  low") makes the below-threshold and custom-threshold tests self-documenting: two
  10x10 boxes offset by (5, 5) give intersection 25 / union 175 = IoU ≈ 0.143, which
  is below the default 0.5 threshold but above a relaxed 0.1 one — one fixture pair
  covers both the "doesn't match by default" and "does match with a lower threshold"
  cases with a comment stating the exact number instead of a vague description.
- No surprises versus Task 2's setup: same venv activation (`source ../.venv/bin/activate`
  from `training/`), same ruff (100-char line length) and mypy (`strict = true`)
  config applied cleanly with no adjustments needed.

## Task 4

- `stub_inference`'s draw order was `local_latency_noise, offload_latency_noise,
  local_accuracy_noise, offload_accuracy_noise`. Since `apply_condition_overhead` drops
  accuracy noise entirely, its draw order (`local_latency_noise, offload_latency_noise`)
  is a prefix of the old order — same `rng.normal` calls in the same sequence for the
  two draws that survive, so seed-derived latency noise values are bit-identical to what
  `stub_inference` would have produced for the same seed, condition, and base latency.
  Not required by the task, but worth knowing if anyone compares old vs. new latency
  distributions.
- Left an explicit deliberate gap in the grep sweep: `training/datagen/cli/run_simulation.py`
  (live `from datagen.simulate.stub_inference import stub_inference` import) and
  `training/datagen/simulate/labeling.py` (a docstring comment naming `stub_inference`)
  both still reference the deleted module by name — both out of this task's `Files` list,
  both Task 7's job to rewire. Confirmed via `mypy` and `pytest` (full suite, not just the
  scoped `tests/datagen/simulate/` directory) that this produces exactly one failure point:
  `tests/datagen/cli/test_run_simulation.py` fails to collect
  (`ModuleNotFoundError: No module named 'datagen.simulate.stub_inference'`), and `mypy .`
  reports exactly one `import-not-found` error, both pointing at
  `datagen/cli/run_simulation.py:57`. Nothing else in the full suite is affected — worth
  the orchestrator double-checking this stays a single, expected, localized failure when
  Task 7 lands, rather than spreading.
- Reworded my own new `apply_condition_overhead` docstring to avoid literally saying
  "`stub_inference`" (used "the old formula" instead), so the `grep -rn "stub_inference"
  training/` sweep's only non-`egg-info` hits are the two known out-of-scope files above —
  makes it obvious at a glance that nothing *I* touched still references the deleted name.
- `training/training.egg-info/SOURCES.txt` (a build artifact, not source) also still lists
  `datagen/simulate/stub_inference.py`; harmless and regenerated on next build, not worth
  touching.

## Task 5

- `pip install ultralytics` pulls in `torch` + a full CUDA/cuDNN wheel set (cublas,
  cudnn, cufft, cusparse, cusolver, curand, nvjitlink, nvtx, ...) as transitive
  dependencies — several GB total, took well over an hour on this connection even
  though each individual wheel download itself wasn't slow. Not a hang; just a
  genuinely large one-time install. Confirmed progress throughout by watching the
  pip process's open file descriptors (`/proc/<pid>/fd`, which wheel it currently
  had open for unpacking) and `/proc/<pid>/io`'s `write_bytes` counter climbing,
  rather than assuming a stall from silent stdout (pip's progress bar doesn't
  flush cleanly through a piped/redirected `tail`).
- `ultralytics` (8.4.159, resolved from `ultralytics>=8.0`) ships its own
  `py.typed` marker and is fully inline-typed — the task's own guidance to
  "add `ultralytics.*` to `ignore_missing_imports` if it ships without inline
  type stubs" turned out not to apply; I added the override first, then verified
  ultralytics actually has `py.typed` and removed it again since it was a no-op
  at best and would have silently masked real type errors in this dependency
  going forward. Two real (non-missing-stub) mypy findings came from this instead:
  1. `from ultralytics import YOLO` triggers `[attr-defined]` under `strict`'s
     implied `no_implicit_reexport`, because `ultralytics/__init__.py` builds its
     `__all__` via `*MODELS` (a runtime tuple unpack) rather than a literal string
     list/tuple — mypy's re-export check doesn't statically evaluate that, even
     though `"YOLO"` genuinely ends up in `__all__` at runtime. Resolved with a
     narrow `# type: ignore[attr-defined]` plus a comment explaining why (not a
     missing-stub issue).
  2. `Model.__call__`'s real return type is
     `Iterator[Results | Tensor] | list[Results] | list[Tensor]` (a single
     non-overloaded signature covering both `stream=True` and `stream=False`),
     so `results[0]` doesn't type-check directly. Since this module never passes
     `stream=True`, `cast(list[Results], model(...))` narrows it correctly and
     documents the assumption inline. Also needed `assert boxes is not None`
     in `_to_boxes` since `Results.boxes` is typed `Boxes | None` (only `None`
     for non-detection tasks — segmentation/pose/classification — which this
     module never loads).
- Chose wall-clock `time.perf_counter()` around the `model(...)` call itself
  over `results[0].speed['inference']` for `latency_ms`: the task's own guidance
  flagged this as a judgment call, and wall-clock is the more honest "what would
  a real caller actually observe" number for this simulator's purposes (it also
  includes ultralytics' own pre/post-processing inside the call, not just the
  raw forward pass, which is closer to what a real on-device or offload caller
  would experience end-to-end).
- Hand-smoke-tested `build_local_inference_fn()`/`build_offload_inference_fn()`
  against 3 real `val2017` images with real ground truth loaded via
  `ground_truth.load_ground_truth`, picking images_id 289343, 61471, 472375
  (each has 3-4 real ground-truth boxes, categories including dog/person/bench/
  bicycle/bottle/toilet/motorcycle/cup). Observed real `DetectionResult`s:
  - image 289343 (gt: dog, person, bench, bicycle) — local: latency 701.8ms,
    accuracy 0.75; offload: latency 686.7ms, accuracy 1.0. (This first call's
    latency for both paths includes one-time CUDA-context/lazy-init warmup
    inside the wall-clock window — not representative of steady-state
    per-frame cost; see the next two images.)
  - image 61471 (gt: dog, bottle, toilet) — local: latency 29.0ms, accuracy
    0.667; offload: latency 23.5ms, accuracy 1.0.
  - image 472375 (gt: dog, motorcycle, cup, cup) — local: latency 27.6ms,
    accuracy 0.25; offload: latency 21.5ms, accuracy 0.5.
  All six results have `latency_ms > 0` and `accuracy` in `[0, 1]`, and accuracy
  values genuinely vary (0.25 to 1.0, never uniformly 0.0 or uniformly 1.0
  across every image) — confirms COCO category names from `model.names` really
  do match `ground_truth.Box.category_name` values for `score_accuracy`'s
  name-based matching, not silently broken/always-zero or always-perfect
  matching.
- `YOLO("yolov8n.pt")`/`YOLO("yolov8x.pt")` download their pretrained weights
  into the current working directory (`training/yolov8n.pt`, `training/yolov8x.pt`
  in this run, ~6.2MB and ~130.5MB respectively) on first use, not into a
  `~/.cache`-style location as the task's practical guidance suggested — worth
  the orchestrator/Task 7 deciding whether these should be `.gitignore`d (they
  currently aren't tracked since nothing has `git add`ed them, but nothing
  currently prevents an accidental `git add .` from picking them up).
- `model.to(device)` (called once, at builder time, before the closure is
  returned) works cleanly for both `"cpu"` and `"cuda"`; also passing `device=`
  again per-call (`model(image_path, device="cpu"/"cuda", ...)`) is redundant
  with the `.to()` call but harmless — kept both since the task's practical
  guidance explicitly showed the per-call `device=` form and it costs nothing
  to be explicit at the call site too.

## Task 6

- `get_known_model_inference`/`store_model_inference` follow `get_known_complexity`/
  `store_complexity_scores`'s shape exactly, just keyed on the composite
  `(file_name, model_path)` pair instead of `file_name` alone: the already-known
  check selects both `ModelInference.file_name` and `ModelInference.model_path`
  columns and builds a `set` of the resulting `Row` tuples — `Row` (SQLAlchemy's
  `select(...).all()` row type) compares equal to a plain Python tuple, so
  `(file_name, model_path) not in known_pairs` works directly with no need to
  convert each `Row` to a tuple first.
- `persistence.py` now imports `DetectionResult` from `datagen.simulate.inference`
  — the first cross-import from `persistence.py` into `datagen.simulate`. No
  circular-import issue: `inference.py` doesn't import `persistence`.
- Verified the task's stated pre-existing-breakage boundary still holds exactly
  as scoped: `ruff check .` is fully clean (after fixing one new `E501` from an
  over-100-char one-line docstring, wrapped into a multi-line docstring instead),
  and `mypy .` reports exactly the same single pre-existing `import-not-found`
  error at `datagen/cli/run_simulation.py:57` (the deleted `stub_inference`
  module) — nothing new introduced by this task's changes. Used the scoped test
  path `pytest tests/datagen/test_persistence.py -v` per the task's instruction,
  not a bare `pytest`, since the full suite still fails to collect due to that
  same known `run_simulation.py` import (Task 7's job, untouched here).
- Test coverage added: round-trip storing both `Label` values for one file and
  reading them back nested correctly; skip-already-known behavior (re-storing an
  existing `(file_name, model_path)` pair with different values doesn't overwrite
  it, while a genuinely new pair on the same file still gets inserted alongside
  it); and an empty/unknown dataset returning `{}`. All 11 tests in
  `test_persistence.py` (3 new + 8 pre-existing) pass against real ephemeral
  Postgres.

## Task 7

- The broken `stub_inference` import was the only thing blocking test
  collection for the whole `training/` suite (as Task 6's note above
  confirmed) — fixing it in `run_simulation.py` alone was enough to bring the
  full `pytest -q` back from "fails to collect" to "93 passed", no other file
  needed touching.
- `_compute_missing_model_inference` reuses `COMPLEXITY_SCORE_FLUSH_BATCH_SIZE`
  as instructed, but that has a side effect worth flagging for anyone reading
  test output: because the constant is a shared module attribute (not
  captured per-function at def time), monkeypatching it in a test affects
  *both* the complexity-scoring loop and the model-inference loop's flush
  cadence at once. `test_complexity_scoring_logs_progress_every_batch`
  originally asserted on *all* `INFO`-level log records unfiltered by
  message; once `_compute_missing_model_inference`'s own "Computed inference
  for X/Y images." progress lines started firing at the same cadence, that
  assertion broke (8 messages instead of the expected 4). Fixed by filtering
  `progress_messages` to only messages starting with `"Scored"` — keeps the
  test scoped to what it's actually named/documented to check (the
  complexity-scoring loop's own progress logging) rather than accidentally
  asserting on a second loop's unrelated log lines.
- `test_run_simulation_creates_expected_rows_with_full_linkage`'s
  `call_count["n"] == POOL_SIZE` assertion on the shared `resolve_image` fake
  broke the same way, for a different reason: every pool image that's new to
  *both* caches now gets `resolve_image` called on it twice — once by the
  complexity-scoring loop, once by `_compute_missing_model_inference` — since
  each loop independently needs the resolved path for its own work (scoring
  vs. running inference) and neither shares its resolution with the other.
  Updated the assertion to `POOL_SIZE * 2`. This is expected/correct
  behavior per the task's design (each loop calls `resolve_image` itself),
  not a bug — `resolve_image` is a cheap local path lookup, not an expensive
  I/O op worth caching across loops for this.
- A parameter literally named `ground_truth` (shadowing the module import
  `from datagen.simulate import ground_truth, inference`) works fine both at
  runtime and under `mypy --strict`, given `from __future__ import
  annotations`: the annotation `ground_truth: dict[int, list[ground_truth.Box]]`
  resolves `ground_truth.Box` against the *module* in the enclosing scope
  (annotations are deferred strings, evaluated by mypy against the module's
  global namespace), while the function *body* only ever needs the parameter
  (the dict), never the module — no actual name collision in practice. Used
  this for both `run_simulation`'s and `_compute_missing_model_inference`'s
  new `ground_truth` parameter, matching the plan's specified parameter name
  exactly rather than renaming it to dodge the shadow.
- `_compute_missing_model_inference` recomputes *both* `Label.LOCAL` and
  `Label.OFFLOAD` for a file even if only one of the two is missing from the
  cache (per the task's explicit instruction) — `store_model_inference`'s
  existing per-`(file_name, model_path)` dedup silently no-ops the
  already-cached one on insert, so this never double-persists, it just
  costs one redundant inference call in the (expected to be rare) partial-
  cache case. Not optimized further since the task called this out
  explicitly as the intended shape.
- New tests: `test_condition_never_changes_accuracy_for_a_given_frame` groups
  `simulation_results` rows by `frame_id` and asserts each frame's
  `local_accuracy`/`offload_accuracy` form a single-element set across all
  `CONDITION_VECTOR_COUNT` sampled conditions for that frame, while also
  asserting latency actually *does* vary (guards against a vacuously-passing
  fake). `test_run_simulation_reuses_known_model_inference_on_second_call`
  mirrors the existing complexity-cache-reuse test's shape: runs
  `run_simulation()` twice against the same pool with call-counting
  `run_local_inference`/`run_offload_inference` fakes, and asserts the second
  run's fakes are called zero times, plus that `model_inference` row count
  stays at `POOL_SIZE * 2` (not doubled) after the second run.
- Full gate confirmed clean: `pytest -q` → 93 passed (91 pre-existing incl. 2
  fixed above, + 2 new), `ruff check .` → all checks passed, `mypy .` →
  success on all 42 source files (the Task 4-6 note's single pre-existing
  `import-not-found` error at `run_simulation.py:57` is gone, not just
  reduced — confirmed by re-running after all edits landed). `run-simulation
  --help` also confirmed working.
  Postgres.

## Review fix — B1

- First-call latency inflation was real and large: measured on this machine
  (RTX 5070) before the fix, `build_local_inference_fn()`/
  `build_offload_inference_fn()`'s returned closure's *first* call paid for
  `ultralytics`' lazy weight/backend init (and CUDA context setup for the
  offload path) inside the timed wall-clock window, since nothing had
  touched the model between `model.to(...)` and the first real
  `model(...)` call.
- Fix: one discarded warmup inference — `model(np.zeros((640, 640, 3),
  dtype=np.uint8), device=..., verbose=False)` — right after `model.to(...)`
  in both `build_local_inference_fn` and `build_offload_inference_fn`,
  before the closure is defined/returned. This pays the one-time init cost
  during the *builder* call (outside any per-frame timing), not during the
  first `run_inference` call.
- Hand-verified before/after with a real image (`000000000139.jpg`) against
  real ground truth from `instances_val2017.json`, run through the fixed
  code:
  - LOCAL (CPU, yolov8n): first call 35.9 ms, second call (different image)
    32.5 ms — both in the same tens-of-ms ballpark, no outlier.
  - OFFLOAD (CUDA, yolov8x): first call 55.9 ms, second call 24.9 ms — also
    both tens-of-ms, no outlier. (First slightly higher than second here,
    plausibly image-content-dependent NMS/decode cost, not init — nowhere
    near the ~700 ms the finding reported pre-fix.)
  - The one-time cost didn't vanish, it moved: `build_local_inference_fn()`
    itself took ~1.4 s and `build_offload_inference_fn()` ~1.7 s
    (warmup inference plus model load), which is fine since builders run
    once per simulator invocation, not once per frame.
- No real (non-ephemeral) `model_inference` cache existed to clean up — the
  only persisted rows anywhere were from ephemeral pytest fixtures that
  tear themselves down — so the finding's "delete affected rows" step was
  correctly out of scope for this fix per the task instructions.
- Full gate re-confirmed clean after the fix: `pytest -q` → 93 passed
  (unaffected, since the existing suite uses fake inference functions, not
  real models), `ruff check .` → all checks passed, `mypy .` → success on
  all 42 source files.

## Review fix — S1

- `all_complexity` in `run_simulation` accumulates from `get_known_complexity`
  (every persisted `scene_complexity` row for the dataset name, which can
  span a wider or different pool than the current invocation's
  `image_records`) plus any newly scored images from the current pool. It
  was passed straight into `stratified_sample`, so a stale/wider history
  under the same `--dataset` could hand back a frame with no matching
  `all_model_inference` entry — a bare `KeyError` at
  `all_model_inference[frame_id][Label.LOCAL]`, after the `simulation_runs`
  row had already been created.
- Fix: build `pool_file_names = {record.file_name for record in
  image_records}` and filter `all_complexity` down to just those names
  (`pool_complexity`) before calling `stratified_sample`. This is correct
  behavior unconditionally — sampling should only ever draw from the
  current pool — not just a guard against the mismatch case, so no
  fail-early/error-message branch was needed.
- Regression test
  (`test_run_simulation_excludes_out_of_pool_frames_from_sampling`) seeds an
  extra `SceneComplexity` row directly via a `Session(postgres_engine)`
  block (mirroring how other tests in this file seed data directly) for a
  file name never in `_write_fake_pool`'s pool, under the same
  `config.DATASET_NAME` the default-dataset test runs use, with a
  mid-range `scene_complexity` value (0.5) so it would be a plausible pick
  if the filter were missing rather than one `stratified_sample` would have
  skipped anyway. Asserts `run_simulation()` completes normally and that
  the out-of-pool file name never appears among the persisted
  `simulation_results.frame_id` values.
- Full gate after the fix: `pytest -q` → 94 passed (was 93; +1 for the new
  regression test), `ruff check .` → all checks passed, `mypy .` → success
  on all 42 source files.

## Review fix — S2

- `load_ground_truth` (`training/datagen/simulate/ground_truth.py`) was
  turning every COCO `annotations` entry into a `Box` to match against,
  including `iscrowd=1` entries. A crowd annotation is a single box drawn
  around a whole cluster of instances (e.g. "a crowd of people"), which
  standard COCO evaluation excludes from per-instance matching — a
  detector's normal per-object boxes essentially never hit IoU ≥ 0.5
  against one, so leaving them in silently deflated accuracy on ~8% of
  val2017 images for both models alike (a wash in relative terms, but wrong
  in absolute terms and wrong per the intent behind "each ground-truth
  box").
- Fix: `if entry.get("iscrowd", 0) == 1: continue` at the top of the
  `annotations` loop, before the `image_id`/`bbox` extraction — skips
  building a `Box` for that entry entirely, so it's absent from both the
  per-image list and any downstream matching. `.get(..., 0)` treats a
  missing `iscrowd` field as not-crowd, matching real COCO files (the field
  is always present in the real annotations file, but synthetic/fixture
  JSON in tests doesn't have to include it).
- Verified against the real file
  (`training/data/coco/annotations/instances_val2017.json`): 36,781 total
  annotation entries, 446 flagged `iscrowd=1`, 36,335 remain after the
  filter — exactly the count the finding predicted, dropped 1:1.
- New regression test:
  `test_load_ground_truth_excludes_crowd_annotations` in
  `training/tests/datagen/simulate/test_ground_truth.py` — one image with a
  crowd-flagged "cat" box and a regular "car" box; asserts only the "car"
  `Box` survives for that image.
- Full gate after the fix: `pytest -q` → 95 passed (was 94; +1 for the new
  test), `ruff check .` → all checks passed, `mypy .` → success on all 42
  source files.

## Review fix — S3

- Finding: `persistence.py` transitively imported `ultralytics`/`torch` via
  `from datagen.simulate.inference import DetectionResult` (`inference.py`
  itself imported `ultralytics` at module level for the YOLO builders),
  making `import datagen.persistence` cost ~0.9s and tying every DB test /
  `relabel_run` / `preview_sample` invocation to the heavy CV stack.
- Fix: split `training/datagen/simulate/inference.py` by concern. Created
  `training/datagen/simulate/yolo_inference.py` holding
  `build_local_inference_fn`, `build_offload_inference_fn`, and `_to_boxes`
  (moved verbatim — no dedup of the two near-identical builders, that's a
  separate finding N1). `inference.py` now keeps only `DetectionResult`,
  `RunInferenceFn`, `score_accuracy`, `_iou`, and `apply_condition_overhead`
  — no `ultralytics`/`torch` import anywhere in it. `_to_boxes` and the
  builders import `score_accuracy`/`DetectionResult`/`RunInferenceFn` back
  from `inference`, and `ground_truth`/`Box` from
  `datagen.simulate.ground_truth`, same as before the split. The mypy
  `# type: ignore[attr-defined]` comment on `from ultralytics import YOLO`
  moved with the import into the new module.
- `training/datagen/cli/run_simulation.py` updated: imports
  `yolo_inference` alongside `ground_truth`/`inference`
  (`from datagen.simulate import ground_truth, inference, yolo_inference`),
  and `main()`'s two builder call sites now read
  `yolo_inference.build_local_inference_fn()` /
  `yolo_inference.build_offload_inference_fn()`. `RunInferenceFn`/
  `DetectionResult` type references stayed on `inference` — they didn't
  move.
- `training/tests/datagen/simulate/test_inference.py` needed no changes —
  it only ever tested `score_accuracy`/`apply_condition_overhead`, never
  the builders (confirmed by reading the file; matches Task 5's original
  plan that the real builders are hand-smoke-tested, not pytest-covered).
- Verified: `import datagen.persistence` import time dropped from the
  finding's measured ~0.9s to ~0.15s; `python -X importtime -c "import
  datagen.persistence" 2>&1 | grep -i ultralytics` now returns nothing
  (empty match, confirmed via grep's exit code 1). `from datagen.simulate
  import yolo_inference` imports cleanly on its own. Hand-smoke-tested both
  builders against a real val2017 image
  (`000000000139.jpg`, 20 ground-truth boxes after crowd-filtering) with
  CUDA available: local (YOLOv8n/CPU) latency_ms=39.09, accuracy=0.4;
  offload (YOLOv8x/CUDA) latency_ms=56.79, accuracy=0.7 — both positive
  latency, both accuracy in [0,1], same shape as before the split.
- Full gate: `pytest -q` → 95 passed (unchanged — no test added/removed),
  `ruff check .` → all checks passed, `mypy .` → success on all 43 source
  files (was 42; +1 for the new module).

## Review fix — S4

- Finding: every existing `run_simulation()` call in
  `test_run_simulation.py` passed `ground_truth={}`, so
  `_compute_missing_model_inference`'s `ground_truth.get(record.image_id,
  [])` lookup (`run_simulation.py:315`) was never exercised with real data.
  A regression swapping the key to `record.file_name`, or silently dropping
  the boxes before the inference call, would have passed every test in the
  suite unnoticed.
- Fix: added `test_run_simulation_passes_correct_ground_truth_to_inference_functions`.
  Gave three distinct pool `image_id`s (0, 1, 2) three distinct
  ground-truth box counts (3, 0 via a deliberately absent dict entry, 1),
  and a new fake inference function,
  `_make_ground_truth_sensitive_inference_fn`, whose `DetectionResult.accuracy`
  is `min(1.0, len(ground_truth_boxes) / 3.0)` — an exact, invertible
  function of the box count it's called with (unlike the module's existing
  `_make_fake_inference_fn`, which returns a fixed `DetectionResult`
  regardless of input, so couldn't distinguish "got the right boxes" from
  "got nothing"). Ran `run_simulation()` with this `ground_truth` dict and
  fake, then queried the persisted `ModelInference` rows (keyed by
  `file_name`, so mapped `image_id -> file_name` via the same
  `ImageRecord`s `_write_fake_pool` returned) and asserted each of the
  three images' `accuracy` decodes back to exactly the box count that image
  was given.
- Key realization while writing this: `_compute_missing_model_inference`
  runs inference over *every* pool image on a fresh dataset, not just the
  frames `stratified_sample` later selects for `simulation_results` — so
  the test didn't need `frame_count` tricks to guarantee the three chosen
  `image_id`s were covered; querying `ModelInference` directly (rather than
  `SimulationResult`) sidesteps sampling nondeterminism entirely, per the
  finding's own suggestion that `ModelInference` is more direct.
- Full gate: `pytest -q` → 96 passed (was 95; +1 new test, all others
  green), `ruff check .` → all checks passed, `mypy .` → success on all 43
  source files (unchanged file count — only `test_run_simulation.py`
  touched).

## Review fix — S5

- Confirmed empirically (not just from reading source) that `ultralytics`'s
  `YOLO(...)` auto-downloads even when given a full, non-relative path: the
  download decision in `ultralytics.utils.downloads.attempt_download_asset`
  is made by checking the path's *basename* against a hardcoded list of
  known official asset names (`GITHUB_ASSETS_NAMES`), not by whether the
  string "looks like" a bare model name vs. a path. Both `"yolov8n.pt"` and
  `"yolov8x.pt"` are in that list. So `Path("/tmp/nowhere/yolov8n.pt")`
  missing on disk still triggers `safe_download(..., file=<that full
  path>)` — it just downloads to the (now-fixed) full path instead of
  wherever cwd happened to be. Fixing the path alone (per the finding's
  first suggested fix) is therefore *not* sufficient to stop the silent
  download; an explicit existence check before constructing `YOLO(...)` is
  required too. Proved this with `unittest.mock.patch` on
  `ultralytics.utils.downloads.safe_download` to observe the call without
  letting an actual download run — see the S5 fix commit/PR discussion for
  the repro script shape if this needs re-verifying against a future
  `ultralytics` version.
- Added `LOCAL_MODEL_WEIGHTS_PATH`/`OFFLOAD_MODEL_WEIGHTS_PATH` to
  `datagen/config.py`, resolved via `Path(__file__).resolve().parents[1]`
  (config.py lives at `training/datagen/config.py`, weights at
  `training/yolov8n.pt`/`training/yolov8x.pt`, one level up from
  `datagen/`) — same fixed-location-regardless-of-cwd pattern as this
  repo's other `Path(__file__).resolve()`-based lookups
  (`relabel_run.py`/`preview_sample.py`/`run_simulation.py`'s `.env`
  loading). Note: the finding's suggestion that `config.py` already had a
  `DATASET_DIR`/`ANNOTATIONS_PATH`/`IMAGES_DIR` precedent to mirror turned
  out not to match the current codebase — those names don't exist anywhere
  in `training/`; `images_dir`/`annotations_path` are CLI args, not config
  constants. Followed the general style of `config.py` (uppercase module
  constant + explanatory comment) instead of a literal nonexistent
  precedent.
- `yolo_inference.py`'s two builders now call a shared `_require_weights_file`
  helper (fail-fast `FileNotFoundError` naming the expected path, styled
  after `sourcing/image_source.py`'s `load_image_index`/`resolve_image_path`
  error messages) before constructing `YOLO(...)`, so a missing weights file
  fails clearly instead of downloading.
- Hand-verified the actual regression: running
  `build_local_inference_fn()` from the **repo root** (not `training/`) now
  loads correctly via the fixed path — before this fix it would have looked
  for `yolov8n.pt` relative to the repo root and either failed or
  downloaded there. Also verified the fail-fast path directly by
  monkeypatching `config.LOCAL_MODEL_WEIGHTS_PATH` to a nonexistent path
  and confirming `FileNotFoundError` is raised with no download attempted.
- Full gate after this fix: `pytest -q` → 96 passed (no new tests were
  needed — this finding was a robustness fix to model-loading plumbing that
  the existing suite doesn't exercise against real weight files; verified
  by hand instead, per the task instructions), `ruff check .` → all checks
  passed, `mypy .` → success on all 43 source files.
