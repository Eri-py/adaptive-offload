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
