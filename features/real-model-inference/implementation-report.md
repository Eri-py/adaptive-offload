# Real Model Inference for the Data-Gen Simulator — Implementation Report

**Feature:** Real Model Inference for the Data-Gen Simulator
**Directory:** `features/real-model-inference/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Add the `model_inference` cache table | completed |
| Task 2 — Ground-truth parsing module | completed |
| Task 3 — Real-inference core types and IoU-based accuracy scoring | completed |
| Task 4 — Condition-driven latency overhead (replaces `stub_inference`) | completed |
| Task 5 — Real YOLO-backed inference builders | completed |
| Task 6 — Persistence functions for the model-inference cache | completed |
| Task 7 — Wire real inference into `run_simulation`'s orchestration | completed |
| Task 8 — Regression test run | completed |

## Files changed

### Added

- `database/migrations/versions/0002_create_model_inference_table.py` — Alembic migration for the new `model_inference` table; reuses the existing `label` Postgres enum type rather than recreating it.
- `training/datagen/simulate/ground_truth.py` — `Box(NamedTuple)` and `load_ground_truth(annotations_path) -> dict[int, list[Box]]`, parsing a COCO-format annotations file's `annotations`/`categories` arrays into per-image ground-truth boxes.
- `training/tests/datagen/simulate/test_ground_truth.py`
- `training/datagen/simulate/inference.py` — `DetectionResult`, `RunInferenceFn`, `score_accuracy` (IoU-based match-rate scoring), `apply_condition_overhead` (synthetic latency overhead on a real base), `build_local_inference_fn`/`build_offload_inference_fn` (real YOLOv8n-on-CPU / YOLOv8x-on-GPU builders).
- `training/tests/datagen/simulate/test_inference.py`

### Edited

- `.gitignore` — added `*.pt` (downloaded pretrained model weights land in the working directory and must not be tracked).
- `database/common/models.py` — added the `ModelInference` table (composite PK: `dataset`, `file_name`, `model_path` reusing the `Label` enum; `latency_ms`, `accuracy`, `computed_at`).
- `database/tests/common/test_models.py` — round-trip coverage for the new table.
- `training/datagen/config.py` — removed 9 dead synthetic-accuracy-modeling coefficients and base-latency constants; kept the 6 latency-overhead coefficients (now added to a real base latency instead of a constant).
- `training/datagen/persistence.py` — added `get_known_model_inference`/`store_model_inference`, mirroring the existing `scene_complexity` cache functions.
- `training/tests/datagen/test_persistence.py` — round-trip, skip-known-pairs, and empty-dataset coverage for the above.
- `training/pyproject.toml` — added `ultralytics>=8.0`.
- `training/datagen/cli/run_simulation.py` — `run_simulation()` gained required `run_local_inference`/`run_offload_inference`/`ground_truth` params; new `_compute_missing_model_inference` precompute-and-cache pass (parallel to the existing complexity-scoring pass); per-(frame, condition) loop now looks up cached real inference and applies synthetic condition overhead instead of calling `stub_inference`; `main()` wires the real YOLO builders and loads ground truth.
- `training/tests/datagen/cli/test_run_simulation.py` — all 10 existing `run_simulation(...)` call sites updated to inject fake inference functions; 2 new tests added.

### Deleted

- `training/datagen/simulate/stub_inference.py` and `training/tests/datagen/simulate/test_stub_inference.py` — replaced by `inference.py`.

## Tests

**Unit** (pure functions / no database):
- `training/tests/datagen/simulate/test_ground_truth.py` (9 tests): COCO annotations/categories parsing, bbox coordinate conversion, missing-annotations-for-an-image handling, and 5 malformed-input error cases.
- `training/tests/datagen/simulate/test_inference.py` (13 tests): `score_accuracy`'s IoU-based matching (perfect/no/partial match, category mismatch, zero-ground-truth edge case, custom threshold) and `apply_condition_overhead`'s determinism, accuracy passthrough, and latency-overhead direction.

**Integration** (real ephemeral Postgres database):
- `database/tests/common/test_models.py` (1 test, extended): round-trips all four tables including the new `ModelInference`.
- `training/tests/datagen/test_persistence.py` (+3 tests): `get_known_model_inference`/`store_model_inference` round-trip, skip-already-known-pairs, empty-dataset behavior.
- `training/tests/datagen/cli/test_run_simulation.py` (+2 tests): condition never changes a given frame's accuracy (only latency varies); a second `run_simulation()` call reuses cached model-inference results without re-invoking the (fake, in tests) inference functions.

Real YOLO model-loading/inference code (`build_local_inference_fn`/`build_offload_inference_fn`) is hand-smoke-tested against real COCO images rather than covered by pytest — loading a 68M-parameter model on every test run was judged impractical; this was verified independently by both the implementing subagent and the orchestrator against real `training/data/coco/val2017/` images, producing plausible non-uniform accuracy values (0.25–1.0) and positive latency for both paths.

All tests pass: `database/` 1/1, `training/` 93/93 (72 pre-existing at this feature's start + 21 net new), 0 skipped. `ruff check .` and `mypy .` clean in both packages.

## Commits

`feature/datagen-modularity..feature/real-model-inference` (10 commits):

- `de96399` — Add feature spec for real model inference in the data-gen simulator
- `a81528c` — Add implementation plan for real model inference
- `10e4722` — Task 1: add the model_inference cache table
- `c00d091` — Task 2: ground-truth parsing module
- `77c986f` — Task 3: real-inference core types and IoU-based accuracy scoring
- `2c87343` — Task 4: condition-driven latency overhead, replacing stub_inference
- `066cee9` — Task 5: real YOLO-backed inference builders
- `476bbf1` — Task 6: persistence functions for the model-inference cache
- `217b196` — Task 7: wire real inference into run_simulation's orchestration
- `ba6f355` — Task 8: regression test run

## Notable events

- Task 4 deliberately left the full `training/` test suite unable to collect (a known, expected consequence of deleting `stub_inference.py` while `run_simulation.py` still imported it) until Task 7 rewired that import three tasks later. Verification for Tasks 4–6 used scoped test paths instead of the full suite; Task 7 restored a fully green full-suite run, confirmed in Task 8.
- Task 5's `pip install ultralytics` took roughly 80 minutes of wall-clock time (large `torch`/CUDA transitive dependency download); confirmed as genuine progress rather than a hang via `/proc/<pid>/io` before continuing to wait.
- Task 5 surfaced that `ultralytics` downloads pretrained weight files into the current working directory rather than a cache directory; the orchestrator added `*.pt` to `.gitignore` (outside the implementing subagent's own file scope) before those files could be accidentally committed.
