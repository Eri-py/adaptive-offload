# Implementation Report — Real Inference Paths

**Feature:** Real Inference Paths: Phone Model Export and the Server Classify
Endpoint
**Directory:** `features/final-demo/01-inference-paths/`

## Tasks

| # | Task | Outcome |
|---|---|---|
| 1 | Export module and its unit tests | Completed |
| 2 | Produce and commit the real export artifacts | Completed |
| 3 | Server scaffolding, config, startup and health | Completed |
| 4 | Preprocessing and classification service | Completed |
| 5 | Classify endpoint | Completed |
| 6 | Request-logging table, migration and writer | Completed |
| 7 | End-to-end classify integration test | Completed |
| 8 | Regression test run | Completed |

## Files changed

### Added

- `training/birds/export.py` — the `python -m birds.export` entry point.
- `training/tests/birds/test_export.py`
- `app/assets/models/small.tflite` — exported phone model, 17,945,960 bytes,
  float32.
- `app/assets/models/cub200-species.json` — 200 species names by class index.
- `app/assets/birds/*.jpg` (15) and `app/assets/birds/expected.json` — spike
  fixtures: 224x224 crops with PyTorch's predictions on the same bytes.
- `app/hooks/useBirdSpike.ts`, `app/screens/BirdSpikeScreen.tsx`
- `server/api/core/config.py`, `server/api/core/app.py`
- `server/api/services/classification/{model,preprocess,classify,request_log}.py`
- `server/api/contracts/classification.py`
- `server/api/controller/{health,classification}.py`
- `server/api/dependencies/providers.py`
- `database/migrations/versions/0005_create_classify_requests_table.py`
- Package markers under `server/api/` and `server/tests/`.

### Modified

- `server/pyproject.toml` — real runtime and dev dependencies, pytest and mypy
  config.
- `database/common/models.py` — the `ClassifyRequest` model; docstring updated
  from six tables to seven.
- `database/tests/common/test_models.py` — round-trip extended to the new table.
- `app/App.tsx`, `app/README.md`, `app/package.json`, `app/package-lock.json`.

### Deleted

- `app/screens/BenchmarkScreen.tsx`, `app/hooks/useBenchmark.ts`
- `app/assets/models/yolov8n.tflite`, `app/assets/images/*.jpg` (15)
- `app/assets/models/small-nhwc.tflite` (added then removed within the branch)

Note: `git diff --name-status` reports
`app/assets/models/yolov8n.tflite -> app/assets/models/small.tflite` as a rename
at 51% similarity. It is not one — both are TFLite flatbuffers of similar size,
and rename detection matched them. They are an unrelated deletion and addition.

## Tests

### Unit

| File | Tests | Covers |
|---|---|---|
| `training/tests/birds/test_export.py` | 10 | Label ordering and cleaning, the 200-entry count, missing `classes.txt`, parity pass/fail on disagreement and on drift, and an unusable staged file publishing nothing. |
| `server/tests/core/test_app.py` | 2 | The startup warmup runs exactly one forward pass at (1, 3, 224, 224); the stubbed app does not warm up. |
| `server/tests/controller/test_health.py` | 3 | Health reports the model loaded, reports it unloaded on a bare app, and startup fails clearly without CUDA. |
| `server/tests/services/classification/test_preprocess.py` | 8 | The 224 path is not geometrically transformed and other sizes are, grayscale conversion, normalization values, and three undecodable-body cases. |
| `server/tests/services/classification/test_classify.py` | 5 | Hand-computed probabilities, top-2 margin and top-3 ordering, and that timing wraps the forward pass. |
| `server/tests/services/classification/test_request_log.py` | 4 | One row per request, plus three swallowed-failure paths (raising session, missing table, unset `DATABASE_URL`). |
| `server/tests/controller/test_classification.py` | 5 | Full response payload, the transform flag on a non-224 image, HTTP 400 on a non-image body, HTTP 503 with no model, and determinism across repeat posts. |
| `database/tests/common/test_models.py` | 1 | Round-trips all seven tables against an ephemeral Postgres database. |

### Integration

| File | Tests | Covers |
|---|---|---|
| `server/tests/test_integration_classify.py` | 2 | Body in, response out, exactly one row logged with the posted byte count and `compute_ms < total_ms`; and a classify still returning 200 with no row when the database is unreachable. |

All tests pass: `training/` 301, `database/` 1, `server/` 29 — 331 in total, none
skipped. `ruff` is clean in all three areas. `mypy` is clean in `database/` and
`server/`, and reports only the 9 pre-existing `ndarray` type-arg errors in
`training/coco/` that predate this work.

## Commits

`main..HEAD` on `feature/final-demo` — 13 commits, `ac54275` through `75070e9`.

## Notable events

- **The parity gate was changed mid-task to verify the published artifact.** As
  first written it compared the in-memory converted model, so a truncated or
  failed write would have passed the gate and been promoted into place by
  `os.replace`. It now reloads the staged file via `litert_torch.load` and checks
  that. A test covers an unusable staged file.
- **An on-device spike was run between Tasks 2 and 3**, outside the plan, to
  settle whether the Core ML delegate accepts MobileNetV3-Large before the server
  half was built. It does: 4.20 ms mean against CPU's 7.19 ms, all 15 photos
  matching PyTorch, and tensor layout making no difference. The channel-last model
  built as a hedge was removed as redundant. This also produced the project's
  first real on-device bird-model latency, which A7 named as its most valuable
  missing measurement.
- **A startup warmup was added to Task 5 after a real end-to-end run.** The first
  classify request cost 2224 ms to CUDA's lazy kernel initialisation against ~10 ms
  for the second. Left alone it would have stalled the first scan of a live demo
  and written a server-compute figure roughly 350x too large into the research log.
  After the fix the first call is 16.86 ms.
- **Task 7's scope was corrected.** The plan listed only a test file, but Task 6
  deliberately left the writer unwired, so no row could be logged until the route
  called it. The wiring was folded into Task 7 and recorded in `progress.md`.
- **A real-database hazard was caught during Task 7.** `server/tests/conftest.py`
  loads `database/.env`, which sets `DATABASE_URL`, so once the route called the
  writer the existing controller tests would have aimed inserts at the real
  database. The shared app fixture now overrides the logger with a no-op. The real
  database was confirmed untouched afterwards: still at alembic revision `0004`
  with no `classify_requests` table.
- **Migration `0005` is authored and deliberately not applied.** Per `CLAUDE.md`
  the user runs it (`cd database && alembic upgrade head`). It was verified
  against the ORM model by capturing its `create_table` call with a stand-in for
  `alembic.op`: all ten columns match on name, type and nullability, and so does
  the primary key. That check was necessary because the ephemeral test database is
  built with `Base.metadata.create_all` rather than by running migrations, so the
  suite cannot detect a model/migration mismatch.
- **One open question is recorded in `learnings.md`:** server compute measured
  through the endpoint is ~12.5 ms, against 6.08-6.60 ms measured directly in a
  tight loop, while the experiment's cost model assumes 6.731 ms. The cause is not
  yet established. Offload cost is network-dominated, so the effect on band
  boundaries is small, but the report should not quote 6.731 ms as the live figure
  until this is understood.
