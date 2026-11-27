# Real Inference Paths: Phone Model Export and the Server Classify Endpoint — Implementation Plan

## Summary

Produces the two runnable ends the demo app will route between: a `python -m
birds.export` step that converts the trained small checkpoint to a float32
TFLite file plus a species-name list, gated on a parity check against PyTorch
over real test photos; and the server's first real endpoints — a health probe
and a classify endpoint that runs the large checkpoint on CUDA and records every
request to Postgres.

## Approach & Key Decisions

- **The server does not import the `birds` training package.** It rebuilds the
  large architecture from torchvision directly and loads the checkpoint by path.
  Importing `birds` would drag the research package's whole dependency set
  (mlflow, ultralytics, opencv) into the serving area and invert the layering
  that `.claude/coding-guidelines.md` sets up, where `server/` and `training/`
  are independent consumers of `database/` and nothing else. The cost is ~5
  duplicated lines of head replacement; a drift between them fails loudly at
  `load_state_dict`, not silently.
- **One species-name file, two consumers.** The export writes
  `app/assets/models/cub200-species.json` and the *server* reads that same file
  rather than deriving names itself. This is a correctness decision, not a
  convenience one: phone and server must agree on what class index 57 means, and
  a single tracked artifact makes disagreement impossible. The file is ordered by
  0-based model class index, derived from the dataset's `classes.txt` the same way
  `birds/dataset/splits.py` derives training labels (`int(class_id) - 1`).
- **Classify takes a raw JPEG body, not multipart.** `python-multipart` is not
  installed, and more importantly the research's offload cost is built on
  `PAYLOAD_KB = 11.39` measured as *image bytes*; a multipart envelope would make
  the live payload systematically larger than the number the report's band
  boundaries are derived from.
- **The parity gate lives in the export command, not only in tests.** The
  command samples real CUB test photos, compares TFLite against PyTorch, and
  refuses to leave an artifact behind on failure. Unit tests cover the parity and
  label logic with stub models and synthetic tensors so the suite never needs the
  gitignored dataset.
- **Export is `python -m birds.export`**, matching `python -m birds.train`
  rather than adding a `[project.scripts]` entry — the guidelines' CLI-surface
  rule is about the datagen tools, and birds entry points are already run as
  modules.
- **Logging is best-effort by design.** The session factory is built lazily and
  every insert is wrapped, mirroring the "tracking must never break a run"
  pattern in the guidelines' Experiment-tracking section. A dead Postgres must
  not be able to break a live presentation.

## Out of Scope

Carried from the spec: the app, any routing/budget/cascade logic, Tailscale and
device reachability, reviving `features/server-served-benchmark/`, quantization,
measuring JPEG-q85 accuracy cost, re-measuring published accuracy figures,
running the small model server-side, batching, auth and rate limiting.

Further deferred by this plan: no `api/common/` module (nothing is yet shared by
two services), no OpenAPI customisation, no structured-logging framework
(standard library logging only), and no change to `server/api/__init__.py`'s
placeholder beyond what the package layout requires.

## Dependencies and Configuration

- `server/pyproject.toml` gains runtime dependencies: `fastapi`, `uvicorn`,
  `torch`, `torchvision`, `pillow`, `python-dotenv`; dev extras gain `httpx`
  (for Starlette's `TestClient`). All are already present in the shared root
  `.venv`, so no install is strictly required, but the manifest must stop
  claiming `dependencies = []`. `database` stays installed separately via
  `pip install -e ../database`, as that file's existing comment describes.
- New environment configuration, read in `server/api/core/config.py`, with
  repo-relative defaults: `BIRDS_LARGE_CHECKPOINT`
  (default `training/models/birds/large.pt`), `BIRDS_SPECIES_LABELS`
  (default `app/assets/models/cub200-species.json`), and the existing
  `DATABASE_URL` consumed via `common.db.get_engine`. Plain `os.environ` reads —
  `pydantic-settings` is not installed and is not worth adding.
- **One migration, authored not applied:** `0005_create_classify_requests_table`.
  The user runs it; per this repo's `CLAUDE.md` no agent runs migrations.
- The server is started by the user as `uvicorn api.core.app:app` from
  `server/`. No agent starts it.
- `server/tests/conftest.py` reads `POSTGRES_ADMIN_URL` from `database/.env`
  (where it is already defined) for the ephemeral-database fixture, under the
  scoped exception in `CLAUDE.md`.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/birds/export.py` | add | Export entry point: labels, TFLite conversion, parity gate | Birds entry points live at the package root beside `train.py` |
| `training/tests/birds/test_export.py` | add | Unit tests for label ordering/cleaning and parity logic | Mirrors source tree; stubs avoid needing the dataset |
| `app/assets/models/small.tflite` | add | The exported phone model (float32, ~18 MB) | Must be git-tracked or EAS Build omits it |
| `app/assets/models/cub200-species.json` | add | 200 species names by class index | Single source of truth for phone and server |
| `server/pyproject.toml` | edit | Declare real runtime and dev dependencies | Currently claims `dependencies = []` |
| `server/api/core/config.py` | add | Checkpoint/labels/database configuration | Guidelines put app config in `core/` |
| `server/api/core/app.py` | add | FastAPI app, lifespan model load, router registration | Model loaded once at startup, never per request |
| `server/api/services/classification/model.py` | add | Build large architecture, load checkpoint, require CUDA | Server must not import the `birds` package |
| `server/api/services/classification/preprocess.py` | add | Conditional resize/crop then ImageNet normalize | The 224-or-else rule the spec defines |
| `server/api/services/classification/classify.py` | add | Forward pass, top-3, max prob, top-2 margin, compute ms | Inference logic belongs in a service |
| `server/api/services/classification/request_log.py` | add | Best-effort per-request Postgres write | Classification-specific, so not a shared helper |
| `server/api/contracts/classification.py` | add | Pydantic response models for classify and health | Contracts are never raw dicts |
| `server/api/controller/health.py` | add | Health route | One module per resource |
| `server/api/controller/classification.py` | add | Classify route: parse, call service, shape response | Handlers hold no inference logic |
| `server/api/dependencies/providers.py` | add | Dependency providers for model, labels, logger | DI wiring lives here |
| `server/api/{core,services,services/classification,contracts,controller,dependencies}/__init__.py` | add | Package markers | Required by the mandated layout |
| `server/tests/conftest.py` | add | Stub-model app fixture and ephemeral-Postgres fixture | Tests must not need a GPU or the real database |
| `server/tests/services/classification/test_preprocess.py` | add | Conditional-transform tests | Guards the accuracy-critical path |
| `server/tests/services/classification/test_classify.py` | add | Confidence/top-3/timing tests with a stub model | Core research-relevant output |
| `server/tests/services/classification/test_request_log.py` | add | Logging + failure-tolerance tests | Logging must never fail a response |
| `server/tests/controller/test_health.py` | add | Health endpoint test | — |
| `server/tests/controller/test_classification.py` | add | Classify endpoint tests incl. bad-image rejection | — |
| `server/tests/test_integration_classify.py` | add | End-to-end classify writing one row | Boundary wiring test |
| `database/common/models.py` | edit | Add the `ClassifyRequest` ORM model | `database/` owns all shared tables |
| `database/migrations/versions/0005_create_classify_requests_table.py` | add | Migration matching the new model | Schema changes go through Alembic |
| `database/tests/common/test_models.py` | edit | Round-trip test for the new table | Matches the existing per-model test pattern |

## Tasks

### Task 1 — Export module and its unit tests

- **Objective:** Add `python -m birds.export`, producing the TFLite model and the species-name list, gated on a parity check.
- **Files:** `training/birds/export.py`, `training/tests/birds/test_export.py`
- **Details:**
  - Load the small checkpoint on CPU via `birds.modeling.build.load_checkpoint("small", torch.device("cpu"))`; `require_checkpoint` already raises the actionable "run `python -m birds.train --model small`" error, so reuse it rather than writing a new message. Do not call `require_cuda` — export must work without a GPU.
  - Convert with `litert_torch.convert(model, (torch.randn(1, 3, 224, 224),))` then `.export(path)`. `export()` takes only a path; there are no converter flags and no quantization.
  - Species names: read `training/data/cub200/CUB_200_2011/classes.txt` (200 lines, `"1 001.Black_footed_Albatross"`). Entry at index `i` is the name for class `i`, i.e. dataset class `i + 1`, matching `splits.py`'s `int(class_id) - 1`. Strip the numeric prefix and replace underscores with spaces. Write as a JSON array of 200 strings.
  - Parity gate: sample test-split photos (default 50, seeded with `birds.config.SEED`, overridable by a CLI flag), run each through `eval_transform("small")` from `birds.dataset.splits`, and compare TFLite output against the PyTorch model. Require the predicted class to agree on every sampled photo and the maximum softmax probability to differ by at most 1e-3. On failure, raise with the disagreeing count and the worst drift, and do not leave a model file behind.
  - Destination defaults to `app/assets/models/` (`small.tflite`, `cub200-species.json`), overridable by a CLI flag.
  - Tests use a tiny stub `nn.Module` and a temporary `classes.txt`; they must not require the gitignored CUB dataset or run a real conversion. Cover: label ordering against known index/name pairs, prefix and underscore cleaning, the 200-entry count, the actionable error when `classes.txt` is absent, and that the parity comparison fails when predictions disagree and passes within tolerance.
- **Success criteria:**
  - `python -m birds.export --help` runs and the module imports cleanly.
  - New tests pass; `cd training && ruff check . && mypy .` shows no new errors (the 9 pre-existing `ndarray` errors in `training/coco/` remain).

### Task 2 — Produce and commit the real export artifacts

- **Objective:** Run the export for real and commit the two artifacts.
- **Files:** `app/assets/models/small.tflite`, `app/assets/models/cub200-species.json`
- **Details:**
  - Run `python -m birds.export` from `training/` against the real checkpoint and dataset, letting the parity gate run over real photos.
  - Confirm both files are git-tracked. `training/models/` is gitignored wholesale and `*.pt` is ignored, but `app/assets/models/` is tracked (`yolov8n.tflite` already is) and no `*.tflite` ignore rule exists — verify with `git check-ignore -v` rather than assuming.
  - Record the observed parity numbers (photos compared, agreement count, worst probability drift) and the file size in the task output, for the implementation report.
- **Success criteria:**
  - Both artifacts exist, are tracked, and the species file has exactly 200 entries with index 0 reading "Black footed Albatross".
  - The parity gate passed on real photos with 100% class agreement.

### Task 3 — Server scaffolding, config, startup and health

- **Objective:** Stand up the FastAPI app with the mandated layout, load the large model once at startup, and serve the health endpoint.
- **Files:** `server/pyproject.toml`, `server/api/core/config.py`, `server/api/core/app.py`, `server/api/services/classification/model.py`, `server/api/contracts/classification.py`, `server/api/controller/health.py`, `server/api/dependencies/providers.py`, the `__init__.py` package markers, `server/tests/conftest.py`, `server/tests/controller/test_health.py`
- **Details:**
  - Replace `dependencies = []` with the list in **Dependencies and Configuration**; add `httpx` to dev extras.
  - `config.py`: read `BIRDS_LARGE_CHECKPOINT` and `BIRDS_SPECIES_LABELS` from the environment with the repo-relative defaults named above; load `server/.env` if present via `python-dotenv`.
  - `model.py`: build `torchvision.models.convnext_base(weights=None)`, replace `classifier[2]` with a 200-way `nn.Linear` (mirroring `birds/modeling/build.py` but without importing it), `load_state_dict` from the configured path, `.eval()`, move to CUDA. Raise a clear startup error if `torch.cuda.is_available()` is false — no CPU fallback, because the reported server-compute figure is a GPU number. Also load and validate the species list here (exactly 200 entries).
  - `app.py`: a lifespan handler that loads model and labels into application state once, and registers the health and (in Task 5) classify routers.
  - Health contract reports status and whether the model is loaded; the handler reads application state only, so it stays cheap enough to serve as an RTT probe.
  - `conftest.py`: an app fixture that overrides the model/labels providers with a stub module returning deterministic logits, so no test needs a GPU or the 335 MB checkpoint. Also add the `postgres_engine` fixture mirroring `database/tests/conftest.py`, loading `database/.env` for `POSTGRES_ADMIN_URL`, for use in Task 6.
- **Success criteria:**
  - `cd server && ruff check . && mypy . && pytest` all pass, with mypy strict clean.
  - The health test passes against the stubbed app and reports the model as loaded.
  - A test asserts startup fails with a clear error when CUDA is reported unavailable (monkeypatched).

### Task 4 — Preprocessing and classification service

- **Objective:** Implement the conditional transform and the inference service with its confidence measures and compute timing.
- **Files:** `server/api/services/classification/preprocess.py`, `server/api/services/classification/classify.py`, `server/tests/services/classification/test_preprocess.py`, `server/tests/services/classification/test_classify.py`
- **Details:**
  - `preprocess.py`: decode JPEG bytes with Pillow, convert to RGB. If the decoded image is exactly 224x224, apply only `ToTensor` + ImageNet normalize (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`). Otherwise apply `Resize(232)` + `CenterCrop(224)` first. Return the tensor and a flag recording whether the geometric transform was applied. A body that Pillow cannot decode raises a dedicated error type the controller maps to a client error.
  - `classify.py`: run the forward pass under `torch.inference_mode()`, softmax the logits, and return top-1 index and name, maximum probability, top-2 margin (top-1 minus top-2 probability), the top-3 (index, name, probability), and the compute time in milliseconds. Time **only** the forward pass — not decode, transform or serialization — since the experiment's cost model treats server compute as a term separate from transfer. Use `torch.cuda.synchronize()` around the timed region when on CUDA, or the measurement is meaningless because kernels launch asynchronously.
  - Tests use a stub model with hand-chosen logits so expected probabilities, margin and ordering are exact. Cover: a 224x224 input is not geometrically transformed and the flag says so; a non-224 input is, and the flag says so; margin and top-3 ordering are correct; an undecodable body raises the dedicated error.
- **Success criteria:**
  - New tests pass; `ruff`, `mypy` clean in `server/`.
  - A test pins the top-2 margin and top-3 ordering against hand-computed values.

### Task 5 — Classify endpoint

- **Objective:** Expose `POST /classify` over the service, accepting a raw JPEG body.
- **Files:** `server/api/contracts/classification.py`, `server/api/controller/classification.py`, `server/api/core/app.py`, `server/api/dependencies/providers.py`, `server/tests/controller/test_classification.py`
- **Details:**
  - The endpoint reads the raw request body as bytes (`Content-Type: image/jpeg`), not a multipart form — see the Approach note on payload size.
  - Response contract fields: top-1 species name and class index, maximum probability, top-2 margin, three ranked species with probabilities, model compute milliseconds, and whether the geometric transform was applied.
  - The handler only reads the body, calls the service, and shapes the response; it contains no inference logic. An undecodable body maps to HTTP 400 with a descriptive message.
  - Register the router in `app.py`.
  - Tests drive the stubbed app with `TestClient`: a real 224x224 JPEG built in-test with Pillow returns the full payload; a non-224 JPEG reports that it was transformed; a non-image body returns 400; the same image posted twice returns identical class, probability and margin.
- **Success criteria:**
  - All endpoint tests pass, including the 400 case and the determinism case.
  - `ruff`, `mypy` clean in `server/`.

### Task 6 — Request-logging table, migration and writer

- **Objective:** Add the `classify_requests` table, its migration, and the best-effort writer.
- **Files:** `database/common/models.py`, `database/migrations/versions/0005_create_classify_requests_table.py`, `database/tests/common/test_models.py`, `server/api/services/classification/request_log.py`, `server/tests/services/classification/test_request_log.py`
- **Details:**
  - ORM model `ClassifyRequest` following the style of `BirdPrediction` in the same file: surrogate UUID string primary key, `requested_at` (timezone-aware, Python-side `_utcnow` default), `payload_bytes` (Integer), `was_transformed` (Boolean), `predicted_label` (Integer), `predicted_name` (String), `max_probability` (Float), `top2_margin` (Float), `compute_ms` (Float), `total_ms` (Float). Update the module docstring's table count and list.
  - Migration `0005` with `down_revision = "0004"`, authored to match the model exactly, following `0004`'s structure and its note about who runs migrations. No enum columns. **Do not run it.**
  - `request_log.py`: build the engine lazily via `common.db.get_engine` and insert one row per request. Any failure — unreachable database, missing table, bad URL — is caught, logged as a warning, and swallowed so the response still succeeds.
  - Add a round-trip test for the new model in `database/tests/common/test_models.py` using the existing `postgres_engine` fixture, matching how the other models are tested.
  - Server-side tests: one row written per classify with the expected field values (ephemeral database), and a failure-injection test proving a raising session does not propagate.
- **Success criteria:**
  - `cd database && ruff check . && mypy . && pytest` pass, including the new round-trip test.
  - The logging failure test proves the exception is swallowed and a warning is emitted.
  - The migration file exists and is **not** applied; the task output states that the user must run it.

### Task 7 — End-to-end classify integration test

- **Objective:** Prove the wired path — body in, response out, exactly one row logged.
- **Files:** `server/tests/test_integration_classify.py`
- **Details:** Drive the stubbed app with `TestClient` against an ephemeral Postgres database, posting a real in-test JPEG. Assert the response is complete, that exactly one row exists with the payload byte count matching the posted body length, and that the row's `compute_ms` is strictly less than its `total_ms` — the spec's criterion that compute time excludes decode and transform work.
- **Success criteria:**
  - The integration test passes.
  - A second assertion confirms a classify request still succeeds when the database is unreachable, with no row written.

### Task 8 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest`, `cd database && pytest`, and `cd server && pytest` all pass with no skips introduced by this work.
  - `ruff check .` and `mypy .` clean in all three areas, except the 9 pre-existing `ndarray` type-arg errors in `training/coco/`.
  - Baseline for comparison: 291 tests passing in `training/` before this plan.

## Open Questions

- Whether the Core ML delegate covers MobileNetV3-Large's operator set well enough to run the whole graph on the Neural Engine is still unverified, and resolves in spec 02 when the model runs on the phone. It does not affect this plan, which produces a standard float32 model file.
