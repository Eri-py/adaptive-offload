# Data-Gen Simulator — Implementation Plan

## Summary

Build the standalone `training/datagen` simulator that produces the router's
training data: it downloads and scores the full COCO val2017 image pool for
scene complexity, draws a stratified 500-frame sample from it, crosses every
frame with 50 continuous condition vectors sampled per a named preset, runs
fully-stubbed local/offload inference on each pair, computes a utility-based
win/loss label, and persists everything to three new Postgres tables owned
by `server/common/`. This is also the first real code in the repo, so it
establishes `server/common/`'s DB layer and `training/`'s package from
scratch.

## Approach & Key Decisions

- `server/common/` gets its first real content: a SQLAlchemy engine/session
  factory (reads `DATABASE_URL`) and the three shared ORM models. Per
  `.claude/coding-guidelines.md`'s Database section, `training/` imports
  this rather than opening its own connection or redefining the tables —
  it depends on `server/common` via a local editable install
  (`pip install -e ../server`), since guidelines require `server/` and
  `training/` to keep separate manifests.
- Three tables: `simulation_runs` (one row per invocation, holding the
  resolved config snapshot), `simulation_results` (one row per
  frame × condition pair, FK'd to its run), and `scene_complexity` (one row
  per frame, keyed by dataset + file name, holding the computed edge-density
  value). Keeping scene complexity in its own table — rather than a column
  on `simulation_results` — avoids storing the same value 50× per frame per
  run, and means it's computed once ever, not once per run. This corrects
  the plan's original design, which cached computed values in a local
  file-based cache; `.claude/coding-guidelines.md` is explicit that one
  Postgres instance backs everything and rules out a separate local store
  for anything but the raw downloaded image files themselves (which stay on
  local disk purely to avoid re-fetching from COCO's CDN).
- Full-pool scoring, not partial: the simulator downloads all 5,000 val2017
  images once (caching the files locally) and Canny-scores every one,
  persisting each score to `scene_complexity` as it's computed. Stratified
  sampling then draws `frame_count` frames from this complete, known
  distribution using equal-frequency (quantile) bucketing. This was an
  explicit correction to the original spec draft, which assumed a cheaper
  annotation-only proxy; the user chose the full-download, self-consistent
  approach instead.
- `simulation_results` does not carry a `scene_complexity` column. A
  training-time query (feature 02, out of scope here) joins
  `simulation_results` to `scene_complexity` on frame id when it needs the
  complexity feature — normal SQL, no special-casing.
- Condition vectors are drawn with Latin Hypercube Sampling
  (`scipy.stats.qmc.LatinHypercube`), seeded independently from the
  frame-sampling RNG stream so tuning one doesn't perturb the other.
- The Alembic migration is authored but never applied — verified with
  `alembic upgrade head --sql` (prints DDL, never touches a database),
  honoring `CLAUDE.md`'s rule that only the user runs migrations against the
  real instance.
- Stub-model coefficients (base latencies/accuracies, noise scales, the
  packet-loss and device-load penalty weights) live in
  `training/datagen/config.py` alongside the other tunables, not hardcoded
  in the stub logic — per the user's explicit "this should sit in a config
  so we can tweak the value at will" instruction from spec review.
- No task in this plan requires a live Postgres connection. All DB-writing
  code is tested against an in-memory SQLite engine using the same
  SQLAlchemy models (SQLite is close enough to Postgres for schema/ORM-level
  assertions), so implementation and tests never need infrastructure
  `CLAUDE.md` forbids starting. A real `DATABASE_URL` and running Postgres
  instance are only needed later, when the user applies the migration and
  runs `run_simulation.py` for real — both outside this plan.
- Real COCO downloads (annotations JSON + 5,000 images, ~1GB) require
  outbound internet access and only happen when the simulator is actually
  run. The download/cache function is written to hit the real COCO CDN, but
  all automated tests substitute a mocked HTTP layer / small fixture set so
  the test suite stays fast and doesn't depend on network access.

## Out of Scope

Everything the spec excludes (real inference, real network/device-load
injection, adaptive/auto-tuned condition sampling, router training and its
train/test split, the mobile app and FastAPI endpoint, applying the
migration or starting Postgres, live validation, Open Images V7). This plan
additionally does not touch `server/api/` (controllers, contracts,
services) — only `server/common/` — and does not package `server/common` as
a versioned/distributable library; a local editable install is sufficient
for a single-repo prototype.

## Dependencies and Configuration

- New `server/pyproject.toml`: `sqlalchemy`, `psycopg[binary]`, `alembic`,
  `ruff`, `mypy` (dev).
- New `training/pyproject.toml`: `numpy`, `scipy`, `opencv-python-headless`,
  `requests`, `ruff`, `mypy` (dev), plus an editable local dependency on
  `server/common` (`pip install -e ../server`).
- New environment variable `DATABASE_URL` (Postgres connection string),
  read by `server/common/db.py`. The user must set this before applying the
  migration or running the simulator for real; not required for this plan's
  tests (see above).
- New Alembic setup under `server/`: `alembic.ini`, `migrations/env.py`,
  `migrations/script.py.mako`, one versioned migration.
- Root `.gitignore` already added (this session, ahead of implementation):
  Python artifacts (`__pycache__/`, `*.egg-info/`, `.venv/`), `training/data/`
  (the local COCO cache), and `.env`.
- The COCO val2017 dataset has already been downloaded (this session, ahead
  of implementation, since it's slow on the user's connection) to
  `training/data/coco/`: annotations under `training/data/coco/annotations/`,
  images under `training/data/coco/val2017/`. `coco.py` (Task 4) must read
  from/populate this same path, not a different cache location.
- Running the simulator for real requires outbound internet access to
  `images.cocodataset.org`.
- No new ports, no `docker-compose.yml` — the Postgres instance already
  exists outside the repo per `CLAUDE.md`.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `.gitignore` | already added | exclude Python artifacts, `training/data/`, `.env` | done ahead of implementation, alongside the dataset download |
| `server/pyproject.toml` | add | server package deps + ruff/mypy config | `server/`'s own manifest per guidelines |
| `server/common/__init__.py` | add | package marker | — |
| `server/common/db.py` | add | SQLAlchemy engine/session factory reading `DATABASE_URL` | shared DB access point per guidelines |
| `server/common/models.py` | add | `SimulationRun`, `SimulationResult`, `SceneComplexity` ORM models | shared table models per guidelines |
| `server/tests/__init__.py` | add | package marker | — |
| `server/tests/common/__init__.py` | add | package marker | — |
| `server/tests/common/test_models.py` | add | schema/round-trip test on in-memory SQLite | verifies all three ORM models without live Postgres |
| `server/alembic.ini` | add | Alembic config | migration tooling entrypoint |
| `server/migrations/env.py` | add | Alembic env wired to `models` metadata | enables offline SQL generation |
| `server/migrations/script.py.mako` | add | Alembic's default revision template | required by Alembic |
| `server/migrations/versions/0001_create_simulation_tables.py` | add | migration creating all three tables | spec acceptance criterion |
| `training/pyproject.toml` | add | training package deps + ruff/mypy config + editable dep on `server` | `training/`'s own manifest per guidelines |
| `training/datagen/__init__.py` | add | package marker | — |
| `training/datagen/config.py` | add | all tunables: presets, frame count, condition-vector count, seed, λ, stub coefficients, bucket count | central config per spec + user instruction |
| `training/datagen/coco.py` | add | COCO val2017 annotation fetch + local image-file download/cache | dataset acquisition |
| `training/datagen/complexity.py` | add | edge-density (Canny) scene-complexity proxy | scene-complexity proxy per spec |
| `training/datagen/persistence.py` | add | reads/writes all three tables via `server.common` | Postgres persistence per spec, including the scene-complexity cache table |
| `training/datagen/sampling.py` | add | stratified frame sampling via quantile bucketing | frame selection per spec |
| `training/datagen/conditions.py` | add | LHS-based condition-vector sampling per preset | continuous condition model per spec |
| `training/datagen/stub_inference.py` | add | condition-driven synthetic latency/accuracy model | stubbed inference per spec |
| `training/datagen/labeling.py` | add | utility computation + win/loss label | labeling per spec |
| `training/datagen/run_simulation.py` | add | standalone CLI script orchestrating the full pipeline | "runs as a standalone script" requirement |
| `training/tests/__init__.py` | add | package marker | — |
| `training/tests/datagen/__init__.py` | add | package marker | — |
| `training/tests/datagen/test_coco.py` | add | cache logic tests with mocked HTTP | avoids real 5,000-image download in tests |
| `training/tests/datagen/test_complexity.py` | add | edge-density proxy tests on synthetic images | correctness |
| `training/tests/datagen/test_persistence.py` | add | round-trip test on in-memory SQLite for all three tables | data-gen-logging testing bar |
| `training/tests/datagen/test_sampling.py` | add | stratified-sampling determinism + coverage tests | reproducibility + coverage AC |
| `training/tests/datagen/test_conditions.py` | add | LHS determinism + per-axis coverage tests | reproducibility + coverage AC |
| `training/tests/datagen/test_stub_inference.py` | add | stub-formula monotonicity + determinism tests | correctness |
| `training/tests/datagen/test_labeling.py` | add | label matches utility formula | routing-decision-logic testing bar |
| `training/tests/datagen/test_run_simulation.py` | add | end-to-end integration test with a small fake image pool + in-memory SQLite | verifies the wired-together pipeline and reproducibility AC |

## Tasks

### Task 1 — `server/common` DB scaffolding

- **Objective:** Stand up `server/common`'s SQLAlchemy engine and the three shared ORM models.
- **Files:** `server/pyproject.toml`, `server/common/__init__.py`, `server/common/db.py`, `server/common/models.py`, `server/tests/__init__.py`, `server/tests/common/__init__.py`, `server/tests/common/test_models.py`
- **Details:** `db.py` exposes a function that builds a SQLAlchemy engine from the `DATABASE_URL` env var (raise a clear error if unset) and a session factory. `models.py` defines:
  - `SimulationRun`: `run_id` str/UUID PK, `preset_name` str, `frame_count` int, `condition_vector_count` int, `condition_ranges` JSON (the per-axis min/max used), `seed` int, `lambda_value` float, `created_at` timestamp. Table: `simulation_runs`.
  - `SimulationResult`: `id` PK, `run_id` FK → `simulation_runs.run_id`, `frame_id` str, `network_bandwidth_mbps` float, `network_latency_ms` float, `network_packet_loss_pct` float, `device_load_pct` float, `local_latency_ms` float, `local_accuracy` float, `offload_latency_ms` float, `offload_accuracy` float, `label` str/enum of `local`/`offload`, `created_at` timestamp. No complexity column. Table: `simulation_results`.
  - `SceneComplexity`: composite PK (`dataset` str, `file_name` str), `scene_complexity` float, `computed_at` timestamp. Table: `scene_complexity`. `dataset` holds a value like `"coco_val2017"` so other datasets can be added later without a schema change.
- **Success criteria:**
  - `cd server && ruff check . && mypy .` clean
  - `test_models.py` builds an in-memory SQLite engine from these same models, inserts one row per table (a `SimulationResult` linked to a `SimulationRun`, plus an independent `SceneComplexity` row), and reads all three back with fields intact

### Task 2 — Alembic migration

- **Objective:** Author (not apply) the migration creating all three tables.
- **Files:** `server/alembic.ini`, `server/migrations/env.py`, `server/migrations/script.py.mako`, `server/migrations/versions/0001_create_simulation_tables.py`
- **Details:** Wire `migrations/env.py`'s target metadata to `server.common.models`'s declarative base so the migration matches Task 1's models exactly. The migration must never be applied here.
- **Success criteria:**
  - `cd server && alembic upgrade head --sql` runs without connecting to a database and prints `CREATE TABLE` statements for `simulation_runs`, `simulation_results`, and `scene_complexity` with the columns from Task 1

### Task 3 — `training/datagen` package + config module

- **Objective:** Stand up the `training` package and the single tunable config module.
- **Files:** `training/pyproject.toml`, `training/datagen/__init__.py`, `training/datagen/config.py`, `training/tests/__init__.py`, `training/tests/datagen/__init__.py`
- **Details:** `config.py` defines: `FRAME_COUNT = 500`, `CONDITION_VECTOR_COUNT = 50`, `STRATIFICATION_BUCKET_COUNT = 5`, `SEED`, `DEFAULT_LAMBDA`, `DATASET_NAME = "coco_val2017"`, stub-model coefficients (base local/offload latency and accuracy, noise magnitudes, device-load and packet-loss penalty weights — pick reasonable illustrative defaults), and a `PRESETS` dict keyed by name with entries `baseline`, `network-stress`, `device-stress`, `degraded-network-idle-device`, each specifying `(min, max)` ranges for `bandwidth_mbps`, `network_latency_ms`, `packet_loss_pct`, `device_load_pct` (e.g. `network-stress` narrows bandwidth to `(0.2, 5)` and raises latency to `(150, 400)`; `device-stress` narrows `device_load_pct` to `(60, 100)`; `degraded-network-idle-device` combines a poor-network range with a low `device_load_pct` range like `(0, 20)`). `training/pyproject.toml` declares the dependencies listed above, including the editable `server/common` dependency, plus ruff/mypy config.
- **Success criteria:**
  - `cd training && ruff check . && mypy .` clean
  - A test importing `config` asserts every preset defines all four ranges and that `FRAME_COUNT`/`CONDITION_VECTOR_COUNT` are positive ints

### Task 4 — COCO acquisition and local image caching

- **Objective:** Fetch COCO val2017 annotations and download/cache all 5,000 image files locally.
- **Files:** `training/datagen/coco.py`, `training/tests/datagen/test_coco.py`
- **Details:** The dataset is already present at `training/data/coco/` (downloaded ahead of implementation) — `instances_val2017.json` under `training/data/coco/annotations/`, images under `training/data/coco/val2017/`. One function reads that annotations file to get the list of 5,000 `(image_id, file_name)` pairs. Another resolves (and, if genuinely missing, downloads to `training/data/coco/val2017/`) each image, returning its local file path — treat the pre-downloaded files as the common case, not the exception. This local cache is purely for image bytes — it does not store computed values (those go to Postgres, see Task 6). The HTTP fetch path must go through an injectable/mockable layer (e.g. a `requests.Session` or fetch function passed as a parameter) so tests never hit the network.
- **Success criteria:**
  - `test_coco.py` mocks the HTTP layer with a handful of fake images/annotations and verifies: first call downloads and caches, second call reuses the cached file without re-fetching
  - `cd training && ruff check . && mypy .` clean

### Task 5 — Scene-complexity proxy

- **Objective:** Implement the edge-density scene-complexity proxy.
- **Files:** `training/datagen/complexity.py`, `training/tests/datagen/test_complexity.py`
- **Details:** A pure function taking an image (path or array) that converts to grayscale, runs Canny edge detection, and returns the fraction of edge pixels as a float in `[0, 1]`. No I/O, no database access.
- **Success criteria:**
  - Tests on synthetic images (a blank/uniform image and a high-frequency checkerboard/noise image) confirm the blank image scores near 0 and the noisy image scores meaningfully higher
  - `cd training && ruff check . && mypy .` clean

### Task 6 — Persistence layer (all three tables)

- **Objective:** Implement read/write access to `simulation_runs`, `simulation_results`, and `scene_complexity` via `server.common`.
- **Files:** `training/datagen/persistence.py`, `training/tests/datagen/test_persistence.py`
- **Details:** Functions, each taking a SQLAlchemy session/engine as a parameter (not a hardcoded connection) so tests can substitute an in-memory SQLite engine built from `server.common.models`:
  - `get_known_complexity(dataset) -> {file_name: score}` — reads all cached `scene_complexity` rows for a dataset.
  - `store_complexity_scores(dataset, {file_name: score})` — inserts new `scene_complexity` rows (skips file names already present).
  - `create_run(config_snapshot) -> run_id` — inserts one `SimulationRun` row.
  - `store_results(run_id, rows)` — bulk-inserts `SimulationResult` rows tagged with that `run_id`.
- **Success criteria:**
  - `test_persistence.py` runs against an in-memory SQLite engine: stores and re-reads `scene_complexity` rows (confirming a second `store_complexity_scores` call doesn't duplicate existing file names), creates a run, stores results, and reads everything back confirming fields and FK linkage
  - `cd training && ruff check . && mypy .` clean

### Task 7 — Stratified frame sampling

- **Objective:** Given a `{file_name: complexity_score}` mapping, select `frame_count` frames via quantile bucketing.
- **Files:** `training/datagen/sampling.py`, `training/tests/datagen/test_sampling.py`
- **Details:** A pure function taking a `{file_name: complexity_score}` mapping (as returned by Task 6's `get_known_complexity`), a target count, a bucket count, and a seed; splits the pool into `bucket_count` equal-frequency (quantile) buckets by score and samples an even share from each (seeded) to reach the target count. No I/O in this module.
- **Success criteria:**
  - Same seed + same input mapping → identical sample (determinism)
  - Sampled frames' complexity scores span low/mid/high buckets rather than clustering (coverage)
  - `cd training && ruff check . && mypy .` clean

### Task 8 — Condition-vector sampling

- **Objective:** Implement Latin Hypercube sampling of condition vectors per preset.
- **Files:** `training/datagen/conditions.py`, `training/tests/datagen/test_conditions.py`
- **Details:** A function taking a preset (from `config.PRESETS`), a count, and a seed; uses `scipy.stats.qmc.LatinHypercube` to draw that many 4-dimensional samples and scales each dimension into the preset's `(min, max)` range, returning a list of `(bandwidth_mbps, network_latency_ms, packet_loss_pct, device_load_pct)` tuples.
- **Success criteria:**
  - Same seed + preset → identical vectors (determinism)
  - Each axis's sampled values span close to the preset's full configured range rather than clustering
  - `cd training && ruff check . && mypy .` clean

### Task 9 — Stub inference model

- **Objective:** Implement the condition-driven synthetic latency/accuracy model for both paths.
- **Files:** `training/datagen/stub_inference.py`, `training/tests/datagen/test_stub_inference.py`
- **Details:** A function taking a condition vector, a frame's `scene_complexity`, and a seed, returning `(local_latency_ms, local_accuracy, offload_latency_ms, offload_accuracy)`. Local latency scales with `device_load_pct` (plus noise) off `config`'s base local latency; offload latency scales with `bandwidth_mbps`/`network_latency_ms` (transfer time + round trip) plus a packet-loss penalty (plus noise) off `config`'s base offload latency; each path's accuracy is that path's base rate reduced by `scene_complexity` and by that path's own condition severity (plus noise). All coefficients come from `config.py` (Task 3).
- **Success criteria:**
  - Determinism: same inputs + seed → identical outputs
  - Monotonicity: increasing `device_load_pct` strictly increases `local_latency_ms`; decreasing `bandwidth_mbps` strictly increases `offload_latency_ms`; increasing `packet_loss_pct` decreases `offload_accuracy`
  - `cd training && ruff check . && mypy .` clean

### Task 10 — Win/loss labeling

- **Objective:** Implement the utility-based win/loss label.
- **Files:** `training/datagen/labeling.py`, `training/tests/datagen/test_labeling.py`
- **Details:** A function taking the four stub-inference outputs and a λ, computing `utility = accuracy - λ * (latency_ms / 1000)` for each path and returning whichever path scored higher (`"local"` or `"offload"`).
- **Success criteria:**
  - Constructed cases (one where local should clearly win, one where offload should) match hand-computed utility values
  - `cd training && ruff check . && mypy .` clean

### Task 11 — Orchestration script + end-to-end integration test

- **Objective:** Wire Tasks 3–10 together into the standalone simulator script.
- **Files:** `training/datagen/run_simulation.py`, `training/tests/datagen/test_run_simulation.py`
- **Details:** A CLI script (`python -m training.datagen.run_simulation --preset <name>`) that: builds the real engine from `DATABASE_URL`; reads known scene-complexity scores (Task 6), downloads/scores (Tasks 4–5) any pool images not yet covered, and persists the new scores (Task 6); stratified-samples `frame_count` frames from the full known distribution (Task 7); samples `condition_vector_count` condition vectors for the chosen preset (Task 8); crosses every frame with every condition vector, computing stub inference (Task 9) and the label (Task 10) for each pair; creates the run record and persists all result rows (Task 6).
- **Success criteria:**
  - `test_run_simulation.py` runs the full pipeline against a small fake image pool (e.g. 20 synthetic images, via a mocked `coco.py` layer) and an in-memory SQLite engine, asserting exactly `frame_count × condition_vector_count` result rows are created with every field populated and correctly linked to the run, and that `scene_complexity` holds exactly one row per pool image
  - Running the pipeline twice with an unchanged seed/preset/pool produces identical sampled frames, condition vectors, and result values (ignoring `run_id`), and the second run does not recompute any `scene_complexity` row already present — the spec's reproducibility and reuse acceptance criteria
  - `cd training && ruff check . && mypy .` clean

### Task 12 — Regression test run

- **Objective:** Run every test added across `server/` and `training/` and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd server && ruff check . && mypy . && pytest` passes
  - `cd training && ruff check . && mypy . && pytest` passes
