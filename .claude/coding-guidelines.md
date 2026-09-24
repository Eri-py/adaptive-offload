# Coding Guidelines — adaptive-offload

## Scope & bar

This is a single-semester research prototype (Senior Seminar), not a production
system. Optimize for iteration speed and correctness of the research pipeline
(routing decisions, data-gen labels, benchmark numbers) — not for scale,
security hardening, or exhaustive production robustness.

## Repo layout

- `app/` — React Native (iOS-first) client: capture trigger, scene-complexity
  proxy, cached network/load reads, calls to the local model and/or server.
- `database/` — the shared DB layer: SQLAlchemy engine/session setup, the
  shared table models, the ephemeral-test-DB fixture helper, and the Alembic
  migrations. Owned by neither `server/` nor `training/` — both depend on it
  symmetrically via editable install (`pip install -e ../database`), since
  both read from and write to the same Postgres instance. (This used to live
  under `server/common/`, which made `server/` look like the owner even
  though `training/` was an equal consumer — moved out to fix that.)
- `server/` — everything offload-endpoint-related, split like a layered
  dotnet solution:
  - `server/api/` — the FastAPI app itself:
    - `controller/` — route handlers (~ Controllers), one module per resource.
    - `contracts/` — Pydantic request/response models.
    - `services/` — business/inference logic, grouped by domain
      (e.g. `services/detection/`), never inline in a router.
    - `core/` — app config/startup wiring (~ Configuration).
    - `dependencies/` — FastAPI dependency providers / DI wiring
      (~ Extensions/ServiceRegistrations).
  - `server/tests/` — mirrors `server/api/`'s structure, grouped by the same
    domain folders (e.g. `tests/services/detection/`).
- `training/` — all research/training code in one place, with subfolders by
  concern:
  - `training/datagen/` — the data-gen simulator: frame set, condition
    sweep, per-(frame, config) logging, dataset export.
  - `training/router/` — decision-layer code: feature extraction,
    direct-classifier and utility-regression models, training/eval scripts,
    saved model artifacts.

Each top-level area (`app/`, `database/`, `server/`, `training/`) owns its
own dependency manifest and lint/type config — don't share one config
across them pretending they're the same kind of code. `database/`,
`server/`, and `training/` are all Python but serve different purposes
(shared DB layer vs. serving vs. research), so keep their `pyproject.toml`
files (and each one's own `[tool.ruff]`/`[tool.mypy]` config) separate.
`database/`, `server/`, and `training/` do, however, share a single Python
virtualenv at the repo root (`.venv/`) rather than one venv each — their
actual dependency sets don't clash (Postgres tooling vs. web tooling vs.
numerical/CV tooling, overlapping only on `ruff`/`mypy`/`pytest`), and
managing separate venvs for a single-developer prototype was pure friction
with no isolation benefit actually being used. Editable-install all three
into that one venv (`pip install -e database[dev] -e server[dev] -e training[dev]`
from the repo root); run each package's lint/type/test commands from its
own directory as before (`cd database && ruff check . && mypy . && pytest`,
same for `server/`/`training/`) — only the venv location changed, not which
config applies where.

## Database

- One existing Postgres instance backs everything — server-side
  request/routing logs *and* the data-gen simulator's per-(frame, config)
  sweep results. No SQLite, no separate CSV/Parquet store. There is no
  `docker-compose.yml` for this — the instance and database already exist
  outside the repo.
- `database/` owns the SQLAlchemy engine/session setup and the shared table
  models. Both `server/api/services/` and `training/` import from there
  rather than opening their own connections or redefining tables — neither
  owns it, both depend on it the same way.
- The data-gen simulator (`training/datagen/`) writes each (frame, config) row
  straight to its Postgres table as it's produced.
- Training code (`training/router/`) builds its working DataFrame with a SQL
  query against that table (`pd.read_sql(query, engine)`), not by reading
  files off disk.
- Schema changes go through a migration script (Alembic, authored in
  `database/migrations/`). See the repo's `CLAUDE.md` for the rule on who
  runs it.

## TypeScript / React Native (`app/`)

- Functional components + hooks only, no class components.
- One component per file, filename matches the component (`PascalCase.tsx`).
- Co-locate a component's styles in the same file (`StyleSheet.create` at the
  bottom); pull into a shared theme file only once a style is reused across
  screens.
- Type all props with an explicit `<Component>Props` interface — no `any`.
- Anything that isn't purely presentational (feature computation, networking,
  the local-vs-offload call itself) lives in a hook or a plain `.ts` service
  module, never inline in a component body.
- Wrap every async boundary (camera capture, server request, on-device
  inference call) in explicit try/catch and surface the failure to the UI —
  don't swallow it silently, since a silent failure here also corrupts the
  latency/outcome log.

## Python / FastAPI (`server/`)

- Type hints on every function signature; `mypy` must run clean.
- Request/response bodies are Pydantic models in `server/api/contracts/`,
  never raw dicts.
- Handlers in `server/api/controller/` only parse the request, call a
  service, and shape the response — no inference or business logic inline.
- Inference and model-loading logic lives in `server/api/services/`, grouped
  by domain (e.g. `services/detection/`).
- Load the model once at startup (FastAPI lifespan/startup event) in
  `server/api/core/`, never per-request.
- Cross-cutting helpers used by more than one service or by tests go in a
  `server/api/common/` module (create it if it doesn't exist yet), not
  duplicated per-service. Keep this separate from `database/`, which is
  only for the shared DB layer, not general server-internal utilities —
  and note the name: `database`'s importable package is top-level `common`
  (`import common`), so a server-internal helpers module must never be a
  second top-level `common/` under `server/`, or it will silently shadow
  the DB layer on `sys.path`.

## Data-gen / router training code (`training/datagen/`, `training/router/`)

- Every simulation run logs enough to reproduce it: frame id, condition
  params, path(s) run, latency, accuracy, scene-complexity proxy, timestamp —
  one row per (frame, config) cell, written to Postgres (see Database above).
- Anything the pipeline depends on is a runnable script, not notebook-only
  logic. Notebooks are for exploration, not for producing artifacts other
  code reads.
- `run-simulation` never fetches data itself — it assumes the caller already
  has a COCO-format annotations file and every referenced image present
  locally (`--annotations`/`--images`), and fails clearly rather than trying
  to top up anything missing. Acquiring/downloading a dataset, if ever
  needed again, is a separate, explicitly-run concern outside this pipeline.
- `training/datagen/cli/`'s standalone entry points are registered as
  `[project.scripts]` in `training/pyproject.toml` — once the shared venv is
  activated, run them by name (e.g. `run-simulation --preset baseline`), not
  `python -m datagen.cli.<name>`. The CLI surface is deliberately kept to
  just two tools — `run-simulation` (the actual simulation run) and
  `score-complexity` (scores images for scene complexity) — rather than
  accumulating preview/debug/one-off tools; don't add a third without a real
  need. If a new one is genuinely warranted, it goes in both places: the
  module under `training/datagen/cli/` and an entry in
  `training/pyproject.toml`'s `[project.scripts]` (then `pip install -e
  ./training` again to regenerate the installed script).
- Seed all randomness (condition sampling, train/test split) for
  reproducibility.
- The frame-level train/test split is enforced by a helper function that
  guarantees no frame appears in both sets — not by convention.

## Code organization (all areas)

- One file, one concern: a hook file exports one hook, a route module handles
  one resource, a training script does one job (data loading, training, or
  eval — not all three stitched together).
- Tests live in a parallel `tests/` tree mirroring source
  (e.g. `server/tests/services/detection/test_detect.py` next to
  `server/api/services/detection/detect.py`), never colocated as
  `thing.test.ts` beside `thing.ts`.

## Comments

- One line, under ~100 chars. Two lines only when truly needed.
- Explain *why*, not what the code already says. Skip if self-evident.
- No multi-line block comments restating the obvious.

## Testing bar

Prototype bar, not production bar. Real tests are required for anything that
would silently corrupt research output if it broke: data-gen logging, the
frame-level train/test split, and the routing decision logic. UI polish and
unlikely error states don't need exhaustive coverage unless they'd affect
pipeline correctness.
