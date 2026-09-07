# Coding Guidelines — adaptive-offload

## Scope & bar

This is a single-semester research prototype (Senior Seminar), not a production
system. Optimize for iteration speed and correctness of the research pipeline
(routing decisions, data-gen labels, benchmark numbers) — not for scale,
security hardening, or exhaustive production robustness.

## Repo layout

- `app/` — React Native (iOS-first) client: capture trigger, scene-complexity
  proxy, cached network/load reads, calls to the local model and/or server.
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
  - `server/common/` — utilities shared across `api/` and `tests/`, plus the
    DB engine/session setup and shared table models (SQLAlchemy) — this is
    also what `training/` imports to reach the same Postgres instance.
  - `server/tests/` — mirrors `server/api/`'s structure, grouped by the same
    domain folders (e.g. `tests/services/detection/`).
- `training/` — all research/training code in one place, with subfolders by
  concern:
  - `training/datagen/` — the data-gen simulator: frame set, condition
    sweep, per-(frame, config) logging, dataset export.
  - `training/router/` — decision-layer code: feature extraction,
    direct-classifier and utility-regression models, training/eval scripts,
    saved model artifacts.

Each top-level area (`app/`, `server/`, `training/`) owns its own dependency
manifest and lint/type config — don't share one config across them pretending
they're the same kind of code. `server/` and `training/` are both Python but
serve different purposes (serving vs. research), so keep their dependency
files separate even if a shared internal package later makes sense.

## Database

- One existing Postgres instance backs everything — server-side
  request/routing logs *and* the data-gen simulator's per-(frame, config)
  sweep results. No SQLite, no separate CSV/Parquet store. There is no
  `docker-compose.yml` for this — the instance and database already exist
  outside the repo.
- `server/common/` owns the SQLAlchemy engine/session setup and the shared
  table models. Both `server/api/services/` and `training/` import from
  there rather than opening their own connections or redefining tables.
- The data-gen simulator (`training/datagen/`) writes each (frame, config) row
  straight to its Postgres table as it's produced.
- Training code (`training/router/`) builds its working DataFrame with a SQL
  query against that table (`pd.read_sql(query, engine)`), not by reading
  files off disk.
- Schema changes go through a migration script (e.g. Alembic), authored in
  the repo. See the repo's `CLAUDE.md` for the rule on who runs it.

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
- Cross-cutting helpers used by more than one service or by tests go in
  `server/common/`, not duplicated per-service.

## Data-gen / router training code (`training/datagen/`, `training/router/`)

- Every simulation run logs enough to reproduce it: frame id, condition
  params, path(s) run, latency, accuracy, scene-complexity proxy, timestamp —
  one row per (frame, config) cell, written to Postgres (see Database above).
- Anything the pipeline depends on is a runnable script, not notebook-only
  logic. Notebooks are for exploration, not for producing artifacts other
  code reads.
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
