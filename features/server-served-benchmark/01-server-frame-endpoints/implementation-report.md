# Server Frame Endpoints — Implementation Report

**Feature:** Server-Served Benchmark, spec 01 — Server Frame Endpoints
**Directory:** `features/server-served-benchmark/01-server-frame-endpoints/`

## Tasks

Implemented in one pass rather than as numbered tasks, under the A6 progress-report
deadline (2026-10-27, 2:30 PM). The spec-driven flow's plan and review steps were
skipped deliberately for that reason — noted here so the gap is visible rather than
silent. Everything the spec's Acceptance Criteria list is covered by a test.

| Area | Outcome |
|------|---------|
| Frame-list endpoint | completed |
| Frame-image endpoint | completed |
| Eligibility query (complexity + local timing + file on disk) | completed |
| Deterministic even spread across the complexity range | completed |
| Database-unavailable handling | completed |
| Tests, one per acceptance criterion | completed |

## Files changed

The range is `main..feature/server-frame-endpoints`.

### Added

- `server/api/core/config.py`: `Settings` (dataset, images folder, default frame count) read from the environment with repo-local defaults, cached per process and used directly as a FastAPI dependency.
- `server/api/core/app.py`: `create_app()` — router wiring plus the handler that turns connection-level SQLAlchemy failures into a 503.
- `server/api/contracts/frames.py`: `BenchmarkFrame`, `BenchmarkFrameSet`.
- `server/api/services/benchmark/frames.py`: the eligibility join, `_evenly_spaced_indices`, `select_frames`, and `resolve_frame_image`.
- `server/api/controller/frames.py`: the two routes under `/benchmark`.
- `server/api/dependencies/database.py`: per-request session over `database/`'s shared engine, built lazily.
- `server/api/main.py`: `app = create_app()` for `uvicorn api.main:app`.
- `server/tests/support.py`, `server/tests/conftest.py`: `FrameFixture`/`seed_frames` helpers, the ephemeral-Postgres fixture (mirroring `database/tests/conftest.py`), and a client factory that seeds a database and wires an app to it.
- `server/tests/controller/test_frames.py` (10 tests), `server/tests/services/benchmark/test_frames.py` (4 tests).
- Package `__init__.py` files across `server/api/` and `server/tests/`.

### Edited

- `server/pyproject.toml`: `fastapi`/`uvicorn` as dependencies, `httpx`/`python-dotenv` as dev dependencies, and `mypy_path = "../database"` so `from common... import ...` type-checks (the same workaround `training/pyproject.toml` carries).

### Not changed

No migration, no model change, no image bytes written to Postgres — all three are explicit spec requirements.

## Tests

14 new tests, `server/` having had none before. Full repo suite afterwards: **237
passing, 0 skipped** (222 `training/`, 1 `database/`, 14 `server/`). `ruff check`
clean; `mypy --strict` clean (22 files).

### Controller

Default count of 15; an explicit count; fewer eligible frames than requested;
complexity spread including both ends; byte-identical repeat requests; exclusion of
frames with no local timing, with only an offload timing, or with no file on disk;
byte-for-byte JPEG delivery; 404 for an unknown identifier; 404 for an identifier
escaping the image folder (with the out-of-folder file left unserved); 503 when the
database is unreachable.

### Service

`_evenly_spaced_indices` spanning both ends; ordering by complexity with ties broken
on file name; the stored desktop timing being reported as-is; `resolve_frame_image`
rejecting unknown, relative-escape and absolute identifiers.

### Manual verification against real data

Run in-process (no server started, per `CLAUDE.md`) against the real
`adaptive_offload` database and the real `training/data/coco/val2017` folder:
`count=5` returned frames at complexity 0.0013 / 0.0575 / 0.0930 / 0.1415 / 0.3450
with desktop YOLOv8n times of 14.86–19.28 ms; the default returned 15 frames; the
first frame's image returned 18,810 bytes of `image/jpeg`, byte-identical to the file
on disk; an unknown identifier returned 404.

## Notable events

- **Eligibility spans two tables with no ORM `relationship()`**, and the spec forbids
  a schema change, so the join is written explicitly on the composite key
  (`dataset`, `file_name`) restricted to `model_path = LOCAL`. The file-on-disk check
  stays in Python — it can't be expressed in SQL — applied after the ordered query so
  ordering is still the database's.
- **The database lookup is what authorizes an image read**, rather than sanitizing the
  caller's string. `resolve_frame_image` returns `None` unless `scene_complexity` has
  the row, with a resolved-parent containment check as a second guard. Both the
  relative-escape and absolute-path cases are tested.
- **The 503 handler is registered for `OperationalError` and `InterfaceError`.**
  Anything narrower leaks a 500 when Postgres is simply down; anything broader
  (`SQLAlchemyError`) would hide real bugs behind "database unavailable".
- **The engine is built lazily** (`functools.lru_cache`), so importing the app doesn't
  require `DATABASE_URL` and a down database surfaces as a handled 503 rather than an
  import-time crash.
- **`server/.env`** was created locally (gitignored, copied from `database/.env`) so
  the test fixture finds `POSTGRES_ADMIN_URL`, mirroring `training/.env`.
- **Postgres was down for the first ~40 minutes of the session**, so nothing
  DB-backed could be verified until the user started it; the two tests that don't
  need a database (including the 503 case) were the only ones runnable until then.

## Out of scope, as specified

The app still bundles its own images (spec 02), and per-stage phone timing plus the
phone-vs-desktop comparison view is spec 03.
