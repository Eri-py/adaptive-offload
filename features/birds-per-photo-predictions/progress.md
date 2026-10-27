# Progress — Birds Per-Photo Predictions

Base: `main` at `d40a578`. Branch: `feature/birds-per-photo-predictions`.

## Task 1 — Table model and migration
- Status: completed
- Started: 2026-10-27 13:19:07
- Completed: 2026-10-27 13:20:40
- Notes: `bird_predictions`, PK (dataset, photo_id, model_name). Model and
  migration `0004` verified column-for-column identical. Migration authored
  only, not applied. `database/`: ruff clean, mypy clean, 1 test passed.

## Task 2 — Per-photo record computation
- Status: completed
- Started: 2026-10-27 13:20:57
- Completed: 2026-10-27 13:23:59
- Notes: `PhotoPrediction` frozen dataclass + `predict_photos(...)`, no DB
  imports. 5 new tests (227 total), ruff clean, mypy still 9. Raised a gap
  the plan's file table missed: a split's labels have no public accessor, so
  the alignment check needs one added to `birds/dataset/splits.py` — see
  Task 4.

## Task 3 — Record store
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 4 — Recording entry point
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Summary tables
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 6 — Migration, then the real run
- Status: not started
- Started: —
- Completed: —
- Notes: Stops for the user to run migration 0004 (`CLAUDE.md`, Infrastructure).

## Task 7 — Rewrite the findings
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 8 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
