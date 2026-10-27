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
- Status: completed
- Started: 2026-10-27 13:24:30
- Completed: 2026-10-27 13:26:47
- Notes: `write_predictions` (delete-then-insert for one dataset/split/model)
  and `read_predictions`, ordered by numeric photo id to match `splits.py`.
  The reader also filters on dataset and model, not split alone as the plan
  worded it — the PK includes `model_name`, so a split-only read would mix
  the two models into one frame. 3 new tests (230 total) on the ephemeral
  Postgres fixture; ruff clean, mypy still 9.

## Task 4 — Recording entry point
- Status: completed
- Started: 2026-10-27 13:27:23
- Completed: 2026-10-27 13:31:14
- Notes: `splits.split_labels()` added (the approved amendment, plan amended
  in 23edb5b); `record_split(...)` and `main()` added to `predictions.py`.
  6 new tests (236 total), ruff clean, mypy still 9. All four success criteria
  have tests, including the integration one on the ephemeral database.
  The subagent's hand-back report was empty (the literal "placeholder"), so
  this was accepted on direct verification of the diffs and the gate rather
  than on its report. Two design smells recorded in `learnings.md` for the
  reviewer: private-name imports from `evaluate.py`, and a predictions/store
  import cycle worked around with a local import.

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
