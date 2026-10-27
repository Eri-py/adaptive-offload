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
- Status: completed
- Started: 2026-10-27 13:31:48
- Completed: 2026-10-27 13:35:19
- Notes: `summarize()` is pure over two DataFrames; `load_summary()` reads both
  models' test rows; `main()` prints and logs one standalone MLflow run.
  Cascade is a stable sort on (confidence, numeric photo id), so ties cannot
  reorder between runs. Escalated count is floor (`n * percent // 100`), which
  differs by one photo from the throwaway run's rounding at 5% and 20% — the
  real figures will differ in the last decimal from the numbers quoted during
  planning, which is why the plan forbade hardcoding them. 10 new tests
  (246 total), ruff clean, mypy still 9. Only public APIs used, no new import
  cycle.

## Task 6 — Migration, then the real run
- Status: completed
- Started: 2026-10-27 13:36:30
- Completed: 2026-10-27 13:44:06
- Notes: Migration 0004 applied by the user at ~13:36 (0003 -> 0004,
  `bird_predictions` created); DDL had been verified offline beforehand via
  `alembic upgrade 0003:0004 --sql`.
  First real run **failed**: `main()` in both new entry points calls
  `get_engine()` without loading `training/.env`, so `DATABASE_URL` was unset.
  Every COCO entry point that touches Postgres calls
  `load_dotenv(<training>/.env)` inside `main()` first; the new ones skipped
  it, and the unit tests missed it because they monkeypatch `get_engine`.
  Fixed by sharing an `ENV_PATH` constant and calling `load_dotenv` in both
  `main()`s, plus a test pinning the resolved path (247 tests).
  Re-run succeeded. All four criteria met:
  - 23,576 rows (5,400 train / 594 val / 5,794 test per model), and every row
    carries its checkpoint's `training_run_id`.
  - Derived test accuracies 0.7798 / 0.8699 match `evaluate.py`'s independent
    figures exactly.
  - Two consecutive summary runs produced byte-identical output (`diff` clean).
  - Output captured for Task 7 at
    `<scratchpad>/summary_run1.txt`.
  New since the pilot: the large model's own confidence AUCs (0.8816
  max-probability, 0.8864 top-2 margin), and top-2 margin beats max
  probability at every escalation level.

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
