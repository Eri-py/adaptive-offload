# Progress — birds-revamp

## Task 1 — Rename flowers to birds
- Status: completed
- Started: 2026-09-28 23:25:09
- Completed: 2026-09-28 23:26:43
- Notes: git mv flowers->birds (package+tests), imports/paths/docs updated, DATA_DIR data/cub200, WEIGHTS_DIR models/birds, pyproject include coco*/birds*, reinstalled. Orchestrator added scratch/ to mypy exclude (mypy ignores .gitignore). 189 pass, ruff clean, mypy 9 known.

## Task 2 — CUB-200 data, split and download
- Status: completed
- Started: 2026-09-28 23:26:43
- Completed: 2026-09-28 23:29:11
- Notes: CUB loader + split_ids (5400/594/5794, val identical to pilot) + download (already present, 11788). models.py prose fixed. 8 new data tests replace flowers ones; 192 pass.

## Task 3 — Flowers review fixes and 200-way models
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 4 — Record the pilot rationale
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Train and evaluate on birds
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 6 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
