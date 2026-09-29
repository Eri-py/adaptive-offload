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
- Status: completed
- Started: 2026-09-28 23:29:11
- Completed: 2026-09-28 23:31:27
- Notes: metrics.top1_accuracy shared (S2); train/evaluate build data before model/checkpoints (S1); new tests: missing checkpoint, per-model eval transform, step order + missing dataset builds no model (S3); one-line docstrings (S4); monkeypatch (N1); 200-way model tests. 196 pass.

## Task 4 — Record the pilot rationale
- Status: completed
- Started: 2026-09-28 23:31:27
- Completed: 2026-09-28 23:32:00
- Notes: birds/findings.md: pilot settings, flowers-vs-birds table, cascade table, label-smoothing caveat, 2-sentence interpretation. All numbers checked against pilot_results.txt/cub_pilot.log.

## Task 5 — Train and evaluate on birds
- Status: completed
- Started: 2026-09-28 23:32:00
- Completed: 2026-09-28 23:43:31
- Notes: small: best val 0.7879 @ 27/30, test 0.7798, 5.713 ms/photo CPU (6 threads), ~2m54s. large: best val 0.8973 @ 13/15, test 0.8699, 6.701 ms/photo CUDA, ~5m27s. Pilot test was 0.7727 / 0.8676. Checkpoints untracked.

## Task 6 — Regression test run
- Status: completed
- Started: 2026-09-28 23:43:31
- Completed: 2026-09-28 23:43:47
- Notes: training: ruff clean, mypy 9 known, 196 passed; database: 1 passed.
