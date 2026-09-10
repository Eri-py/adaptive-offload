## Task 1 — Split presets out of config
- Status: completed
- Started: 2026-09-10 13:02:57
- Completed: 2026-09-10 13:04:48
- Notes: Mechanical move, no consumer missed. Test count preserved at 48 (relocated, not lost). ruff/mypy clean.

## Task 2 — Complexity-scoring CLI
- Status: completed
- Started: 2026-09-10 13:04:58
- Completed: 2026-09-10 13:06:29
- Notes: score_folder(folder) -> dict[str, float], no DB, no COCO dependency. ruff/mypy clean, 52/52 tests passed (4 new).

## Task 3 — Stratified-sample preview CLI
- Status: completed
- Started: 2026-09-10 13:06:40
- Completed: 2026-09-10 13:14:02
- Notes: preview_sample(engine, dataset, frame_count, bucket_count, seed) -> list[tuple[str, float]]. ruff/mypy clean, 55/55 tests passed (3 new). No stray test database left behind.

## Task 4 — Condition-vector preview CLI
- Status: completed
- Started: 2026-09-10 13:09:44
- Completed: 2026-09-10 13:10:38
- Notes: preview_conditions(preset_name, count, seed) -> list[tuple[float, float, float, float]]. No DB/dotenv at all. ruff/mypy clean, 58/58 tests passed (3 new).

## Task 5 — get_run_results + re-labeling CLI
- Status: completed
- Started: 2026-09-10 13:11:13
- Completed: 2026-09-10 13:13:49
- Notes: get_run_results(engine, run_id) -> list[ResultRow]; relabel_run(engine, run_id, lambda_value) -> list[tuple[str, Label, Label]]. Never calls create_run/store_results. ruff/mypy clean, 64/64 tests passed (6 new). No stray test database left behind.

## Task 6 — Split COCO acquisition: coco.py refactor + cache-population CLI
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 7 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
