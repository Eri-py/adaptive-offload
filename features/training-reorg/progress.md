# Progress — training-reorg

## Task 1 — Capture the "before" baseline
- Status: completed
- Started: 2026-09-28 20:40:30
- Completed: 2026-09-28 20:44:11
- Notes: Baseline in scratchpad/reorg_before/: both experiments' stdout (stderr empty), figure, 174 passed, CLI --help, ruff clean, mypy 9 known. No tracked changes.

## Task 2 — Move the code and tests
- Status: completed
- Started: 2026-09-28 20:44:11
- Completed: 2026-09-28 20:52:13
- Notes: git mv datagen/router + tests under coco/; imports -> coco.*; 9 path lines fixed (weights parents[2], dotenv +1 parent); pyproject scripts/packages; editable reinstall. 174 pass, ruff clean, mypy same 9 at new paths.

## Task 3 — Verify behaviour is unchanged
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 4 — Update layout documentation
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
