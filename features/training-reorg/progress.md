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
- Status: completed
- Started: 2026-09-28 20:52:13
- Completed: 2026-09-28 20:54:53
- Notes: Both experiments' stdout, the figure, and both --help outputs byte-identical to baseline; 174 both times; ruff same; mypy same 9 (paths moved; file count +4 = new package markers). Committed figure unchanged after default-path run.

## Task 4 — Update layout documentation
- Status: completed
- Started: 2026-09-28 20:54:53
- Completed: 2026-09-28 20:57:23
- Notes: Guidelines (layout paragraph + paths), implementer.md quality-gate paths, findings.md run commands updated. One stale old-path comment remains in database/migrations/env.py (out of scope; was already wrong pre-move).

## Task 5 — Regression test run
- Status: completed
- Started: 2026-09-28 20:57:23
- Completed: 2026-09-28 20:57:40
- Notes: training: ruff clean, mypy 9 known, 174 passed; database: 1 passed. No files changed.
