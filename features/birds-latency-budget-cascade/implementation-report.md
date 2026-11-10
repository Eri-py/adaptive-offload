# Birds Cascade Under a Hard Latency Budget — Implementation Report

## Feature

Birds cascade under a hard latency budget —
`features/birds-latency-budget-cascade/`.

Evaluates the committed birds cascade (MobileNetV3-Large escalating to
ConvNeXt-Base on low confidence) against a hard per-request latency budget,
using the per-photo prediction records on `main`. Birds latency does not vary
per photo, so local and offload cost are scenario parameters rather than
per-row columns; the experiment is a grid over (local cost, offload cost) by
budget.

## Tasks

| Task | Outcome |
|---|---|
| 1 — Cost model, scenario policies and cell evaluation | Completed |
| 2 — Boundary heatmap | Completed |
| 3 — Wire the entry point | Completed |
| 4 — Real run and write-up | Completed |
| 5 — Regression test run | Completed (nothing changed, no commit) |

## Files changed

**Added**

| Path | Purpose |
|---|---|
| `training/birds/router/__init__.py` | Package marker. |
| `training/birds/router/conditions.py` | Cost model: payload, named network conditions, offload cost composition, the swept ranges. |
| `training/birds/router/policies.py` | Scenario-level policy outcomes, the within-budget oracle, band classification, lockout width. |
| `training/birds/router/budget_experiment.py` | Threshold tuning, cell evaluation, the closed-form grid, report formatting, CLI entry point. |
| `training/birds/router/boundary_plot.py` | The boundary heatmap. |
| `training/birds/router/latency_budget_boundary.png` | The rendered figure the write-up links. |
| `training/tests/birds/router/` (5 files) | Tests, listed below. |

**Edited**

| Path | Purpose |
|---|---|
| `findings/birds.md` | New section carrying the run's results. |

Workflow artifacts (`spec.md`, `implementation.md`, `progress.md`,
`learnings.md`) are excluded from this list.

## Tests

28 tests across four files, all passing. Full suite: 290 passed, none skipped.

**Unit**

- `test_conditions.py` (4 tests) — transfer-time arithmetic, offload cost
  composition, the named conditions' costs.
- `test_policies.py` (3 tests) — per-photo outcomes per action, the
  within-budget oracle and its tie-breaking, band classification including the
  case where offload is cheaper than local.
- `test_boundary_plot.py` (5 tests) — cell classification and tie tolerance,
  the marks drawn on each panel, that attempt 1's COCO phone figures appear
  nowhere on the figure, and that the figure saves.
- `test_budget_experiment.py` (14 of 16 tests) — hand-computed metrics for all
  five policies, threshold tuning, the fewest-escalations search, budget
  validation, the closed-form grid against `evaluate_cell` across all four
  bands, and that the report carries both required bootstrap intervals.

**Integration**

- `test_budget_experiment.py` (2 of 16 tests) — the entry point against a real
  ephemeral Postgres database: that it prints the tables, writes the figure and
  is deterministic across two runs, and that missing records stop the run before
  anything is written.

## Commits

`main..HEAD` — `c1a47b0..1a9af06`, ten commits.

## Notable events

- **Task 4 was blocked for the length of a session** on the large model's
  measured per-photo GPU latency, which the entry point requires and which has
  no default by design. It was unblocked by running
  `python -m birds.evaluation.evaluate` on the user's explicit authorisation,
  giving 6.731 ms. That run also reproduced both models' test accuracies
  (0.7798 / 0.8699) unchanged.
- **Task 2 found that per-cell threshold tuning was both too slow and
  unnecessary.** A full-resolution grid would have taken roughly 110 minutes per
  budget. The tuned threshold is identical across every cell of the escalation
  band and irrelevant outside it, so it is computed once per confidence measure;
  a full run now takes about 8 seconds. A test pins the closed-form grid against
  `evaluate_cell` across all four bands so the fast path cannot drift.
- **Task 4 surfaced an acceptance-criterion miss from Task 3.** The spec asks
  for bootstrap intervals against budget-only *and* always-local;
  `vs_always_local` was computed but never printed, and `format_report` had no
  test. Fixed in commit `1a9af06`, which adds the interval, a test for it, and a
  background box for a figure label that was illegible against the fill.
- **An implementer accidentally ran `ruff format .` across `training/`** during
  Task 3, reformatting unrelated files. Reverted; only the task's own files were
  committed.
