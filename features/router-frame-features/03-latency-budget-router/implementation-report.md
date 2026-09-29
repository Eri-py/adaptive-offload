# Latency-Budget Router — Implementation Report

**Feature:** Latency-Budget Router
**Directory:** `features/router-frame-features/03-latency-budget-router/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Generic paired bootstrap | completed |
| Task 2 — Policies, oracle and metrics | completed |
| Task 3 — Offload-latency predictor and confidence scores | completed |
| Task 4 — Figure | completed |
| Task 5 — Experiment script (integration) | completed |
| Task 6 — Run the experiment and write findings | completed |
| Task 7 — Regression test run | completed |

## Files changed

These are the files this sub-plan changed, from `7bb55f3` to `HEAD` on
`feature/router-frame-features`. The spec (`e923ed1`, reworded in `7bb55f3`)
and the plan (`de60976`) were committed before execution started.

### Added

- `training/router/latency_policies.py`
- `training/router/budget_plot.py`
- `training/router/budget_experiment.py`
- `training/router/latency_budget_curves.png`
- `training/tests/router/test_latency_policies.py`
- `training/tests/router/test_budget_plot.py`
- `training/tests/router/test_budget_experiment.py`

### Edited

- `training/router/evaluation.py`: new `bootstrap_mean_difference`; `bootstrap_router_vs_offload` now delegates to it, with identical output.
- `training/tests/router/test_evaluation.py`: tests for the generic bootstrap.
- `training/router/findings.md`: new "Latency-budget experiment (spec 03)" section; existing content unchanged.
- `training/pyproject.toml`: `matplotlib>=3.8` declared, since it is now imported directly.

## Tests

### Unit

- `training/tests/router/test_latency_policies.py` (18 tests): both policies branch by branch, with the budget boundary inclusive. Also covers escalation cost, on-time zeroing, the oracle's choice and fallback, and a seeded property check that the oracle is never below either policy. For the predictor and scores: the offload-latency model tracks a 1/bandwidth relationship, the learned score never trains on a simulated frame, and both scores stay in [0, 1] and are deterministic.
- `training/tests/router/test_evaluation.py` (3 new tests): the generic bootstrap matches a hand-built case, is deterministic, and the existing bootstrap gives the same result as the generic one.
- `training/tests/router/test_budget_plot.py` (3 tests): writes a valid PNG, handles missing scores or points, and rejects an empty input.

### Integration

- `training/tests/router/test_budget_experiment.py` (1 test, ephemeral Postgres): an end-to-end run on a synthetic dataset, twice. Results are identical, all budgets and policies are present, every number is finite, and the oracle is never below any policy.

All tests pass: `training/` 167 passed (142 before + 25 new), `database/` 1 passed, none skipped.

## Commits

`7bb55f3..HEAD` on `feature/router-frame-features`: one commit per task for Tasks 1–6. Task 7 changed no files.

## Notable events

- **Before/after checks in this repo can't use a `git worktree`.** The shared venv's editable install always imports the main working tree. Task 1 swapped the file in place to confirm spec 02's output was byte-identical. The orchestrator also confirmed that spec 02's current output matches its `findings.md` section exactly. That covers an earlier review fix (spec 02 S3) whose worktree-based check had compared the new code with itself.
- **Task 4 inspected the figure visually** and fixed a clipped legend, which the smoke test couldn't catch.
- **Task 6's first draft of the findings needed revising before commit.** It reported the formal verdict correctly: the tuned cascade is not useful at any budget compared with budget-only. But its recommendation omitted the table's main result:
  - budget-only lands within 0.4–1.6 points of the oracle's on-time accuracy at every budget
  - it beats both always-local and always-offload

  The orchestrator had the section revised to lead with that result and to recommend resuming the phone/server work to test it on real hardware. The orchestrator also added a caveat that the cited iPhone latencies come from a Debug build.
- **Real results (read-only, two identical runs):**
  - the cascade is not useful at any of the six budgets; every interval against budget-only lies below zero
  - budget-only against the oracle, on-time accuracy:

    | Budget (ms) | 100 | 150 | 200 | 300 | 400 | 500 |
    |---|---|---|---|---|---|---|
    | Budget-only | 0.4447 | 0.6425 | 0.6481 | 0.6839 | 0.7294 | 0.7651 |
    | Oracle | 0.4484 | 0.6439 | 0.6543 | 0.6958 | 0.7454 | 0.7727 |

  Budget-only against the fixed strategies was not bootstrap-tested; the findings recommend that as a follow-up.
