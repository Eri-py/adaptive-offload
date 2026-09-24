# Progress — latency-budget-router

## Task 1 — Generic paired bootstrap
- Status: completed
- Started: 2026-09-24 17:21:27
- Completed: 2026-09-24 17:24:44
- Notes: bootstrap_mean_difference added; bootstrap_router_vs_offload delegates. 3 new tests, 145 pass. feature_experiment output byte-identical vs HEAD (in-place file swap; worktrees don't work with the editable install).

## Task 2 — Policies, oracle and metrics
- Status: completed
- Started: 2026-09-24 17:24:44
- Completed: 2026-09-24 17:28:51
- Notes: latency_policies.py (budget_only/cascade/oracle actions, outcomes, on-time accuracy, RouterMetrics); budget boundary inclusive; 12 new tests, 157 pass.

## Task 3 — Offload-latency predictor and confidence scores
- Status: completed
- Started: 2026-09-24 17:28:51
- Completed: 2026-09-24 17:32:16
- Notes: Offload-latency GBT regressor, raw_score (mean_confidence), learned_score (stage-1 p_local_good_enough on CONFIDENCE_COLUMNS). 6 new tests, 163 pass.

## Task 4 — Figure
- Status: completed
- Started: 2026-09-24 17:32:16
- Completed: 2026-09-24 17:36:54
- Notes: budget_plot.py (BudgetPoint, BudgetPanelData, plot_budget_curves; Agg; bbox_inches=tight fixed a clipped legend); matplotlib declared in pyproject; 3 new tests, 166 pass.

## Task 5 — Experiment script (integration)
- Status: completed
- Started: 2026-09-24 17:36:54
- Completed: 2026-09-24 17:45:09
- Notes: budget_experiment.py (run_experiment -> BudgetExperimentResult, report, build_budget_panel, main --dataset/--figure); 1 integration test, 167 pass; --help works.

## Task 6 — Run the experiment and write findings
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 7 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
