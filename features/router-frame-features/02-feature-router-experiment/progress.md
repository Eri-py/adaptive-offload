# Progress — feature-router-experiment

## Task 1 — Frame-level data loading
- Status: completed
- Started: 2026-09-24 13:42:05
- Completed: 2026-09-24 13:45:21
- Notes: frame_dataset.py (load_frame_table, load_simulated_frame_ids, load_simulated_rows, IMAGE_COLUMNS, CONFIDENCE_COLUMNS); 6 new tests, 115 pass.

## Task 2 — Feature diagnostics
- Status: completed
- Started: 2026-09-24 13:45:21
- Completed: 2026-09-24 13:47:08
- Notes: diagnostics.py compute_feature_diagnostics (Spearman, |rho| desc); 5 new tests, 120 pass.

## Task 3 — Evaluation and bootstrap
- Status: completed
- Started: 2026-09-24 13:47:08
- Completed: 2026-09-24 13:51:03
- Notes: evaluation.py (utilities, decide-first/cascade scoring, summarize, vectorised photo-level bootstrap); 9 new tests, 129 pass.

## Task 4 — Two-stage routers
- Status: completed
- Started: 2026-09-24 13:51:03
- Completed: 2026-09-24 13:58:36
- Notes: two_stage.py (train/predict stage1+2, cascade_label, run_router -> RouterResult with .picks plus trained-frame sets for leakage tests); single-class stage-2 label -> constant pick; 6 new tests, 135 pass.

## Task 5 — Experiment script (integration)
- Status: completed
- Started: 2026-09-24 13:58:36
- Completed: 2026-09-24 14:05:14
- Notes: feature_experiment.py (run_experiment -> ExperimentResult, main, report); fixed a silent row-order misalignment by deriving the scoring rows the same way run_router does; 1 integration test, 136 pass.

## Task 6 — Run the experiment and write findings
- Status: completed
- Started: 2026-09-24 14:05:14
- Completed: 2026-09-24 14:08:45
- Notes: Real run x2 identical. All four routers below always-offload with 95% CIs entirely < 0. router.baseline reproduces 0.6812 / 0.6811. findings.md section appended.

## Task 7 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
