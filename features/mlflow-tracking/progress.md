# Progress — mlflow-tracking

## Task 1 — The tracking module
- Status: completed
- Started: 2026-09-29 10:59:10
- Completed: 2026-09-29 11:06:52
- Notes: birds/tracking.py (run context manager, RunHandle, log_metrics, BIRDS_TRACKING=off switch, warns and no-ops on MLflow failure); mlflow 3.16.1 installed (~50 deps, bumped shared fastapi/pydantic — training/database/server all still pass); mlruns gitignored; 5 new tests, 204 pass. MLflow 3 needs MLFLOW_ALLOW_FILE_STORE=true for a plain dir store.

## Task 2 — Track training runs
- Status: completed
- Started: 2026-09-29 11:06:52
- Completed: 2026-09-29 11:09:54
- Notes: fit gains keyword-only on_epoch_end (default None); main logs params, per-epoch train_loss/val_accuracy with step, best val+epoch; run_sidecar_path helper writes <ckpt>.run.json and removes a stale one when tracking is off; autouse conftest fixture sets BIRDS_TRACKING=off. 4 new tests, 208 pass, no mlruns/ created.

## Task 3 — Attach evaluation results to the training run
- Status: completed
- Started: 2026-09-29 11:09:54
- Completed: 2026-09-29 11:13:42
- Notes: evaluate.main resumes the training run from the sidecar and logs test_accuracy/mean_latency_ms metrics plus an eval_device param; absent or malformed sidecar falls back to a standalone run and prints a note. 7 new tests, 215 pass, no mlruns/.

## Task 4 — Retrain both models under tracking
- Status: completed
- Started: 2026-09-29 11:13:42
- Completed: 2026-09-29 11:26:03
- Notes: Retrained both under tracking. Results identical to pre-tracking: small best val 0.7879@27 test 0.7798; large best val 0.8973@13 test 0.8699. Store read back: 2 runs, all params, 30/15 epochs of metrics, test_accuracy+mean_latency_ms+eval_device on the SAME run, git_commit tag, sidecars match run ids, no standalone evaluate-* runs. Latency differs (7.40/7.73 vs 5.71/6.70 ms) — machine-load noise.

## Task 5 — Document the convention
- Status: completed
- Started: 2026-09-29 11:26:03
- Completed: 2026-09-29 11:26:53
- Notes: Three bullets in coding-guidelines under the training-code section: what is tracked and where, tracking must never break a run plus the test off-switch, and the UI command. Orchestrator confirmed the MLFLOW_ALLOW_FILE_STORE requirement is enforced in the shared FileStore class, so the documented command is correct.

## Task 6 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
