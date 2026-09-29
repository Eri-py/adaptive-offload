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
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 4 — Retrain both models under tracking
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Document the convention
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 6 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
