# Learnings — mlflow-tracking

## Task 1
- pip resolved mlflow 3.16.1 (>=2.14 allows it); install took ~2m45s and pulled ~50 packages
  (Flask, fastapi, uvicorn, pyarrow, pydantic 2.13, typing-extensions 4.16, opentelemetry, ...).
  pip warned onnx wants protobuf>=6.31.1 (have 5.29.6) and xdsl wants typing-extensions<4.13.
- MLflow 3.x rejects a plain-directory file store unless MLFLOW_ALLOW_FILE_STORE=true. tracking.run
  sets it via os.environ.setdefault; `mlflow ui --backend-store-uri training/mlruns` will need the
  same env var (or a sqlite store instead).
- Patch MLflow in tests via the string path "birds.tracking.mlflow.start_run"; mypy strict rejects
  `tracking.mlflow` as a non-exported attribute.

## Task 2
- `train.main` now touches WEIGHTS_DIR (sidecar write/unlink) even with tracking off, so main-level
  tests must monkeypatch `train_module.WEIGHTS_DIR` to tmp_path or they would delete a real sidecar.
- Suite-wide off switch is an autouse fixture in tests/conftest.py (monkeypatch.setenv), so a test
  can still opt in with monkeypatch.setenv; `run_sidecar_path` lives in birds/train.py.
- mypy strict rejects `train_module.tracking` (implicit re-export); import `birds.tracking` in tests.

## Task 3
- Resuming a run with `mlflow.start_run(run_id=..., run_name=X)` renames it to X (fluent.py updates
  the name on resume), so evaluate resumes with the training run's own name ("train-<model>").
- evaluate imports `run_sidecar_path` from birds.train (no cycle; train never imports evaluate).
- Patch `tracking.run` / `tracking.log_metrics` on the `birds.tracking` module in tests; `ev.tracking`
  fails mypy strict (implicit re-export).

## Task 4
- Retraining is deterministic here: val/test accuracies reproduced the recorded values exactly
  (small 0.7879@27 / 0.7798; large 0.8973@13 / 0.8699). Runs took ~3 min (small, 30 epochs) and
  ~5.5 min (large, 15 epochs), evaluation ~2 min.
- The "birds" experiment is created on first tracked run; the store also has an empty "Default" experiment.
- Latency is noisy run to run (small 5.71 -> 7.40 ms CPU, large 6.70 -> 7.73 ms CUDA); accuracy is not.
- Read-back needs MLFLOW_ALLOW_FILE_STORE=true plus set_tracking_uri(MLRUNS_DIR.as_uri()).
