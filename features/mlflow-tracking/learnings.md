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
