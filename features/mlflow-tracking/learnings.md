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
