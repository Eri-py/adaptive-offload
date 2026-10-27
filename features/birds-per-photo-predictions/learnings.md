# Learnings — birds-per-photo-predictions

- Task 1: `python`/`ruff`/`mypy` are not on PATH; use the repo-root `.venv/bin` (`export PATH=/home/eriol/projects/adaptive-offload/.venv/bin:$PATH`). `ruff format --check` flags two pre-existing files (`common/models.py` SimulationRun line, `migrations/env.py`) that predate this task; `ruff check` and `mypy` are clean.
- Task 2: `split_ids()` returns ids only, so `predict_photos` takes `expected_labels` from the caller; there is no public accessor for a split's labels (CubDataset._records is private), so Task 3 must supply them (e.g. add one to splits.py) or the alignment check has nothing to compare against.
- Task 3: BirdPrediction's PK excludes split, so write_predictions deletes by (dataset, split, model_name) while reads filter the same triple; photo-id ordering is done in pandas (int key) since ids are stringified ints and SQL string order would put "10" before "2".
