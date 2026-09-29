# MLflow Experiment Tracking — Implementation Report

**Feature:** MLflow Experiment Tracking
**Directory:** `features/mlflow-tracking/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — The tracking module | completed |
| Task 2 — Track training runs | completed |
| Task 3 — Attach evaluation results to the training run | completed |
| Task 4 — Retrain both models under tracking | completed |
| Task 5 — Document the convention | completed |
| Task 6 — Regression test run | completed |

## Files changed

The range is `feature/birds-revamp..feature/mlflow-tracking`.

### Added

- `training/birds/tracking.py`: the only module that calls MLflow. A `run(...)` context manager, `log_metrics(...)`, the `BIRDS_TRACKING=off` switch and the failure handling.
- `training/tests/birds/test_tracking.py`
- `features/mlflow-tracking/spec.md`, `implementation.md`

### Edited

- `training/birds/config.py`: the store location and experiment name.
- `training/birds/train.py`: `fit` gained an optional per-epoch callback; `main` opens a run, logs settings, progress and outcome, and writes the run-id sidecar next to the checkpoint.
- `training/birds/evaluate.py`: `main` resumes each model's training run from its sidecar and logs test accuracy, timing and device; an absent or malformed sidecar falls back to a standalone run and prints a note.
- `training/tests/conftest.py`: an autouse fixture turns tracking off for the whole suite.
- `training/tests/birds/test_train.py`, `test_evaluate.py`: cover the new wiring.
- `training/pyproject.toml`: `mlflow>=2.14` added (3.16.1 installed).
- `.gitignore`: `training/mlruns/`.
- `.claude/coding-guidelines.md`: the convention and the command for viewing history.

## Tests

### Unit

- `training/tests/birds/test_tracking.py` (5 tests): the off switch yields no run id and writes nothing; a real run against a temporary store records its params and metrics, read back through MLflow's client; resuming by run id; an MLflow failure on start still runs the body and warns; an exception inside the body propagates.
- `training/tests/birds/test_train.py` (9 tests, 4 new): the per-epoch callback fires once per epoch with that epoch's values; the sidecar is written with the run id when tracking is on, absent when off, and a stale one is removed; plus the existing checkpoint, split-isolation and ordering tests.
- `training/tests/birds/test_evaluate.py` (7 tests, 7 new/rewritten): a present sidecar resumes that run and logs the metrics; six parametrised cases cover an absent, unparseable, wrong-shaped or empty sidecar, each falling back to a standalone run, still printing the table, and printing the note.

All tests pass: `training/` 215 passed, `database/` 1 passed, none skipped. Ruff is clean and mypy reports only the 9 pre-existing errors. Running the suite adds nothing to the MLflow store, verified by counting the store's files before and after.

## Commits

`feature/birds-revamp..feature/mlflow-tracking`: spec, plan, Tasks 1–3 and 5, and a commit recording the tracked retraining. Tasks 4 and 6 changed no tracked files.

## Notable events

- **Installing MLflow changed shared dependency versions.** It pulled in about 50 packages including its own web stack, and moved `fastapi` and `pydantic` in the virtual environment that `server/` also uses. All three packages were checked afterwards and still pass: training 215, database 1, server clean (it has no tests yet). Pip also warned of two unrelated version conflicts (`onnx` wanting newer protobuf, `xdsl` wanting older typing-extensions) that predate this change.
- **MLflow 3 refuses a plain-directory store** unless `MLFLOW_ALLOW_FILE_STORE=true`. `tracking.run` sets it internally, and the documented UI command sets it too. The requirement was confirmed to live in the shared store class both the client and the UI use.
- **Two subagent reports came back as the literal word "placeholder"** (Tasks 2 and 4). In both cases the orchestrator verified the work directly from the diff, the logs and the MLflow store instead of relying on the report; Task 2's real report arrived afterwards, Task 4's did not.
- **Retraining reproduced the previous results exactly**, which is the evidence that adding tracking changed nothing:

  | Model | Best validation (epoch) | Test accuracy | Previously |
  |---|---|---|---|
  | MobileNetV3-Large | 78.79% (27) | 77.98% | 77.98% |
  | ConvNeXt-Base | 89.73% (13) | 86.99% | 86.99% |

  Per-photo timing differed (7.40 and 7.73 ms against 5.71 and 6.70 ms). Timing is machine-load sensitive and was not expected to reproduce exactly.
- **The store was read back with MLflow's client** rather than inspected by eye: two runs named for their models, all settings recorded, 30 and 15 epochs of per-epoch metrics, the git commit tagged, and the evaluation's test accuracy, latency and device on the *same* run as the training. No standalone evaluation run was created, and both sidecars matched their run ids.
