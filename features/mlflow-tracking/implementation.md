# MLflow Experiment Tracking — Implementation Plan

> **Superseded in part (2026-09-29).** The plan below describes the local file
> store this feature originally shipped. After review, the backend moved to a
> dedicated `mlflow` Postgres database and the module moved from
> `training/birds/tracking.py` to `training/shared/tracking.py`. See the amendment
> note in `spec.md` and the "Post-review changes" section of
> `implementation-report.md`.

## Summary

Add a small tracking module to `training/birds/` that records each training
run's settings, per-epoch progress and outcome to a local MLflow file store,
and attaches the evaluation's test accuracy and timing to the run that
produced each model. Then retrain both bird models under tracking so the
history starts complete.

## Approach & Key Decisions

- **Tracking lives in the entry points, not in `fit`.** `train.main` and
  `evaluate.main` open and close runs; `fit` gains one optional callback
  invoked after each epoch. `fit` stays a pure function over its arguments,
  so its existing tests are unaffected and nothing in the training loop
  depends on MLflow.
- **One module owns every MLflow call:** `birds/tracking.py`. Everything else
  imports two things from it: a `run(...)` context manager and a
  `log_metrics(...)` helper. Concentrating it there is what makes the next
  two decisions cheap.
- **Failures never break training.** The context manager catches any
  exception from MLflow itself, prints a clear warning naming what failed,
  and yields a no-op handle so the caller's code runs unchanged. Exceptions
  from the caller's own body propagate normally — only tracking is
  swallowed. A run that fails to record still trains and still prints its
  result.
- **Tests never write run history.** `tracking.run` checks an environment
  variable (`BIRDS_TRACKING=off`) and no-ops when set; `training/tests/conftest.py`
  sets it for the whole suite. The existing `main` tests therefore keep
  passing untouched, and the tracking module's own tests drive MLflow
  explicitly against a `tmp_path` store rather than relying on the default.
- **Evaluation attaches to the training run via a sidecar file.** `train.main`
  writes `<checkpoint>.run.json` holding the MLflow run id next to the
  checkpoint it saved; `evaluate.main` reads it and logs the test metrics
  into that same run. A sidecar rather than embedding the id in the
  checkpoint, so the checkpoint format is unchanged and a checkpoint trained
  before tracking still evaluates — it just starts a standalone run instead,
  with a note in the printed output.
- **Store location:** `training/mlruns/`, resolved from `__file__` so the
  path does not depend on the working directory, and added to `.gitignore`.
  The user runs `mlflow ui --backend-store-uri training/mlruns` themselves;
  no code starts it (`CLAUDE.md` "Infrastructure").

## Out of Scope

- A tracking server, database-backed store, or anything shared beyond this machine.
- Logging checkpoints, datasets or figures as artifacts.
- Tracking the simulator, router experiments or data-generation pipeline.
- Hyperparameter search, model registry, automated run comparison.
- Retrofitting `training/coco/`.
- Any change to what training does or the results it produces.

## Dependencies and Configuration

- `training/pyproject.toml`: add `mlflow>=2.14` to `dependencies`, then
  `pip install -e training` into the shared venv. MLflow is a sizeable
  install (it pulls in its own web and packaging stack), but only the
  tracking client is used.
- `.gitignore`: add `training/mlruns/`.
- If mypy lacks type information for `mlflow`, add a
  `[[tool.mypy.overrides]]` entry with `ignore_missing_imports`, matching the
  existing `cv2`, `scipy`, `pandas` and `torchvision` entries.
- No database or schema change.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/birds/tracking.py` | add | The only module that calls MLflow: store location, `run` context manager, `log_metrics`, the off switch and failure handling | One place owns tracking |
| `training/birds/config.py` | edit | The store path and experiment name | Settings live here |
| `training/birds/train.py` | edit | `fit` gains an optional per-epoch callback; `main` opens a run, logs settings and outcome, writes the sidecar | Record training |
| `training/birds/evaluate.py` | edit | `main` resumes each model's run and logs test accuracy, timing and device | Record evaluation |
| `training/tests/conftest.py` | edit | Turn tracking off for the whole suite | Tests write no history |
| `training/tests/birds/test_tracking.py` | add | The off switch, failure handling, and a real run against a `tmp_path` store | Cover the new module |
| `training/tests/birds/test_train.py` | edit | The per-epoch callback fires with the right values; the sidecar is written | Cover the new wiring |
| `training/tests/birds/test_evaluate.py` | edit | A present sidecar resumes that run; a missing one still evaluates | Cover the fallback |
| `training/pyproject.toml` | edit | The dependency and any mypy override | New package |
| `.gitignore` | edit | Ignore the store | Keep history out of git |
| `.claude/coding-guidelines.md` | edit | State that training runs are tracked, and how to view them | The standing convention |

## Tasks

### Task 1 — The tracking module

- **Objective:** One module that owns every MLflow call, is off in tests, and
  never breaks its caller.
- **Files:** `training/birds/tracking.py`, `training/birds/config.py`,
  `training/tests/birds/test_tracking.py`, `training/pyproject.toml`,
  `.gitignore`
- **Details:**
  - `config.py`: `MLRUNS_DIR = <training>/mlruns` (resolved from `__file__`)
    and `EXPERIMENT_NAME = "birds"`.
  - `tracking.py`:
    - `run(run_name, params, *, run_id=None)`: a context manager that sets
      the tracking URI to `MLRUNS_DIR`, selects the experiment, starts a run
      (resuming `run_id` when given), logs `params`, tags the run with the
      current git commit, and yields a handle exposing the run id.
    - `log_metrics(handle, metrics, *, step=None)`.
    - Off switch: when `BIRDS_TRACKING` is `off`, `run` yields a no-op handle
      whose id is `None` and logs nothing, without importing or calling MLflow.
    - Failure handling: any exception raised by MLflow inside `run`'s own
      setup or teardown is caught, printed as a warning naming the failure,
      and replaced by the no-op handle. The caller's body always executes,
      and exceptions from the body are not swallowed.
    - The git commit is read with `git rev-parse --short HEAD`; if that
      fails, the tag is omitted rather than raising.
  - Add `mlflow>=2.14` to `pyproject.toml` and install it; add
    `training/mlruns/` to `.gitignore`.
  - Tests (no GPU, no dataset):
    - with the off switch set, a run yields an id of `None`, writes nothing
      to a `tmp_path` store, and the body still runs
    - with tracking on and the store pointed at `tmp_path`, a run records its
      params and metrics, readable back through MLflow's own client
    - when MLflow raises on start, the body still runs and a warning is
      printed (patch the module's MLflow entry point)
    - an exception raised inside the body propagates
- **Success criteria:**
  - New tests pass, ruff is clean, mypy has no new errors.
  - `git status` never reports `training/mlruns/`.

### Task 2 — Track training runs

- **Objective:** Record each training run's settings, progress and outcome.
- **Files:** `training/birds/train.py`, `training/tests/conftest.py`,
  `training/tests/birds/test_train.py`
- **Details:**
  - `fit` gains a keyword-only `on_epoch_end: Callable[[EpochResult], None] | None = None`,
    called after each epoch's validation. Default `None` keeps current
    behaviour, so existing tests are unaffected.
  - `main` wraps the training in `tracking.run`, logging:
    - params: model name, epochs, learning rate, weight decay, batch size,
      seed, dataset (`cub200`), class count
    - per-epoch metrics via the callback: training loss and validation
      accuracy, with the epoch as the step
    - final metrics: best validation accuracy and best epoch
  - After `fit` returns, `main` writes `<checkpoint>.run.json` containing the
    run id, next to the checkpoint. When tracking is off or failed (id is
    `None`), no sidecar is written and any stale one is removed, so
    evaluation never attaches results to an unrelated earlier run.
  - `training/tests/conftest.py` sets `BIRDS_TRACKING=off` for the suite.
  - Tests: the callback fires once per epoch with that epoch's values; the
    sidecar is written with the run id when tracking is on and absent when
    off; a stale sidecar is removed.
- **Success criteria:** New and existing tests pass; ruff clean; no new mypy
  errors; the suite writes no `mlruns/`.

### Task 3 — Attach evaluation results to the training run

- **Objective:** Record each model's test accuracy and timing against the run
  that produced it.
- **Files:** `training/birds/evaluate.py`, `training/tests/birds/test_evaluate.py`
- **Details:**
  - For each model, `main` reads `<checkpoint>.run.json`. If present, it
    resumes that run and logs `test_accuracy`, `mean_latency_ms` and the
    device as a param or tag. If absent, it opens a standalone run named for
    the model and notes in the printed output that the results are not
    attached to a training run.
  - The printed table is unchanged apart from that note.
  - Tests: with a sidecar present, the resumed run id matches it; with none,
    evaluation still completes and prints the table (tracking is off in
    tests, so assert on the call rather than on stored history).
- **Success criteria:** New and existing tests pass; ruff clean; no new mypy
  errors.

### Task 4 — Retrain both models under tracking

- **Objective:** Backfill the history by retraining and re-evaluating.
- **Files:** none changed (checkpoints and run history are gitignored).
- **Details:**
  - From `training/`, run `python -u -m birds.train --model small`, then
    `--model large`, then `python -u -m birds.evaluate`. Run each in the
    background with the output logged, and wait for completion.
  - Confirm each run appears in the store with its params, per-epoch metrics
    and outcome, and that evaluation attached test accuracy and timing to the
    right run. Read this back with MLflow's client rather than by eye.
  - Compare against the recorded results: test accuracy 77.98% (small) and
    86.99% (large). Report any difference above one point rather than
    accepting it.
  - Do not start `mlflow ui`.
- **Success criteria:**
  - Two training runs and their evaluation metrics are in the store.
  - Test accuracies are within a point of the recorded values, or the
    difference is reported.
  - `git status` reports no new tracked files.

### Task 5 — Document the convention

- **Objective:** Record that training runs are tracked.
- **Files:** `.claude/coding-guidelines.md`
- **Details:** In the training-code section, state that every training run is
  tracked in MLflow, that the store is the gitignored `training/mlruns/`,
  that tracking must never break a run, and give the command to view the
  history (`mlflow ui --backend-store-uri training/mlruns`, run by the user).
- **Success criteria:** The guidelines state the convention and the command.

### Task 6 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest` passes, and `cd database && pytest` passes.
  - ruff clean; mypy shows only the 9 known errors.
  - No `mlruns/` directory was created by the test run.
