# Review — MLflow Experiment Tracking

## Verdict

The feature works and the recorded history is real: I read `training/mlruns/`
back with `MlflowClient` and found two runs (`train-small`, `train-large`) with
every planned param, 30 and 15 epochs of `train_loss`/`val_accuracy`,
`best_val_accuracy`/`best_epoch`, a `git_commit` tag, no artifacts, and the
evaluation's `test_accuracy`, `mean_latency_ms` and `eval_device` on the *same*
run rather than a standalone one. Both sidecars match their run ids, accuracies
reproduce the recorded 77.98% / 86.99% to four decimals, `training/` is 215
passed with ruff clean and mypy at the 9 known pre-existing errors, and the test
run added nothing to the store (50 files before, 50 after).

I also exercised the failure paths directly rather than trusting the tests: a
failure at setup, mid-setup after `start_run` succeeded, at teardown, and an
exception raised inside the caller's body. Tracking never broke the caller in
any of them, warnings were printed, and the body's exception propagated with the
run marked `FAILED`. On the "placeholder" subagent reports — I found nothing
half-finished in Tasks 2 or 4; the callback, the sidecar and the retrain are all
present and verifiable.

Two blockers, both narrow and both one-line-ish fixes. B1 is the one that
matters: an interrupted retrain leaves a stale sidecar pointing at the *previous*
run while `fit` has already overwritten the checkpoint, so the next evaluation
attaches the new model's numbers to the old run — silent corruption of exactly
the history this feature exists to protect. B2 is that the new sidecars are not
gitignored, so `git status` now permanently reports `?? training/models/`, which
it did not before this branch.

## Acceptance Criteria

- **Every training run is recorded with settings, per-epoch progress, outcome
  and code version** — **MET**. `training/birds/train.py:152-192` logs the eight
  params, per-epoch metrics through the callback (`train.py:163-172`,
  `train.py:94-97`) and `best_val_accuracy`/`best_epoch`
  (`train.py:186-192`); `tracking.py:75-77` tags `git_commit`. Verified in the
  store: `train-small` has 30 `train_loss` and 30 `val_accuracy` points,
  `train-large` 15 each, `git_commit=92afd8b` on both.
- **Evaluation results recorded against the run that produced the model** —
  **MET** for the normal path. `evaluate.py:83-92` resumes the sidecar's run id
  and logs `test_accuracy`/`mean_latency_ms` plus the `eval_device` param; the
  store shows all three on the two training runs and no `evaluate-*` run exists.
  See B1 for the interrupted-retrain case where this can attach to the wrong run.
- **Run history stored locally in a git-ignored folder; nothing uploaded;
  nothing needs to be running** — **MET**. `config.py:12` resolves
  `MLRUNS_DIR` from `__file__`; `.gitignore:29` ignores `training/mlruns/`;
  `git status` does not report it. `tracking.py` only ever talks to a file store.
- **Viewing history is the user's own step and is documented** — **MET**.
  No code starts a server; `.claude/coding-guidelines.md:162-165` gives the
  command with its `MLFLOW_ALLOW_FILE_STORE=true` prefix and says the user runs
  it. (See S5 on where the text was placed.)
- **No checkpoints in the run history** — **MET**. Nothing calls
  `log_artifact`; `list_artifacts` returns `[]` for both runs and the whole
  store is 256K.
- **Recording failure still finishes, still prints, and is visible** —
  **MET**, verified empirically on four distinct failure points. Setup failure
  and teardown failure each print `WARNING: MLflow tracking ... failed (...);
  continuing.` and the body runs; a body exception propagates unchanged and the
  run is left `FAILED` with its metrics intact.
- **Test suite passes writing no run history** — **MET**. The autouse fixture
  in `training/tests/conftest.py:23-27` sets `BIRDS_TRACKING=off`; the only
  tests that turn it back on (`test_tracking.py`) repoint
  `tracking.MLRUNS_DIR` at `tmp_path`. I counted the store's files before and
  after a full run: unchanged, and no stray `mlruns/` appeared anywhere.
- **Both models retrained under tracking, within a point of 77.98% / 86.99%** —
  **MET**. Store holds 0.7797721781 and 0.8698653780. Latency moved
  (5.71→7.40 ms, 6.70→7.73 ms), which the report calls out as load noise rather
  than accepting silently.
- **Guidelines state runs are tracked; full suite passes, ruff clean, no new
  type errors** — **PARTIAL**. The suite, ruff and mypy are all clean (215
  passed, `All checks passed!`, 9 pre-existing mypy errors in `coco/`, none in
  `birds/`), and all eight changed files are `ruff format` clean. The
  convention is written, but under the `## Data-gen / router training code
  (training/coco/datagen/, training/coco/router/)` heading, which does not
  cover `training/birds/` — see S5.
- **Run history written under a folder `git status` never reports** —
  **PARTIAL**. `training/mlruns/` is ignored, but the new
  `<checkpoint>.run.json` sidecars are not, and `git status` now reports
  `?? training/models/` where it reported nothing before. See B2.

## Scope

The diff matches the plan's Files Changed table exactly — same eleven paths,
same actions, nothing extra. Two documented deviations, both benign: the plan
allowed a mypy `ignore_missing_imports` override for `mlflow` and it turned out
not to be needed (mlflow ships types, mypy is clean without it), and
`run_sidecar_path` landed in `train.py` rather than `tracking.py` (see S6).
Nothing outside `training/birds/`, `training/tests/`, `training/pyproject.toml`,
`.gitignore` and `.claude/coding-guidelines.md` was touched. `training/coco/`
is untouched, as the spec's Out of Scope requires.

## Blockers

#### B1 — An interrupted retrain leaves a stale sidecar pointing at the previous run

- **File:** `training/birds/train.py:192`
- **Issue:** `_record_run_id` is the *last* statement inside the `with
  tracking.run(...)` block, so it only executes when `fit` returns normally.
  `fit` overwrites `<checkpoint>.pt` as soon as an epoch beats the running best
  (`train.py:99-103`), so a Ctrl-C or an OOM part-way through a retrain leaves a
  **new** checkpoint on disk beside a sidecar still naming the **previous**
  run. The next `evaluate` run then silently logs that model's `test_accuracy`,
  `mean_latency_ms` and `eval_device` into the earlier, unrelated run, with no
  note printed because `_training_run_id` did find a usable id. The same window
  exists with tracking off: the stale-sidecar removal at `train.py:115-117` is
  also inside the block, so an interrupted untracked retrain leaves the old id
  in place too. This is the one failure mode the sidecar design was meant to
  rule out, and it corrupts recorded history silently rather than loudly.
- **Fix:** Move `_record_run_id(checkpoint_path, handle.id)` to immediately
  after entering the `with` block, before the `fit` call — the run id is already
  known there, and it makes the sidecar match whichever run is currently
  entitled to write that checkpoint at every instant, crash or not.
- **Decision:** — _(pending)_

#### B2 — The run-id sidecars are not gitignored, so `git status` now reports `training/models/`

- **File:** `.gitignore:29`
- **Issue:** `.gitignore` ignores `*.pt` and the new `training/mlruns/`, but not
  `*.pt.run.json`. Before this branch `training/models/` held only ignored
  files and git said nothing about it; it now reports `?? training/models/` on
  every status, and a `git add .` would commit a machine-local MLflow run id.
  Verified: `git status --porcelain` prints `?? training/models/`, and the
  directory's only non-`.pt` contents are `small.pt.run.json` and
  `large.pt.run.json`.
- **Fix:** Add `*.pt.run.json` (or `training/models/`) to `.gitignore`
  alongside the `training/mlruns/` entry, with the same one-line rationale
  comment.
- **Decision:** — _(pending)_

## Suggestions

#### S1 — Evaluation overwrites the training run's `git_commit` tag

- **File:** `training/birds/tracking.py:75-77`
- **Issue:** `run()` unconditionally tags `git_commit` with the commit at call
  time, including when it is resuming an existing run. Because `evaluate.main`
  resumes the training run, evaluating a model at a later commit silently
  rewrites the tag that is supposed to say which code version *produced* the
  model — the spec's "which version of the code produced it". It happens to be
  correct in the current store only because the retrain and the evaluation ran
  at the same commit (`92afd8b`).
- **Fix:** Only set `git_commit` when `run_id is None` (i.e. when the run is
  actually being created); if the evaluating commit is worth keeping, record it
  under a separate key such as `eval_git_commit`.
- **Decision:** — _(pending)_

#### S2 — Resuming the run during evaluation rewrites its end time, so recorded training duration is wrong

- **File:** `training/birds/evaluate.py:88-90`
- **Issue:** `_log_results` resumes the training run through the full `run()`
  context manager, whose `finally` calls `mlflow.end_run` — which resets the
  run's `end_time` to the evaluation's clock. Measured in the store:
  `train-small` reports a 9.60-minute duration although its last epoch metric
  landed 2.94 minutes after start. Anyone comparing runs on duration gets a
  number that is mostly evaluation and idle time.
- **Fix:** Share the fix with S4 — log evaluation results with
  `MlflowClient().log_metric(run_id, ...)` / `log_param(run_id, ...)` instead of
  resuming the fluent run, e.g. a `tracking.log_to_run(run_id, metrics, params)`
  helper. That leaves the finished run's timing and status untouched.
- **Decision:** — _(pending)_

#### S3 — A teardown failure leaves the run globally active and silently disables tracking for the rest of the process

- **File:** `training/birds/tracking.py:89-94`
- **Issue:** When `mlflow.end_run` raises, the warning is printed but the run
  stays on MLflow's fluent active-run stack. Every subsequent `run()` in the
  same process then fails at `start_run` with "Run with UUID ... is already
  active", yields the no-op handle, and records nothing. Verified: after a
  forced teardown failure, the next `run()` returned `handle.id is None`. In
  `evaluate.main` that means a teardown failure on `small` costs `large`'s
  results too, and unlike the standalone-run fallback nothing is printed in the
  results table to say so.
- **Fix:** In the teardown `except`, drop the run from the active stack before
  continuing (a second best-effort `mlflow.end_run()`, or clearing
  `mlflow.tracking.fluent._active_run_stack`), so one bad teardown costs one
  run rather than the whole process.
- **Decision:** — _(pending)_

#### S4 — `log_metrics` ignores `handle.id` and writes to whichever run is globally active

- **File:** `training/birds/tracking.py:97-105`
- **Issue:** The handle is checked only for `None`; the actual write is
  `mlflow.log_metrics(...)`, which targets MLflow's global active run, not
  `handle.id`. Inside the `with` block those coincide, so today's callers are
  correct — but the signature promises a targeted write that the body does not
  deliver. A `log_metrics(handle, ...)` called after the block, or from a nested
  context, would write to the wrong run or start a fresh one, and nothing in the
  tests would catch it.
- **Fix:** Write through `MlflowClient(tracking_uri=...).log_metric(handle.id,
  key, value, step=step)` so the handle genuinely selects its run. This also
  gives S2 its fix for free.
- **Decision:** — _(pending)_

#### S5 — The convention was documented under the coco/router heading, which does not cover `training/birds/`

- **File:** `.claude/coding-guidelines.md:154-165`
- **Issue:** The three new bullets sit under `## Data-gen / router training code
  (training/coco/datagen/, training/coco/router/)` — a section explicitly scoped
  to the two coco directories, and the one area the spec puts out of scope. The
  file has no birds section at all, so the standing convention for
  `training/birds/` is filed under a heading that says it does not apply there,
  and "new training code keeps that" reads as applying to coco code that is
  meant to stay untracked.
- **Fix:** Either add a `## Birds training code (training/birds/)` heading above
  the new bullets, or widen the existing heading to `## Training code
  (training/)` and keep the per-area scoping in the bullets themselves.
- **Decision:** — _(pending)_

#### S6 — `evaluate.py` imports `run_sidecar_path` from `birds.train`

- **File:** `training/birds/evaluate.py:17`
- **Issue:** Evaluation now imports the training entry-point module purely to
  learn a filename convention, pulling `argparse`, the optimizers and the
  scheduler into every evaluation run. It also cuts against the guidelines'
  "one file, one concern" for training scripts. There is no cycle today only
  because `train` happens never to import `evaluate`.
- **Fix:** Move `run_sidecar_path` into `birds/tracking.py` (it is tracking
  metadata, not training logic) and have both entry points import it from
  there.
- **Decision:** — _(pending)_

#### S7 — MLflow moved shared dependencies in the venv that `server/` also uses

- **File:** `training/pyproject.toml:22`
- **Issue:** Installing `mlflow` pulled ~50 packages into the shared venv and
  moved `fastapi` to 0.141.1 and `pydantic` to 2.13.5 for `server/` as a side
  effect of a training-only dependency. The report's "server clean" is weaker
  than it sounds: `server/` has no `tests/` directory at all, so nothing
  behavioural was actually exercised there — only training and database were.
  Pip also flagged two pre-existing conflicts (`onnx`/protobuf,
  `xdsl`/typing-extensions).
- **Fix:** At this prototype's bar, a separate venv is not worth it. Cheap
  mitigation: record the working resolved versions somewhere reproducible (a
  `requirements.lock` or upper bounds on `fastapi`/`pydantic` in
  `server/pyproject.toml`) so a future `pip install` that moves them again is
  attributable rather than mysterious.
- **Decision:** — _(pending)_

## Nitpicks

#### N1 — `_git_commit` uses `config.MLRUNS_DIR.parent` as its working directory

- **File:** `training/birds/tracking.py:44`
- **Issue:** "The parent of the mlruns directory" is an indirect way to spell
  "the training directory", and it deliberately reads `config.MLRUNS_DIR`
  rather than the module-level `MLRUNS_DIR` that tests repoint — a discrepancy
  a reader has to stop and reason about, with no comment saying why.
- **Fix:** Use an explicit constant (e.g. `config.TRAINING_DIR` or
  `Path(__file__).resolve().parents[1]`), or add the one-line why.
- **Decision:** — _(pending)_

#### N2 — The `git_commit` tag duplicates MLflow's own `mlflow.source.git.commit`

- **File:** `training/birds/tracking.py:75-77`
- **Issue:** MLflow already tags every run with `mlflow.source.git.commit`,
  `mlflow.source.git.branch` and the repo URL. Verified in the store: the custom
  `git_commit=92afd8b` is the short form of the automatic
  `mlflow.source.git.commit=92afd8b6f19...` on both runs. The hand-rolled
  `subprocess` call, its error handling and its test surface all buy a prefix of
  something already recorded.
- **Fix:** Drop `_git_commit` and the tag, and note in the guidelines that
  MLflow records the commit itself. (Skip this if S1 is taken and the explicit
  tag is kept for its resume semantics.)
- **Decision:** — _(pending)_

#### N3 — A param conflict on resume is reported as "setup failed"

- **File:** `training/birds/tracking.py:79-83`
- **Issue:** The single `except` around the whole setup block labels every
  failure "setup". Re-evaluating a model on a different device makes
  `log_params` reject the changed `eval_device`, and the user sees `WARNING:
  MLflow tracking setup failed (MlflowException: Changing param values is not
  allowed ...)` even though the run started fine and the metrics did land. I
  reproduced this; it is confusing rather than harmful.
- **Fix:** Either name the failing step in the message (split the `except`
  around `start_run` from the one around `log_params`/`set_tag`), or log
  `eval_device` as a tag, which MLflow allows to change.
- **Decision:** — _(pending)_

#### N4 — `mlflow>=2.14` is looser than what was actually verified

- **File:** `training/pyproject.toml:22`
- **Issue:** Only 3.16.1 was installed and exercised, and both the
  `MLFLOW_ALLOW_FILE_STORE` handling in `tracking.py:68` and the documented UI
  command are 3.x-specific. A fresh resolve landing on 2.x would still work but
  would not match the documentation.
- **Fix:** Tighten to `mlflow>=3.16` to match what was tested.
- **Decision:** — _(pending)_

## Tests

Added: `training/tests/birds/test_tracking.py` (5), 4 new in `test_train.py`,
7 new/rewritten in `test_evaluate.py`. 215 pass in 12.96s, none skipped, and the
run leaves the store byte-identical.

The coverage is well chosen for the prototype bar. `test_tracking.py` drives a
real MLflow file store in `tmp_path` and reads back through `MlflowClient`
rather than asserting on mocks, `test_body_exception_propagates` also asserts
`mlflow.active_run() is None` afterwards (the right invariant), and the
six-case parametrisation over malformed sidecars — absent, unparseable, `{}`,
`{"run_id": 5}`, `[]`, `{"run_id": ""}` — matches `_training_run_id`'s guards
exactly.

Gaps, in rough order of value:

- **No test for the interrupted-retrain sidecar (B1).** Nothing covers `fit`
  raising with a sidecar already on disk. A test that makes `fake_fit` raise and
  asserts the sidecar no longer names the previous run would have caught it and
  would pin the fix.
- **No teardown-failure test (S3).** `test_mlflow_failure_on_start_warns...`
  patches `start_run` only; the plan called for setup *or teardown*. The
  teardown path — and the fact that it poisons the next run in the process — is
  untested.
- **`log_metrics` is never tested against a wrong active run (S4)**, so the
  handle/global-run mismatch is invisible to the suite.
- **The evaluate tests assert on calls, not stored history**, which the plan
  explicitly sanctioned and is the right trade here — but it means the
  `evaluate` → real-MLflow seam is only ever verified by the manual retrain,
  not by CI.
- `_patch_main_for_tracking`'s `log_metrics` stub is
  `lambda handle, metrics: ...`, which would raise if a caller ever passed
  `step=`. Harmless today, brittle if evaluation ever logs stepped metrics.

Nothing fragile or order-dependent; the `BIRDS_TRACKING` fixture is autouse and
`monkeypatch`-based, so per-test opt-in works and nothing leaks between tests.

## Recommended Decisions

- **B1** — Accept — moving one line above `fit` closes a silent wrong-run
  attachment on the most likely failure (an interrupted retrain), with no
  behaviour change on the happy path.
- **B2** — Accept — one `.gitignore` line, and without it every `git status`
  from now on reports `training/models/` and invites an accidental commit.
- **S1** — Accept — a two-word condition, and without it the training run's
  recorded code version silently becomes the evaluation's.
- **S2** — Accept — take it together with S4; the client-based helper fixes both
  and removes the surprising side effect of evaluation rewriting a finished
  run's duration.
- **S3** — Accept — cheap, and it stops one teardown hiccup from silently
  swallowing every later run in the process.
- **S4** — Accept — makes the handle mean what its signature says and is the
  shared fix for S2; do these two as one change.
- **S5** — Accept — a heading move; leaving the birds convention filed under the
  coco section is exactly the kind of thing that gets missed by the next reader.
- **S6** — Accept — a two-line move that removes an import of the training entry
  point from evaluation and puts the sidecar convention beside the tracking it
  belongs to.
- **S7** — Decline — the coupling is real but already documented in
  `learnings.md` and the report, all three packages were re-checked, and
  splitting the venv or adding a lockfile is scope this semester project did not
  ask for. Revisit if `server/` ever breaks unexplainedly.
- **N1** — Accept — one-line clarity fix in the module that is meant to be the
  single obvious place tracking lives.
- **N2** — Decline — the custom tag is redundant but harmless, it is what the
  plan specified, and if S1 is taken it gains a purpose the automatic tag does
  not have (a commit that is not overwritten on resume).
- **N3** — Accept — take the tag half of the fix (`set_tag("eval_device", ...)`
  instead of `log_params`), which removes the spurious warning outright rather
  than just renaming it.
- **N4** — Accept — one character of edit to make the declared dependency match
  the only version anyone has run.
