# Latency-Budget Router — Implementation Plan

## Summary

A read-only experiment script in `training/router/` that evaluates two routing
policies under a hard per-request latency budget. The first is budget-only;
the second is budget plus a local-confidence cascade, using a raw score and a
learned score. It compares them with always-local, always-offload and a
within-budget oracle for budgets of 100–500 ms, on the existing held-out
photos. It reports mean and on-time accuracy, mean latency and over-budget
share, threshold sweeps, tuned operating points, and bootstrap intervals. It
saves a figure and writes the results and an updated phone/server
recommendation to `training/router/findings.md`.

## Approach & Key Decisions

- **Everything is computed per row, vectorised, from stored values.** Each
  held-out simulated row already has true local and offload latency and
  accuracy.
  - A policy returns each row's action: LOCAL, OFFLOAD or ESCALATE.
  - Outcomes follow from the action:
    - LOCAL → local latency and accuracy
    - OFFLOAD → offload latency and accuracy
    - ESCALATE → local + offload latency, offload accuracy
  - Metrics follow from the outcomes. On-time accuracy counts a result over
    the budget as 0.
  - No inference is re-run.
- **What the router is allowed to know:**
  - *Offload latency* is predicted by a `HistGradientBoostingRegressor
    (random_state=42)` on the four network and device columns, trained on
    training-photo rows only. Gradient-boosted because the simulated offload
    latency scales with 1/bandwidth, which a linear model misfits.
  - *Local latency* is the row's actual value. The cascade has already run
    locally when it decides whether to escalate, so it has observed it. The
    budget-only policy doesn't need it.
- **Confidence scores:**
  - *Raw:* the stored `mean_confidence` (0 when there are no detections, so
    those frames always escalate if the budget allows).
  - *Learned:* `p_local_good_enough` from spec 02's
    `router.two_stage.train_stage1` / `predict_stage1`. It uses family `gbt`
    and only `CONFIDENCE_COLUMNS` as inputs, and trains on frames outside
    the simulation, so it never sees a simulated photo.
  - A request keeps its local result when score ≥ threshold. The swept grid
    is 0.00–1.00 in steps of 0.05. A threshold of 0 means the cascade never
    escalates.
  - The tuned threshold per budget and score is the one with the highest
    on-time accuracy on training-photo rows, with the lowest threshold
    winning ties.
- **The oracle** knows each row's true latencies and accuracies. It takes the
  most accurate option whose latency fits the budget (local, offload, or
  local-then-offload) and falls back to LOCAL when nothing fits. Its on-time
  accuracy is therefore at least any policy's on every row.
- **Reuse, and leave spec 02's results unchanged:**
  - Reused: `frame_dataset` loaders, `dataset.frame_level_split(seed=42)` and
    `two_stage` stage 1.
  - `evaluation.py` gains one generic function: a photo-level bootstrap of
    the mean paired difference. `bootstrap_router_vs_offload` is refactored
    to call it, with the same random-number sequence, so spec 02's numbers
    come out identical (checked in Task 6).
  - Plotting lives in its own module (coding guidelines: one file, one
    concern).
  - The script only reads the database (`CLAUDE.md` "Infrastructure").

## Out of Scope

- Phone-measured or phone-scaled latencies (including an iPhone-CPU what-if).
- Changing spec 02's soft-utility routers or their results.
- Frame rate, battery, bandwidth or energy constraints; stale-result decay.
- More than two models; resolution or bitrate choices.
- New features, detector retraining, new simulation runs, schema changes.
- Deploying the router; hyperparameter search beyond the threshold grid.

## Dependencies and Configuration

- **`training/pyproject.toml`:** add `matplotlib>=3.8` to `dependencies`. It
  is already installed in the shared venv (3.11.2, pulled in by
  `ultralytics`), but it will now be imported directly. It ships its own type
  information, so no mypy override is needed.
- No DB migration, no new config. The script reads `DATABASE_URL` from
  `training/.env`.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/router/latency_policies.py` | add | Offload-latency predictor, confidence scores, both policies, oracle, per-row outcomes and metrics | The routing logic under a budget |
| `training/router/evaluation.py` | edit | Generic photo-level paired bootstrap; existing bootstrap delegates to it | One bootstrap implementation, spec 02 unchanged |
| `training/router/budget_plot.py` | add | Accuracy-versus-latency figure per budget | Plotting as its own concern |
| `training/router/budget_experiment.py` | add | Orchestration plus `main()` and printed report | The runnable experiment |
| `training/router/latency_budget_curves.png` | add | The figure (generated in Task 5) | Spec's write-up figure |
| `training/router/findings.md` | edit | New section with results and recommendation | Spec's write-up |
| `training/pyproject.toml` | edit | Add `matplotlib` | Now a direct import |
| `training/tests/router/test_latency_policies.py` | add | Hand-computed policy, outcome, oracle and metric tests | The verdict rests on this logic |
| `training/tests/router/test_evaluation.py` | edit | Tests for the generic bootstrap | New shared function |
| `training/tests/router/test_budget_plot.py` | add | The figure file gets written | Plot smoke test |
| `training/tests/router/test_budget_experiment.py` | add | End-to-end run on a small synthetic database, twice | Integration and determinism |

## Tasks

### Task 1 — Generic paired bootstrap

- **Objective:** Add a reusable photo-level bootstrap and make the existing one
  use it without changing its results.
- **Files:** `training/router/evaluation.py`,
  `training/tests/router/test_evaluation.py`
- **Details:**
  - Add `bootstrap_mean_difference(frame_ids, diff, *, n_resamples=2000,
    seed=42) -> BootstrapResult`, taking arrays or Series. It is the current
    vectorised body of `bootstrap_router_vs_offload`: per-frame sums and
    counts, one `rng.integers` draw matrix, percentiles 2.5/97.5, and
    `useful = ci_low > 0`.
  - `bootstrap_router_vs_offload` then just computes `router - offload`
    and delegates.
  - The random numbers must be drawn identically, so existing outputs are
    bit-for-bit unchanged.
  - Tests:
    - the generic function matches a hand-built case
    - it is deterministic for a given seed
    - `bootstrap_router_vs_offload` returns exactly the same `BootstrapResult`
      as the generic function called on `router - offload`
- **Success criteria:**
  - All existing `test_evaluation.py` tests still pass unchanged, and the new
    ones pass.
  - ruff clean; mypy shows no errors beyond `training/`'s 9 known ones (use
    `npt.NDArray[...]`, not bare `np.ndarray`).

### Task 2 — Policies, oracle and metrics

- **Objective:** Implement the per-row budget logic as pure, vectorised
  functions.
- **Files:** `training/router/latency_policies.py`,
  `training/tests/router/test_latency_policies.py`
- **Details:**
  - The row columns used are `local_latency_ms`, `local_accuracy`,
    `offload_latency_ms` and `offload_accuracy` (see
    `router.frame_dataset.load_simulated_rows`).
  - Functions:
    - `budget_only_actions(predicted_offload_ms, budget_ms)`: OFFLOAD where
      the prediction is ≤ budget, else LOCAL.
    - `cascade_actions(local_latency_ms, predicted_offload_ms, score,
      threshold, budget_ms)`:
      - LOCAL where score ≥ threshold
      - otherwise ESCALATE where local + predicted offload ≤ budget
      - otherwise LOCAL
    - `oracle_actions(rows, budget_ms)`: among the options whose *true*
      latency fits the budget (LOCAL, OFFLOAD, ESCALATE), pick the highest
      accuracy. Break ties by lower latency, then in the order LOCAL,
      OFFLOAD, ESCALATE. Fall back to LOCAL when nothing fits.
    - `outcomes(rows, actions) -> (latency_ms, accuracy)`: LOCAL → local;
      OFFLOAD → offload; ESCALATE → local + offload latency with offload
      accuracy.
    - `metrics(latency_ms, accuracy, budget_ms)` returns a frozen dataclass:
      `mean_accuracy`, `on_time_accuracy` (accuracy with over-budget results
      set to 0), `mean_latency_ms` and `over_budget_share`.
    - `on_time_accuracy_per_row(latency_ms, accuracy, budget_ms)` for the
      bootstrap.
  - Action labels are the strings `"LOCAL"`, `"OFFLOAD"` and `"ESCALATE"`.
  - Tests use tiny hand-built rows with the answers worked out by hand. They
    cover:
    - each branch of both policies, including a threshold of 0 never
      escalating and an escalation that doesn't fit falling back to local
    - escalation charged local + offload latency with offload accuracy
    - on-time accuracy zeroing late results
    - the oracle picking the most accurate fitting option and falling back
      when nothing fits
    - a property check on seeded random rows: the oracle's on-time
      accuracy per row is ≥ that of budget-only and of the cascade at
      several thresholds
- **Success criteria:** New tests pass; ruff and mypy are clean for the new
  files.

### Task 3 — Offload-latency predictor and confidence scores

- **Objective:** Add the offload-latency model and the two confidence scores
  to `latency_policies.py`.
- **Files:** `training/router/latency_policies.py`,
  `training/tests/router/test_latency_policies.py`
- **Details:**
  - `train_offload_latency_model(train_rows)` fits
    `HistGradientBoostingRegressor(random_state=42)` on `network_bandwidth_mbps`,
    `network_latency_ms`, `network_packet_loss_pct` and `device_load_pct`,
    with target `offload_latency_ms`. `predict_offload_latency(model,
    rows)` returns the predictions.
  - `raw_score(rows)` returns `rows["mean_confidence"]`.
  - `learned_score_model(frame_table, simulated_frame_ids)` calls
    `router.two_stage.train_stage1(frame_table, CONFIDENCE_COLUMNS, "gbt",
    simulated_frame_ids)`. `learned_score(model, rows)` returns the
    `p_local_good_enough` column from `router.two_stage.predict_stage1`.
  - Tests on seeded synthetic data (no database):
    - the latency model tracks a synthetic 1/bandwidth relationship
      (correlation of prediction with truth above 0.9)
    - the learned-score model's trained frames exclude every simulated frame
      (use `Stage1Models.trained_frame_ids`)
    - both scores return one value per row, all within [0, 1]
    - results are deterministic across two runs
- **Success criteria:** New tests pass; ruff and mypy are clean for the
  changed files.

### Task 4 — Figure

- **Objective:** Draw the accuracy-versus-latency figure.
- **Files:** `training/router/budget_plot.py`,
  `training/tests/router/test_budget_plot.py`, `training/pyproject.toml`
- **Details:**
  - Add `matplotlib>=3.8` to `training/pyproject.toml` `dependencies`.
  - One function takes plain per-budget results data and an output path,
    and writes a PNG using the non-interactive `Agg` backend.
  - The figure has one panel per budget, with x = mean latency (ms) and
    y = on-time accuracy. Each panel shows:
    - the threshold-sweep curve for each confidence score
    - the tuned operating points
    - single markers for budget-only, always-local, always-offload and the
      oracle
    - a vertical line at the budget
  - Test: given small synthetic results, the function writes a non-empty
    PNG to `tmp_path`.
- **Success criteria:** New test passes; ruff and mypy are clean for the new
  file.

### Task 5 — Experiment script (integration)

- **Objective:** Wire everything into `python -m router.budget_experiment`.
- **Files:** `training/router/budget_experiment.py`,
  `training/tests/router/test_budget_experiment.py`
- **Details:**
  - An orchestration function `(engine, dataset, budgets=(100, 150, 200, 300,
    400, 500))` returns a results dataclass. It:
    1. Loads data with `router.frame_dataset`.
    2. Gets the train/test photo sets from `router.dataset.frame_level_split
       (rows, test_size=0.2, seed=42)`, then derives the train and test rows
       with `rows[rows["frame_id"].isin(ids)].reset_index(drop=True)`.
       (Spec 02 learned that the split's own DataFrames come back in a
       different row order.)
    3. Trains the latency model on the train rows and the learned-score
       model on non-simulated frames.
    4. For each budget:
       - computes the metrics for always-local, always-offload, budget-only
         and the oracle on the test rows
       - for each score, sweeps thresholds on the test rows and picks the
         tuned threshold on the train rows, then reports it on the test rows
       - bootstraps the tuned cascade's on-time accuracy per row minus
         budget-only's, and minus always-local's, with
         `bootstrap_mean_difference` over the test rows' `frame_id`
       - sets the usefulness verdict: the interval against budget-only is
         entirely above 0
  - `main()` loads `training/.env`, runs it for `coco_val2017`, prints a
    per-budget table and the verdicts, and writes the figure to
    `training/router/latency_budget_curves.png` (with `--figure` to override
    the path).
  - The integration test seeds a small synthetic dataset in ephemeral
    Postgres, reusing the approach of
    `training/tests/router/test_feature_experiment.py`: frames outside the
    simulation, and simulated rows with varied latencies so that both
    policies branch. It runs the orchestration twice and asserts:
    - identical results
    - all six budgets and all policies present
    - every number finite
    - the oracle's on-time accuracy ≥ every policy's at every budget
- **Success criteria:**
  - New tests pass; ruff and mypy are clean for the new files.
  - `python -m router.budget_experiment --help` works.

### Task 6 — Run the experiment and write findings

- **Objective:** Run on the real data (read-only) and record the results.
- **Files:** `training/router/findings.md`,
  `training/router/latency_budget_curves.png`
- **Details:**
  - From `training/`, run `python -m router.budget_experiment` twice, and
    confirm the printed output is identical.
  - Run `python -m router.feature_experiment` and confirm its printed results
    match spec 02's section in `findings.md` exactly, so the bootstrap
    refactor changed nothing.
  - Append a section to `findings.md`, keeping existing content unchanged.
    It covers:
    - why the framing changed (spec 02's result under the soft utility)
    - the per-budget table: mean and on-time accuracy, mean latency and
      over-budget share for every policy, baseline and the oracle, plus the
      tuned thresholds
    - the bootstrap intervals and usefulness verdict per budget
    - the figure, embedded by relative path
    - a related-work paragraph citing the eight papers listed in the spec
    - an updated recommendation on the parked phone/server work
      (`features/server-served-benchmark/`) that explicitly supersedes spec
      02's recommendation
  - Report numbers exactly as printed.
- **Success criteria:**
  - Two runs give identical output.
  - Spec 02's experiment output is unchanged.
  - `findings.md` has the new section and the PNG exists.

### Task 7 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest` passes (including the existing 136 tests).
  - ruff is clean, and mypy shows no errors beyond the 9 known ones.
