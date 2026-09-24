# Feature Router Experiment — Implementation Plan

## Summary

A runnable, read-only experiment script in `training/router/` that uses the
stored frame features to (1) report each feature's relationship with the
per-frame accuracy gap across all 5,000 frames, and (2) train and evaluate a
decide-first router and a cascade router, each in a linear and a
gradient-boosted variant. It compares them with always-local, always-offload
and the oracle on held-out photos, with a bootstrap confidence interval. It
then writes the results and a recommendation to `training/router/findings.md`.

## Approach & Key Decisions

- **Two stages, so the 5,000-frame pool can be used without leakage.**
  - *Stage 1 (per frame)* predicts, from frame features, the expected accuracy
    gap (offload − local) and the probability that local is at least as good.
    It trains only on the ~4,500 frames that never appear in
    `simulation_results`. No simulated photo, train or test, is ever in
    stage 1's training data. That is stricter than the spec's "test photos
    never used in training" and makes stage 1's outputs genuinely
    out-of-sample for every simulated row.
  - *Stage 2 (per simulated row)* is a classifier that combines stage 1's two
    outputs with the network and device features. It is trained on the
    existing frame-level split's training photos
    (`router.dataset.frame_level_split`, seed 42) and evaluated on its 100
    held-out photos.
- **Four routers, one per design and model family:**
  - decide-first × {linear, gradient-boosted}
  - cascade × {linear, gradient-boosted}

  "Linear" is scaled linear/logistic regression in both stages;
  "gradient-boosted" is `HistGradientBoosting*` in both, matching
  `router/baseline.py`'s families (spec requirement). The two designs differ
  in two ways:
  - **Stage-1 inputs:** decide-first uses the image statistics plus scene
    complexity; the cascade also gets the six confidence features.
  - **Stage-2 label:**
    - Decide-first uses the stored LOCAL/OFFLOAD label.
    - The cascade uses a computed ACCEPT/ESCALATE label: escalate when offload
      accuracy minus λ × (local + offload latency) beats local utility.
- **One scoring module owns every utility number.**
  - It uses the same utility formula and `DEFAULT_LAMBDA` as `router/baseline.py`
    and `router/analyze.py`, reimplemented locally as those files do, since
    that function is private.
  - Cascade rows are scored exactly as the spec says: an accepted row gets
    local accuracy and latency; an escalated row gets offload accuracy and
    local + offload latency.
  - The oracle is the existing per-row maximum of local and offload utility,
    and headroom share is (router − offload) / (oracle − offload).
  - The bootstrap resamples held-out photos (not rows) 2,000 times with a
    fixed seed and takes the 2.5/97.5 percentiles. A router is "useful" only
    if the lower bound is above 0.
- **Read-only against the real database, and existing scripts untouched.**
  The experiment only reads `simulation_results`, `simulation_runs`,
  `scene_complexity`, `model_inference` and `frame_features`, and saves no
  model files. `baseline.py`, `analyze.py` and `dataset.py` are imported,
  never modified, so their results can't change (spec AC).
- **Diagnostics use Spearman rank correlation** with the per-frame gap over
  all 5,000 frames, with p-values. This is the same measure as the quick
  check, and it is robust to the gap's many ties at zero.

## Out of Scope

- Computing or storing features (spec 01, done).
- Phone-measured latencies: all latencies are the stored desktop values.
- Changing the utility formula or λ, re-labelling rows, new simulation runs.
- Model families beyond linear and gradient-boosted; hyperparameter search.
- Saving trained router models, or deploying them to the phone or server.
- Any change to `router/baseline.py`, `router/analyze.py` or `router/dataset.py`.
- An oracle specific to the cascade. The spec asks for the existing oracle
  ceiling; the cascade's own maximum is lower, and the findings note that.

## Dependencies and Configuration

None. `scipy`, `scikit-learn` and `pandas` are already in `training/`'s
dependencies. The script reads `DATABASE_URL` from `training/.env`, like the
other router scripts. No migration, since `frame_features` already exists
(spec 01).

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/router/frame_dataset.py` | add | Load the per-frame table (features, complexity, both accuracies, gap), the set of simulated frames, and the simulated rows | Data loading as its own concern |
| `training/router/diagnostics.py` | add | Spearman correlation of each feature with the gap | Spec's feature diagnostics |
| `training/router/evaluation.py` | add | Per-row utilities, cascade scoring, baselines, oracle, headroom share, bootstrap CI and verdict | Keeps every router scored identically |
| `training/router/two_stage.py` | add | Stage-1 frame models, stage-2 row classifiers, the cascade label, and the two router designs | The router logic |
| `training/router/feature_experiment.py` | add | `main()`: load, split, train, evaluate, print the report | The runnable experiment |
| `training/router/findings.md` | edit | New section with the results and recommendation | Spec's write-up |
| `training/tests/router/test_frame_dataset.py` | add | Loader tests (ephemeral Postgres) | Data correctness |
| `training/tests/router/test_diagnostics.py` | add | Correlation tests | Diagnostic correctness |
| `training/tests/router/test_evaluation.py` | add | Hand-computed utility, cascade scoring and bootstrap tests | Evaluation correctness is what the verdict rests on |
| `training/tests/router/test_two_stage.py` | add | Leakage guard, determinism and label tests on synthetic data | Router correctness |
| `training/tests/router/test_feature_experiment.py` | add | End-to-end run on a small synthetic database, twice | Integration and determinism |

## Tasks

### Task 1 — Frame-level data loading

- **Objective:** Load the data the experiment needs from Postgres.
- **Files:** `training/router/frame_dataset.py`,
  `training/tests/router/test_frame_dataset.py`
- **Details:**
  - Provide three functions, each taking `engine` and `dataset`:
    - **The per-frame table**, one row per frame. It joins `frame_features`,
      `scene_complexity` and `model_inference` twice (LOCAL and OFFLOAD
      `model_path`) on (`dataset`, `file_name`). Columns: `file_name`,
      `scene_complexity`, the 11 feature columns, `local_accuracy`,
      `offload_accuracy`, and `gap = offload_accuracy - local_accuracy`.
      Sorted by `file_name`.
    - **The set of frame file names** that appear in `simulation_results`
      (as `frame_id`) for the dataset (via `simulation_runs.dataset`).
    - **The simulated rows:** `router.dataset.load_training_data` joined with
      the per-frame table's feature columns on `frame_id = file_name`.
  - Export two column-name lists as constants: the six image-side columns
    (`scene_complexity` plus the five image statistics) and the six
    confidence columns.
  - Use SQL via `pd.read_sql` with bound parameters, like `router/dataset.py`.
  - Tests use the `postgres_engine` fixture from `training/tests/conftest.py`
    with a few seeded frames. Cover:
    - the join, and the gap computed correctly
    - a frame with no `frame_features` row is excluded
    - dataset scoping
    - the simulated-frame set
- **Success criteria:**
  - New tests pass, and ruff and mypy are clean for the new files. (`training/`
    has 9 known mypy errors in other files; use `npt.NDArray[...]` rather
    than bare `np.ndarray` to avoid adding more.)

### Task 2 — Feature diagnostics

- **Objective:** Report each candidate feature's Spearman correlation with
  the gap.
- **Files:** `training/router/diagnostics.py`,
  `training/tests/router/test_diagnostics.py`
- **Details:**
  - Pure function: per-frame DataFrame plus feature column names → a
    DataFrame of (`feature`, `spearman_rho`, `p_value`), ordered by
    |rho| descending. Uses `scipy.stats.spearmanr`.
  - Tests:
    - a monotonic synthetic feature gives rho = ±1
    - an independent seeded-random feature gives |rho| near 0
    - the ordering is correct
- **Success criteria:** New tests pass, and ruff and mypy are clean for the new
  files.

### Task 3 — Evaluation and bootstrap

- **Objective:** Score router picks and compute baselines, the oracle,
  headroom share, and the bootstrap interval and verdict.
- **Files:** `training/router/evaluation.py`,
  `training/tests/router/test_evaluation.py`
- **Details:**
  - The utility is `accuracy - DEFAULT_LAMBDA * latency_ms / 1000`, with
    `DEFAULT_LAMBDA` from `datagen.config`.
  - Provide functions for:
    - **Per-row utilities** for local and offload.
    - **Decide-first scoring:** a LOCAL pick gets local utility, an OFFLOAD
      pick gets offload utility.
    - **Cascade scoring:** an ACCEPT row gets local utility; an ESCALATE row
      gets `offload_accuracy - λ * (local_latency_ms + offload_latency_ms)/1000`.
    - **A summary** of average utility for the router, always-local,
      always-offload and the oracle (per-row max of local and offload), plus
      headroom share.
    - **A bootstrap** that resamples unique `frame_id`s with replacement
      (2,000 resamples, seeded `numpy.random.default_rng(seed)`, default
      seed 42). It returns the 2.5/97.5 percentiles of the mean per-row
      (router − offload) difference, plus a `useful` flag that is true only
      if the lower bound is above 0.
  - Tests use tiny hand-built DataFrames with utilities worked out by hand:
    - escalated rows are charged local + offload latency
    - headroom share is correct
    - the bootstrap is identical across two calls with the same seed
    - a router that is always better by a constant gets a CI above 0
      (`useful`); one identical to offload does not
- **Success criteria:** New tests pass, and ruff and mypy are clean for the new
  files.

### Task 4 — Two-stage routers

- **Objective:** Implement stage 1, stage 2, the cascade label and both router
  designs in the linear and gradient-boosted variants.
- **Files:** `training/router/two_stage.py`,
  `training/tests/router/test_two_stage.py`
- **Details:**
  - **Stage 1** is fit on a per-frame DataFrame and a list of feature columns.
    It fits two models:
    - a regressor for `gap`
    - a classifier for `gap <= 0` (local at least as good)

    The linear variant is `StandardScaler` + `LinearRegression`/`LogisticRegression(max_iter=1000)`.
    The gradient-boosted variant is `HistGradientBoostingRegressor`/`Classifier(random_state=42)`.
    It predicts two columns: `predicted_gap` and `p_local_good_enough`.
  - **Stage 2** is a classifier of the same family on those two columns plus
    `network_bandwidth_mbps`, `network_latency_ms`, `network_packet_loss_pct`
    and `device_load_pct`.
  - **The cascade label** per row is ESCALATE if the escalated utility (offload
    accuracy, local + offload latency) is greater than local utility, and
    ACCEPT otherwise. Reuse Task 3's utility function.
  - **A router function** takes the per-frame table, the simulated rows, the
    split's train/test photo sets, the set of simulated frames, a design
    (`decide_first` | `cascade`) and a family (`linear` | `gbt`).
    1. It trains stage 1 only on frames not in the simulated set.
    2. It trains stage 2 on train-photo rows, with label = the stored `label`
       column (decide-first) or the cascade label (cascade).
    3. It returns picks for the test rows.

    Decide-first stage-1 inputs are the image-side columns; cascade inputs are
    the image-side plus confidence columns (Task 1 constants).
  - Tests use seeded synthetic frames and rows (no database):
    - **Leakage guard:** stage 1's training frames exclude every simulated
      frame (assert on the frames actually passed to fit).
    - **Leakage guard:** no test photo's rows reach stage 2.
    - Picks are identical across two runs.
    - The cascade label matches hand-computed cases.
    - All four design/family combinations run and return one pick per test
      row.
- **Success criteria:** New tests pass, and ruff and mypy are clean for the new
  files.

### Task 5 — Experiment script (integration)

- **Objective:** Wire Tasks 1–4 into `python -m router.feature_experiment`.
- **Files:** `training/router/feature_experiment.py`,
  `training/tests/router/test_feature_experiment.py`
- **Details:**
  - An orchestration function takes `engine` and `dataset` and returns a
    results structure. It:
    1. Loads the data (Task 1).
    2. Computes diagnostics over all frames (Task 2).
    3. Splits the simulated rows with `router.dataset.frame_level_split(df,
       test_size=0.2, seed=42)`.
    4. Runs the four routers (Task 4).
    5. Evaluates each with the matching scoring, summary and bootstrap
       (Task 3).
  - `main()` loads `training/.env`, runs it for `coco_val2017`, and prints:
    - the diagnostics table
    - a results table: router, avg utility, always-local, always-offload,
      oracle, headroom share, 95% CI, useful yes/no
  - The integration test seeds a small synthetic dataset in ephemeral
    Postgres: frames with features and both accuracies, some of them also
    in simulation runs and results. It runs the orchestration twice and
    asserts:
    - identical results
    - all four routers are present
    - every reported number is finite
- **Success criteria:**
  - New tests pass, and ruff and mypy are clean for the new files.
  - `python -m router.feature_experiment --help` or a plain run does not error
    on import.

### Task 6 — Run the experiment and write findings

- **Objective:** Run the experiment on the real data (read-only) and record
  the results.
- **Files:** `training/router/findings.md`
- **Details:**
  - Run `python -m router.feature_experiment` from `training/`. It only reads
    the database. Run it twice and confirm the output is identical.
  - Append a new section to `findings.md` that keeps the existing content
    unchanged. It covers:
    - the diagnostics table
    - the four routers' results against always-local, always-offload and
      the oracle, with headroom share and 95% CIs
    - which routers meet the usefulness bar
    - spec 01's feature compute times (image features 5.3 ms/frame;
      confidence features 20.5 ms/frame, almost all of it the YOLOv8n run;
      desktop CPU)
    - a note that the cascade's ceiling is below the shown oracle
    - a plain recommendation on whether to resume the parked phone/server
      work (`features/server-served-benchmark/`)
  - Report numbers exactly as printed. Don't round away a negative result.
  - Confirm `python -m router.baseline` still runs. It saves its model files
    to the gitignored `router/models/`, and its printed numbers must match
    the existing findings.
- **Success criteria:**
  - Two runs give identical output.
  - `findings.md` has the new section.
  - `router.baseline`'s printed results are unchanged from the existing
    findings.

### Task 7 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest` passes, including the existing 109 tests.
  - ruff is clean, and mypy shows no errors beyond the 9 known ones.
