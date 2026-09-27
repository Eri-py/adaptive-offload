# Learnings — latency-budget-router

## Task 1 — Generic paired bootstrap

- `bootstrap_router_vs_offload`'s vectorised body ports to the generic
  `bootstrap_mean_difference(frame_ids, diff, ...)` with no behavior change:
  it takes `frame_ids`/`diff` as `pd.Series | npt.NDArray[...]` and calls
  `np.asarray(...)` on each before `np.unique`/`np.add.at`. `np.asarray` on
  a `pd.Series` and `.to_numpy()` (what the original code called) produce
  the same underlying values array, so the draw matrix and resample means
  are bit-for-bit identical — confirmed both by the new
  `bootstrap_router_vs_offload`-matches-generic test and by an end-to-end
  before/after diff of `python -m router.feature_experiment`'s stdout.
- To get a real "before" comparison without `git stash` (blocked in this
  environment, per spec 02's learnings) or a `git worktree` (doesn't work
  here for a package that's `pip install -e`d into the shared venv — the
  editable-install finder (`.venv/lib/python3.12/site-packages
  /__editable___training_0_1_0_finder.py`) hardcodes the **absolute path**
  to `training/router` inside the main worktree, so a second worktree's
  copy of the file is never actually imported, silently making a
  worktree-based before/after comparison compare the same file to itself):
  saved the edited `evaluation.py` aside, checked out `git show
  HEAD:training/router/evaluation.py` over the real file in place, ran the
  experiment script to capture "before" output, then restored the edited
  file from the saved copy and re-ran to capture "after". `diff`'d the two
  captures directly instead of trusting a worktree run.
- `bootstrap_mean_difference`'s "hand-built case" test uses a **constant**
  per-row diff (e.g. every row's diff is exactly 0.1) rather than trying to
  hand-derive percentiles of a resampling distribution: when every row's
  diff is the same constant, every possible resample's mean is that exact
  constant regardless of which frames get drawn or how many times, so
  `ci_low == ci_high == 0.1` exactly — a real assertion on an exact number,
  not a property check, without needing to reimplement the resampling math
  in the test to predict it.

## Task 2 — Policies, oracle and metrics

- Action arrays need an explicit fixed-width string dtype up front, sized to
  the *longest* label ("ESCALATE", 8 chars). `np.full(shape, "LOCAL",
  dtype=np.str_)` infers a `<U5` dtype from the fill value alone, so a later
  `actions[mask] = "ESCALATE"` silently truncates to `"ESCALAT"` — the same
  gotcha spec 02's learnings flagged for `np.full`. Fixed by a module-level
  `_ACTION_DTYPE = "<U8"` constant used in every `np.full(..., dtype=...)`
  call that builds an action array.
- `oracle_actions`' three-way tie-break (highest accuracy, then lowest
  latency, then LOCAL/OFFLOAD/ESCALATE order) vectorises as a single pass
  over the 3 options in that exact order, tracking a running
  `best_accuracy`/`best_latency` per row and overwriting only on a *strict*
  improvement (`accuracy > best` or `accuracy == best and latency < best`).
  Because ties never trigger the overwrite, the first-seen option in
  LOCAL/OFFLOAD/ESCALATE order naturally wins every remaining tie — no
  separate order tie-break pass needed. The "nothing fits" fallback is free
  too: `actions` starts pre-filled with `LOCAL` and a row where no option's
  `fits` mask is true never triggers an overwrite, so it keeps that default.
- The property test (oracle's per-row on-time accuracy dominates
  budget-only's and the cascade's at several thresholds) holds by
  construction, not just empirically: the oracle picks the best-accuracy
  option among the row's *true*-latency fits, so any other policy's actual
  chosen action either also fits (accuracy <= oracle's best-fitting
  accuracy) or misses the budget (on-time accuracy 0 <= oracle's, which is
  never negative). Worth noting in case a future change makes this look
  like it needs a looser tolerance — it doesn't; the seeded random-row test
  uses a tiny `1e-9` slack only for float comparison, not because the
  property is approximate.

## Task 3 — Offload-latency predictor and confidence scores

- The offload-latency `HistGradientBoostingRegressor` is trained on
  condition columns only (`network_bandwidth_mbps`, `network_latency_ms`,
  `network_packet_loss_pct`, `device_load_pct`) — no per-photo stage-1
  output ever reaches it, unlike stage 2's GBT. That's *why* spec 02's S1
  photo-memorisation fix (`STAGE2_MIN_LEAF_PHOTOS` /
  `min_samples_leaf`) doesn't apply here: memorisation there came from a
  leaf splitting on `predicted_gap`/`p_local_good_enough`, which are
  constant per photo; there's no such column in this model's feature set
  for a leaf to isolate a single training photo with, so an unconstrained
  `HistGradientBoostingRegressor(random_state=42)` (matching the plan's
  literal spec) is correct as-is. Documented this reasoning directly in
  `train_offload_latency_model`'s docstring so it isn't mistaken for an
  oversight in a later review pass.
- `learned_score_model`/`learned_score` are thin wrappers around
  `router.two_stage.train_stage1`/`predict_stage1` with
  `CONFIDENCE_COLUMNS` and family `"gbt"` fixed — no new leakage logic to
  write or test beyond what spec 02 already covers; the test here only
  checks the wrapper wires `simulated_frame_ids` through to
  `Stage1Models.trained_frame_ids` correctly, reusing spec 02's
  alternating-gap-sign fixture trick (`test_two_stage.py`'s learnings) so
  any contiguous train/simulated split still has both classes for the
  `gap <= 0` classifier.
- `predict_offload_latency`/`raw_score`/`learned_score` all return plain
  `npt.NDArray[np.float64]` (via `np.asarray(..., dtype=np.float64)`)
  rather than a `pd.Series`, per the task — this lets Task 2's
  `budget_only_actions`/`cascade_actions` (which already accept
  `pd.Series | npt.NDArray[np.float64]`) take these outputs directly
  without a caller-side conversion.
- Correlation test for the latency model: generated `offload_latency_ms =
  1000/bandwidth + N(0, 2)` on `network_bandwidth_mbps ~ Uniform(1, 100)`,
  with the other three condition columns as unrelated noise, fit on 400
  synthetic rows and predicted on 150 held-out synthetic rows (not the
  training rows — a real holdout, not a memorisation check) — comfortably
  clears the required `> 0.9` `np.corrcoef` threshold with
  `HistGradientBoostingRegressor`'s defaults, no hyperparameter tuning
  needed for a two-variable synthetic relationship this smooth.

## Task 4 — Figure

- `budget_plot.py` takes only plain dataclasses (`BudgetPoint` = (mean
  latency ms, on-time accuracy); `BudgetPanelData` = budget_ms +
  `sweep_curves`/`tuned_points` keyed by score name + `single_points` keyed
  by "budget-only"/"always-local"/"always-offload"/"oracle") — no
  `RouterMetrics`, no DB, so the plotting logic is testable with hand-built
  synthetic data and stays decoupled per the coding guidelines' "one file,
  one concern." Task 5 will build these from `latency_policies.RouterMetrics`.
- Categorical color assignment follows the "fixed order, never cycled" rule
  (repo has no design-system palette of its own, so this used the
  Okabe-Ito colorblind-safe set): `raw`/`learned` get fixed hex colors from
  a dict; any other score name gets a deterministic fallback color keyed by
  its sorted position among the unknown names, so a color is never
  reassigned between panels or across a rerun. Baselines/oracle get fixed
  (marker, color) pairs the same way, not `plt`'s default cycler.
  `matplotlib.use("Agg")` must be called before `import matplotlib.pyplot`,
  which ruff's E402 flags — needs `# noqa: E402` on the pyplot import (and
  on the `matplotlib.axes`/`matplotlib.artist` imports that follow it).
- **Legend clipping gotcha:** `fig.legend(..., bbox_to_anchor=(0.5, -0.04))`
  placed the shared legend below the axes, and with enough handles (9: two
  sweeps, two tuned points, four single markers, the budget line) it wraps
  to 2 rows. With only `fig.tight_layout(rect=(0, 0.1, 1, 1))` reserving
  bottom margin and a plain `fig.savefig(...)`, the legend's second row
  silently got clipped off the saved PNG — entries just vanished from the
  file with no error, only visible by actually opening the rendered image
  (mypy/ruff/pytest all stayed green throughout). Fixed by rendering once
  and visually inspecting the output, then adding `bbox_inches="tight"` to
  `fig.savefig(...)`, which expands the saved canvas to include whatever
  the legend actually needs regardless of the `tight_layout` rect guess —
  cheaper and more robust than hand-tuning the rect/ncol until it happens
  to fit. Worth remembering for Task 6's real 6-budget figure: always open
  the rendered PNG once rather than trusting a passing smoke test, since a
  "file is non-empty" test can't catch a legend/label silently clipped at
  the canvas edge.

## Task 5 — Experiment script (integration)

- Kept `BudgetResult`/`BudgetExperimentResult`/`ScoreBudgetResult` free of
  any `pd.DataFrame` field (they hold only `RouterMetrics`/`BootstrapResult`
  dataclasses, floats, strings and plain lists/dicts of those) specifically
  so the integration test's `assert first == second` works with plain `==`
  — `feature_experiment.ExperimentResult` has to avoid `==` on its
  `diagnostics: pd.DataFrame` field for the "truth value of a DataFrame is
  ambiguous" reason spec 02's learnings flag; not storing a DataFrame at all
  sidesteps that rather than writing a custom comparison.
- The threshold sweep is computed on **test** rows (`_threshold_sweep`, used
  for the reported curve and the figure), but the tuned threshold is picked
  by rerunning the same per-threshold cascade+metrics computation on
  **train** rows (`_tuned_threshold`) — two structurally identical loops
  over `THRESHOLDS`, deliberately not shared into one helper, because they
  read from different `_SplitContext`s (train vs test) and one returns the
  full sweep while the other reduces to a single best threshold. Trying to
  unify them would need a callback or a return-type union that's less clear
  than just having two short loops.
- `_tuned_threshold`'s tie-break (lowest threshold wins) reuses
  `oracle_actions`' pattern from Task 2: only a **strict** `>` improvement
  overwrites `best_on_time`, so among equal on-time accuracies the first
  (lowest, since `THRESHOLDS` is ascending) threshold tried is the one that
  survives — no separate tie-break pass needed, same reasoning as the
  oracle's LOCAL/OFFLOAD/ESCALATE order tie-break.
- Verified the printed report end-to-end against the integration test's own
  synthetic Postgres fixture (not the real database — that's Task 6) by
  calling `run_experiment` + the module's `_print_report` directly in a
  throwaway script. Two things worth flagging for reading Task 6's real
  output, so they aren't mistaken for a bug:
  - A confidence score's tuned cascade can print **identical** numbers to
    always-local at a budget (e.g. `cascade-raw` == `always-local` exactly
    in the 100-300ms rows of the synthetic run). This is expected: when the
    tuned threshold ends up high enough (or the budget tight enough) that no
    row's escalation both falls below threshold *and* fits, every cascade
    action is LOCAL, which is bitwise the same policy as always-local on
    that data.
  - When that happens, `vs_always_local`'s bootstrap CI is exactly
    `[0.0000, 0.0000]` — the per-row diff is the constant 0 for every row
    (same mechanism as Task 1's "hand-built case" bootstrap test), not a
    sign of a broken bootstrap.
- The synthetic fixture needed to diverge from `test_feature_experiment.py`'s
  in one deliberate way: that fixture uses a **constant**
  `local_latency_ms`/`offload_latency_ms` per row (it never exercises
  latency-based routing), which would make every budget's routing decision
  degenerate to the same action here. This test instead draws
  `local_latency_ms ~ Uniform(20, 400)` per row and derives
  `offload_latency_ms` from a real `1000/bandwidth_mbps + noise`
  relationship (`bandwidth ~ Uniform(1, 100)`, so offload latency roughly
  spans 10-1000ms) — both ranges straddle the whole 100-500ms budget sweep,
  so budget-only and the cascade both genuinely branch at every budget
  rather than only exercising one action's code path.

## Task 6 — Run the experiment and write findings

- On the real database, the tuned cascade (both raw and learned score)
  turned out **not useful at any of the six budgets** — its 95% CI against
  budget-only has a strictly negative upper bound at every single budget
  (worst [-0.0380, -0.0192] at 400ms, best [-0.0220, -0.0129] at 150ms),
  not a tie or a near-miss. Worth remembering this is a *stronger* negative
  than spec 02's soft-utility result (several of those were ties), even
  though this experiment deliberately reframed the objective to give the
  cascade its best shot — the reframing didn't rescue it, which is itself
  informative and shouldn't be undersold in the write-up.
- Raw and learned confidence scores are functionally interchangeable here:
  their tuned metrics agree to 3-4 decimal places at every budget despite
  choosing different tuned thresholds and the learned score using six
  features against the raw score's one. The bottleneck isn't which frames
  get flagged for escalation, it's a structural cost (see next point), so
  no amount of better scoring input fixes it.
- The reason the cascade loses to budget-only everywhere is mechanical,
  not a modeling gap: the cascade always runs locally first, so an
  escalated request is charged local + offload latency, while budget-only
  decides up front and pays only offload latency when it offloads. This is
  the exact same mechanism spec 02's `casc ceiling` section already named
  ("escalating adds the local pass's latency ... which a cascade router
  can never avoid paying once it decides to escalate") — worth cross-
  referencing rather than re-deriving, since it's the same structural fact
  showing up under a second, unrelated objective.
- Two full `python -m router.budget_experiment` runs on the real database
  were not just numerically identical (stdout `diff` clean) but **byte-
  identical PNGs** (`cmp` clean) — matplotlib's `Agg` backend with this
  project's `savefig` call embeds no timestamp/random metadata that would
  differ between runs, so there was no need to fall back to "identical
  except metadata" reasoning.
- Confirmed by re-running `python -m router.feature_experiment` after all
  of Tasks 1-6 landed: every number in findings.md's existing "Feature-
  router experiment" section (the 12-row Spearman table and all 8 router
  rows, including `casc ceiling`) matched the file exactly, including
  4-decimal utility and CI values — the `bootstrap_mean_difference`
  refactor in Task 1 really did produce bit-for-bit identical output on
  the full real dataset, not just on the unit tests' small fixtures.
- The recommendation section cites two real numbers from earlier in the
  project (Core ML ~17ms vs CPU ~106ms iPhone 15 Pro forward pass) that
  don't appear anywhere in this repo (checked with `grep -rn` across
  `*.md` before writing them down) — they came from the task prompt's
  guidance, not from a file in this repo. If a future task needs to cite
  them again, they aren't independently verifiable from this repo alone;
  the source is this project's earlier (Windows-side) research notes.

## Review fix S2 — bootstrap budget-only vs. static baselines

- Adding `budget_only_vs_always_local`/`budget_only_vs_always_offload` as
  two more `BootstrapResult` fields on `BudgetResult` (same
  `bootstrap_mean_difference(frame_ids, diff, seed=42)` call already used
  for the cascade comparisons, just fed `budget_only_on_time -
  always_local_on_time` / `- always_offload_on_time`) needed one new
  per-row array the loop didn't already have: `always_offload_on_time`
  (`always_local_on_time`/`budget_only_on_time` already existed). Cheap to
  add alongside the other two `_on_time_per_row` calls in `run_experiment`.
- `assert first == second` in the integration test already covers
  determinism for these two new fields for free — they're plain dataclass
  fields on `BudgetResult`, no extra equality-check work needed beyond
  asserting `ci_low`/`ci_high` are finite (mirroring the existing
  `vs_budget_only`/`vs_always_local` finite-check block).
- Confirmed the fix mechanically rather than trusting the numbers by eye:
  captured full stdout from the unmodified script first (worktrees don't
  work here per Task 1's learning above), then diffed it against stdout
  after the change — every previously-printed number was byte-identical
  and the only diff lines were the two new "budget-only vs always-local /
  always-offload" lines per budget, which matched the reviewer-supplied
  intervals exactly. Two consecutive post-fix runs were also identical
  (stdout diff clean), and the figure PNGs `cmp`-identical, consistent with
  Task 6's finding that this script's `Agg` output has no run-to-run
  metadata drift.

## Review fix S3 — offload-dominance shares replace the untested "confidence
   isn't separating" claim

- The reviewer's suspicion was exactly right: on the real held-out set,
  offload accuracy >= local accuracy on 98.00% of test rows (local
  strictly better on only 2.00%), and the tuned raw threshold at every
  budget from 200-500 ms keeps only 1.00% of rows local (learned's tuned
  threshold of 1.00 keeps 0.00%) — the cascades weren't failing to
  discriminate, there was essentially nothing left to discriminate.
- Added `share_score_ge_tuned_threshold` to `ScoreBudgetResult` (computed
  for every score/budget — cheap, one extra `.mean()` on arrays already in
  hand in `_score_budget_result`, no reason to special-case it to only
  200-500 ms/raw in the code even though the findings prose only cites
  those values) and `offload_dominance_share`/`local_strictly_better_share`
  on `BudgetExperimentResult` (computed once outside the budget loop, since
  a row's local/offload accuracy doesn't depend on the budget being
  evaluated). Both are plain `float` dataclass fields, so the integration
  test's existing `assert first == second` covers their determinism for
  free — only needed to add finite/[0,1] assertions.
- Confirmed the "only new lines added" constraint mechanically, the same
  way S2's fix did: captured full stdout from the unmodified script first
  (`python -m router.budget_experiment` against the real database, since
  worktrees don't work here per Task 1's learning), diffed it against
  post-fix stdout, and every previously-printed number was byte-identical
  — the only diff lines were the new offload-dominance line (once, right
  after the held-out split line) and one new `{score} share of test rows
  with score >= tuned threshold: ...` line per score per budget. Two
  consecutive post-fix runs were stdout-identical and produced `cmp`-
  identical figure PNGs, consistent with S2's and Task 6's findings that
  this script has no run-to-run metadata drift.
- When rewriting the findings prose, resisted the temptation to also
  soften the 100-150 ms bullet (it never made the "confidence isn't
  separating" claim the reviewer flagged — it already correctly attributed
  the low tuned thresholds there to the tight budget, not to confidence
  quality) — only added the new share numbers to it for consistency,
  without changing its causal claim, matching the fix's "everything else
  unchanged" instruction.

## Review fix S4 — quantify why budget-only is near-oracle instead of just
   asserting it

- `sklearn.metrics.r2_score`/`mean_absolute_error` need no new model fit or
  predict call: `_SplitContext.predicted_offload_ms` (train and test) was
  already computed by `_make_context` before the budget loop runs, so the
  new R2/MAE block in `run_experiment` just feeds those existing arrays
  plus each split's `offload_latency_ms` column straight into the two
  sklearn functions — one extra `import`, four `float(...)` calls, no
  behavior change to the model or the routing.
- On the real database: test R2 = 0.9951, MAE = 6.60 ms; train R2 = 0.9952,
  MAE = 6.47 ms — train and test are essentially identical, confirming the
  reviewer's "tuning on train rows adds no meaningful optimism" point
  mechanically rather than by assertion.
- Added an optional one-line-per-budget diagnostic (`budget_only_true_on_time`
  on `BudgetResult`) that reruns `budget_only_actions` with each row's
  *true* `offload_latency_ms` instead of the predicted one — same function,
  different input array, so it needed no new routing logic. The predicted
  vs. true gap this exposes is tiny everywhere (largest at 400 ms: 0.7395
  true vs. 0.7294 predicted, a ~1-point difference) — direct, printed
  evidence for the "prediction error costs almost nothing here" claim,
  distinct from the R2/MAE numbers (which describe the predictor in
  isolation, not what its error costs the policy).
- Confirmed "only new lines added" the same mechanical way S2/S3 did:
  captured `python -m router.budget_experiment` stdout before editing
  (worktrees don't work here, per Task 1's learning), diffed after — every
  previously-printed number was byte-identical; the only new lines were
  one predictor R2/MAE line in the preamble and one true-vs-predicted
  on-time-accuracy line per budget. Two post-fix runs were stdout-identical
  and the figure PNG was `cmp`-identical to the pre-fix run (the plotting
  code and its inputs are untouched — only new dataclass fields and print
  lines were added).
- `mypy`'s two-line-per-call style for `r2_score(...)`/`mean_absolute_error(...)`
  wrapped in `float(...)` triggered no line-length or type complaint; no
  stub gaps for these two `sklearn.metrics` functions in this project's
  mypy config, unlike some `sklearn.ensemble` types elsewhere in the repo.

## Review fix S5 — per-panel axis clipping so the tight-budget panels are readable

- The reviewer's finding described the always-offload marker forcing the
  x-axis wide (its mean latency is ~299 ms, near-network-bound, almost
  regardless of budget). Clipping x to the non-offload points + budget line
  fixed that half, but exposed a **second, independent** way the same
  marker can silently disappear: at the 300 ms budget, always-offload's
  *latency* (299 ms) is already within the clipped x-range (close to the
  300 ms budget line), but its *accuracy* (0.36) is far below every other
  point's accuracy (0.62-0.70) — so with `_panel_axis_limits` checking only
  x, the marker would silently vanish off the bottom of the y-axis with no
  error, the exact same failure mode the finding flagged, just on the other
  axis. This is a coincidence worth remembering: the "300 ms, 0.36" example
  value the finding's own guidance text uses for the annotation label is
  this exact panel — it's the one case where offload is x-in-range but
  y-out-of-range, so a fix that only checks x silently fails the finding's
  own worked example. Fixed by checking both `mean_latency_ms > xlim[1]`
  and `on_time_accuracy` outside `ylim` in `_panel_axis_limits`, and, when
  clamping the marker to an edge, clamping **only** the axis that's
  actually out of range (so the 300 ms panel's marker keeps its true
  x-position near the budget line and only its y is clamped to the bottom
  edge) rather than always snapping both coordinates to a corner.
- Caught this only by opening the regenerated PNG and checking each panel
  individually against its printed on-time-accuracy value — a "PNG is
  non-empty" smoke test can't catch a marker that silently clipped off an
  axis, same lesson as Task 4's legend-clipping gotcha. Added a dedicated
  test (`test_panel_axis_limits_clips_far_off_always_offload_accuracy`)
  that reproduces exactly this shape (in-range x, out-of-range y) so a
  future regression here has a fast, non-visual test rather than relying on
  eyeballing the figure again.
- Confirmed "stdout unchanged" mechanically the same way S2/S3/S4 did:
  captured `python -m router.budget_experiment` stdout before touching
  `budget_plot.py`, diffed after — byte-identical, since this fix only
  touches plotting/axis code, never the computed metrics. Two post-fix runs
  produced `cmp`-identical PNGs (no metadata drift, consistent with every
  earlier task/fix touching this script).

## Review fix S6 — test that tuning uses the train context, not the test
   context

- `_score_budget_result` is easy to unit-test in isolation: `_SplitContext`
  only needs `rows` (a `pd.DataFrame` with `frame_id`, `local_latency_ms`,
  `local_accuracy`, `offload_latency_ms`, `offload_accuracy` — the columns
  `outcomes`/`_policy_metrics` read), `local_latency_ms`,
  `predicted_offload_ms` (set equal to the row's true offload latency, so
  there's no prediction error to reason about) and `scores` (a
  `{"score": array}` dict; the key can be anything since `_score_budget_result`
  takes `score_name` as a parameter — it isn't restricted to the module's
  real `SCORE_NAMES`). No Postgres fixture needed.
- To make a swapped train/test argument observable, built two 2-row contexts
  whose best threshold provably differs: with `cascade_actions`' `score <
  threshold` rule, a single row's score sitting strictly between two
  `THRESHOLDS` steps creates a threshold *band* where only that row escalates
  and on-time accuracy peaks, bounded above and below by lower accuracy
  outside the band. Placing that row's score at 0.3 (train) vs. 0.02 (test)
  puts the peak band's first threshold at 0.35 vs. 0.05 respectively — worked
  out by hand-simulating `cascade_actions` across all 21 `THRESHOLDS` values,
  not by trial and error.
- For the flat-sweep case, the simplest reliable construction is a score of
  exactly `1.0` (the sweep's own maximum, since `THRESHOLDS` tops out at
  1.0): `score < threshold` is then false for every threshold in the sweep,
  so every threshold produces the identical all-LOCAL policy regardless of
  what the row's accuracy/latency values are — no need to reason about a
  coincidental tie between two different policies' accuracies.
- Verified the tests actually guard the AC (not just pass vacuously) by
  temporarily changing `_score_budget_result`'s `tuned_threshold =
  _tuned_threshold(train_ctx, ...)` line to pass `test_ctx` instead, which
  failed both new tests (0.55 instead of 0.0 for the flat-sweep case — the
  flat-sweep test's own `test_ctx` fixture has a real, non-flat sweep, so a
  swap doesn't just silently produce another flat/0.0 result), then
  restored the line and confirmed `git diff training/router/budget_experiment.py`
  showed no change.
- Added the two tests to the existing `test_budget_experiment.py` rather
  than a new file: that file already mirrors `budget_experiment.py`
  one-to-one (same pattern as `test_feature_experiment.py`/
  `feature_experiment.py`), and this repo has no precedent anywhere for
  splitting a source file's tests into separate unit/integration files by
  suffix — introducing one for a single review fix seemed worse than a
  short docstring note that the file now holds both kinds of test.

## Review fix N3 — de-duplicate `_ACTION_DTYPE` via a shared `constant_actions` helper

- `latency_policies.py`'s own action-building functions (`budget_only_actions`,
  `cascade_actions`, `oracle_actions`) all start from `np.full(..., LOCAL,
  dtype=_ACTION_DTYPE)` as a *default* that a boolean mask then partially
  overwrites — none of them ever builds a genuinely constant (all-one-value)
  action array, so `constant_actions` had nothing to replace inside that
  module; it only replaced `budget_experiment.py`'s
  `_constant_actions(len(test_rows), LOCAL/OFFLOAD)` calls for
  always-local/always-offload, which really are constant arrays.
- Moving `_ACTION_DTYPE` up next to the LOCAL/OFFLOAD/ESCALATE labels
  incidentally fixed the spacing problem the finding named too: the old
  site (after `learned_score`, before `budget_only_actions`) had only one
  blank line before it because the constant+comment sat between two
  functions; deleting it from there and leaving `learned_score` followed
  directly by `budget_only_actions` restores PEP 8's two-blank-line gap
  with no separate spacing fix needed.
- Confirmed pure refactor, not just by re-running the unit tests: captured
  `python -m router.budget_experiment` stdout against the real database
  before touching either file, diffed after — byte-identical — and `cmp`
  on the regenerated figure PNG was also identical, consistent with every
  earlier task/fix's finding that this script has no run-to-run metadata
  drift.
