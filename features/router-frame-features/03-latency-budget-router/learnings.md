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
