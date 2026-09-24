# Review — Latency-Budget Router

## Verdict

The implementation meets the spec and has no blockers. The policies, oracle,
outcomes and metrics in `latency_policies.py` match the spec exactly. The
threshold is tuned on train rows only. The learned score never trains on a
simulated frame. The offload latency the router uses is predicted, never the
true value. I re-ran `python -m router.budget_experiment` read-only: the
printed output matches `findings.md` and the figure is byte-identical to the
committed PNG. `python -m router.feature_experiment` still reproduces spec
02's numbers, and the 35 new or affected tests pass.

The problems are in the write-up, not the code.

- **The headline claim holds.** Budget-only beats always-local and
  always-offload at every budget. I bootstrapped that comparison (results
  under S2) and every 95% interval lies above zero. The findings should
  report those intervals instead of the current magnitude argument.
- **The explanation of the cascade result is off.** On the held-out rows,
  offload is at least as accurate as local on 98% of rows; local is strictly
  better on only 2%. So the tuned cascade reduces to "run local, then offload
  whenever it fits", and raw and learned scores tie because both hit that
  same limit. The write-up says instead that confidence "isn't separating"
  frames (S3).
- **The near-oracle result depends on the simulator.** Budget-only lands
  close to the oracle partly because the simulated offload latency is almost
  perfectly predictable (test R² 0.995, MAE about 6.6 ms). The write-up
  should say so (S4).
- **One sentence is wrong.** The "Why the framing changed" paragraph says a
  0.09-utility latency cost is "far less than a single accuracy point". It
  is 9 points (S1).

The recommendation to resume the phone/server work to validate budget-only
is sound. S4 makes it better motivated, since real-world latency
predictability is exactly what that work would test.

## Acceptance Criteria

- **All six budgets × all six policies report mean acc, on-time acc, mean latency, over-budget share — MET.** `budget_experiment.py:259-292` builds every `BudgetResult`, and `_print_report` (`budget_experiment.py:347-385`) prints all four metrics for `_POLICY_ROWS` (`:326-333`). Reproduced on the real DB.
- **Budget-only runs local when predicted offload exceeds the budget — MET.** `latency_policies.py:96-104`; test `test_latency_policies.py:33-37`.
- **Cascade keeps local when confidence ≥ threshold — MET.** `latency_policies.py:121-124`; `test_latency_policies.py:40-62`.
- **Below threshold: escalate (local+offload latency, offload accuracy) if it fits the remaining budget, else keep local — MET.** `latency_policies.py:122-124`, `outcomes` at `:169-185`; tests `test_latency_policies.py:40-51, 81-92`.
- **No held-out rows used to train or tune; learned score never trained on a simulated frame — MET.**
  - The latency model is fit on `train_rows` (`budget_experiment.py:253`).
  - The threshold is tuned on `train_ctx` (`:172`).
  - Stage 1 excludes `simulated_frame_ids` (`latency_policies.py:69-78`); test `test_latency_policies.py:274-282`.
  - The tuning path (train ctx vs test ctx) is not covered by a test; see S6.
- **Late results count as 0 in on-time accuracy — MET.** `latency_policies.py:188-197`; `test_latency_policies.py:95-100`.
- **Oracle on-time ≥ every policy; never over budget when an option fits — MET.** The tie-break loop at `latency_policies.py:153-162` only picks options that fit. Property test at `test_latency_policies.py:174-202`, plus the integration check at `test_budget_experiment.py:190-196`. On the real data the oracle is over budget only at 100 ms (30.11%), and only on rows where nothing fits.
- **Full threshold sweep for both scores and the train-tuned operating point shown per budget — MET (figure only).** The sweeps and tuned points are drawn per panel (`budget_plot.py:79-110`). Tuned τ is printed (`budget_experiment.py:369-377`). The sweep values themselves are not printed (N6), and the 100–200 ms panels are hard to read (S5).
- **95% bootstrap CIs vs budget-only and vs always-local, plus usefulness verdict — MET.** `budget_experiment.py:184-197`, printed at `:369-385`; `findings.md` has them per budget.
- **Identical numbers across runs — MET.** Integration test `test_budget_experiment.py:154-157`. My re-run matched `findings.md`, and the PNG was `cmp`-identical to the committed one.
- **`findings.md` has the new section, figure next to it, recommendation reflects results — MET.** `findings.md:300-650`, `training/router/latency_budget_curves.png`. S1, S3 and S4 cover accuracy problems in the text.
- **Spec 02 experiment and earlier router scripts still reproduce — MET.** I re-ran `router.feature_experiment`. All eight router rows' CIs and utilities match the existing `findings.md` section. The refactor at `evaluation.py:132-178` preserves the RNG draw order.

## Scope

The diff matches the plan's Files Changed table exactly: `latency_policies.py`, `evaluation.py`, `budget_plot.py`, `budget_experiment.py`, the PNG, `findings.md`, `pyproject.toml`, and the four test files. There is no scope drift. Nothing touches spec 02's routers or results. No schema change, no DB writes.

## Blockers

None.

## Suggestions

#### S1 — "Why the framing changed" misstates the size of the soft-utility latency cost

- **File:** `training/router/findings.md:306-309`
- **Issue:** The text says "300 ms of extra latency costs only 0.09 utility, far less than a single accuracy point is worth". Accuracy is on a 0–1 scale, so 0.09 utility equals 9 accuracy points. The sentence is factually wrong. The real reason always-offload wins under the soft utility is that offload's mean accuracy gain (0.7709 − 0.6243 ≈ 0.147) outweighs its latency cost (≈225 ms extra × 0.3 ≈ 0.068).
- **Fix:** Rewrite the sentence along those lines. For example: "offload's ~15-point accuracy advantage outweighs the ~0.07 utility its extra ~225 ms costs, so a router has little to gain by avoiding it."
- **Decision:** — _(pending)_

#### S2 — Budget-only vs the static baselines should be bootstrap-tested in the script, not argued from magnitude

- **File:** `training/router/budget_experiment.py:184-189`, `training/router/findings.md:467-478`
- **Issue:** The headline result has no interval. The magnitude argument in its place compares gaps to CI widths from *other* comparisons, which isn't valid reasoning. The two smallest gaps (1.9 pts vs always-local at 150 ms, 2.3 pts vs always-offload at 500 ms) are smaller than the widest interval it cites (0.07). I ran the paired photo-level bootstrap read-only with the existing `bootstrap_mean_difference` (seed 42). Every interval is above zero:
  - budget-only − always-local: 100 [0.0311, 0.0376], 150 [0.0139, 0.0245], 200 [0.0159, 0.0320], 300 [0.0410, 0.0791], 400 [0.0732, 0.1376], 500 [0.1024, 0.1809]
  - budget-only − always-offload: 100 [0.3443, 0.4242], 150 [0.4834, 0.5946], 200 [0.4447, 0.5465], 300 [0.2902, 0.3571], 400 [0.1097, 0.1357], 500 [0.0203, 0.0253]
- **Fix:**
  - In `run_experiment`, compute and print these two intervals per budget (the per-row on-time arrays are already built at `:268-269`).
  - In `findings.md`, replace the "not bootstrap-tested… That said" paragraph with the intervals, and drop the matching follow-up at `findings.md:646-650`.
- **Decision:** — _(pending)_

#### S3 — The cascade explanation blames confidence separation; the data shows there is almost nothing for the cascade to win

- **File:** `training/router/findings.md:497-540`
- **Issue:** The findings say the thresholds saturate because "confidence isn't separating 'local is good enough' frames". The findings also say raw and learned are "equivalent" because "the bottleneck is the local-first latency tax". Neither claim was tested, and the held-out rows point to a simpler cause:
  - Offload accuracy ≥ local accuracy on 98% of held-out rows. Local is strictly better on only 2%.
  - Raw `mean_confidence` is ≥ 0.95 on only 1% of rows.
  - Learned τ = 1.00 escalates every row.
  - So at 200–500 ms, both tuned cascades are "run local, then offload whenever it fits". The two scores tie because they collapse to the same policy, not because they discriminate equally well.
  - The best any cascade could gain is the ≤2% of rows where local wins. The local-first latency cost then decides the comparison against budget-only.
  - The sweep curves in the figure point the same way: on-time accuracy rises monotonically with escalation rate.
- **Fix:**
  - Rewrite the mechanism bullets around the offload-dominance fact, and consider having the script print the local ≥ offload share.
  - Say explicitly that the raw/learned tie says nothing about the scores' ranking quality.
  - Scope the verdict to the spec's local-first cascade: a hybrid that can also offload directly was not tested.
- **Decision:** — _(pending)_

#### S4 — State that budget-only's near-oracle result rests on almost perfectly predictable simulated latency

- **File:** `training/router/findings.md:426-434`, `:602-608`, `:614-620`
- **Issue:** "Within 0.4–1.6 points of the oracle" and "That is a real result" don't say why the result is so strong. The offload-latency model has held-out R² ≈ 0.995 and MAE ≈ 6.6 ms (train MAE 6.5 ms, so tuning on train rows adds no meaningful optimism). Offload is also at least as accurate on 98% of rows. Given both, "offload iff it fits" is close to optimal almost by construction of the simulator. Caveat 1 names the analytical relationship, but not how much of the headline depends on it.
- **Fix:**
  - Add the predictor's test R²/MAE to the headline section.
  - Replace "a real result" with a sentence saying the gap to the oracle is small in simulation because latency is nearly deterministic given the conditions. Present real-network predictability as the key thing the phone/server work must test.
  - The recommendation itself can stay as is.
- **Decision:** — _(pending)_

#### S5 — Three of the six figure panels can't be read

- **File:** `training/router/budget_plot.py:147-149` (and `_plot_panel` axis limits), `training/router/findings.md:550-559`
- **Issue:** In the 100, 150 and 200 ms panels, the x-axis stretches to ~300 ms to fit always-offload. The sweep curves, tuned points, budget-only, always-local and the oracle all overlap in one blob in the top-left corner. The figure is a spec deliverable, and the findings text also says curves "spread out visibly from 200 ms up", which the 200 ms panel doesn't show.
- **Fix:**
  - Clip each panel's x-range to the non-always-offload points plus the budget line, with padding.
  - Draw always-offload as an edge arrow or annotation when it falls outside that range.
  - Consider dropping `sharey=True` so the small gaps are visible.
  - Update the description at `findings.md:550-559` to match.
- **Decision:** — _(pending)_

#### S6 — Train-only threshold tuning and its tie-break have no test

- **File:** `training/router/budget_experiment.py:124-143`, `:171-172`
- **Issue:** The AC "no held-out row used to tune" rests on `_tuned_threshold(train_ctx, …)` receiving the train context. Nothing tests that `_score_budget_result` passes the train context rather than the test context, or that ties go to the lowest threshold. Swapping the two arguments would pass every current test and silently leak held-out rows into tuning.
- **Fix:** Add a unit test with two small `_SplitContext`s whose best thresholds differ. Assert that `_score_budget_result(...).tuned_threshold` equals the train context's pick, and that a flat sweep returns 0.0.
- **Decision:** — _(pending)_

## Nitpicks

#### N1 — Small numeric misstatements in findings

- **File:** `training/router/findings.md:432`, `:604`, `:541`
- **Issue:**
  - "Within about 0.4–1.6 points" of the oracle: the 150 ms gap is 0.0014 (0.14 points).
  - The oracle's headroom over the cascade, given as "~0.03–0.04", is actually 0.019–0.044 (150 ms 0.0186, 200 ms 0.0205, 400 ms 0.0444).
- **Fix:** Change these to "0.1–1.6 points" and "~0.02–0.04".
- **Decision:** — _(pending)_

#### N2 — Multi-line comments break the one-line comment rule

- **File:** `training/router/latency_policies.py:89-92`, `training/router/budget_experiment.py:56-59`, `:242-245`
- **Issue:** Four-line comments, partly restating context. For example, `:56-59` explains that an array is "never mutated… but kept explicit regardless". `.claude/coding-guidelines.md` "Comments" allows one line, or two when truly needed.
- **Fix:** Cut each to one line. For example: `# <U8 fits "ESCALATE"; np.full would otherwise size the dtype from the fill value`.
- **Decision:** — _(pending)_

#### N3 — `_ACTION_DTYPE` is duplicated, and the constant sits mid-module

- **File:** `training/router/budget_experiment.py:60-66`, `training/router/latency_policies.py:88-93`
- **Issue:** The private constant is copied between modules. In `latency_policies.py` it sits after the Task 3 functions, with a single blank line after `learned_score`.
- **Fix:** Move the constant to the top of `latency_policies.py` next to the labels. Expose a `constant_actions(n, action)` helper there and use it in `budget_experiment.py`.
- **Decision:** — _(pending)_

#### N4 — `_tuned_threshold` duplicates the `_threshold_sweep` loop

- **File:** `training/router/budget_experiment.py:107-143`
- **Issue:** The two loops are identical except for the reduction step. `max()` returns the first maximal element, so tuning can reuse the sweep and keep the lowest-threshold tie-break.
- **Fix:** `return max(_threshold_sweep(ctx, score_name, budget_ms), key=lambda t: t[1].on_time_accuracy)[0]`.
- **Decision:** — _(pending)_

#### N5 — Docstrings refer to plan task numbers

- **File:** `training/router/budget_experiment.py:1`, `training/router/budget_plot.py:10`
- **Issue:** "Wires Tasks 1-4 together" and "(Task 5)" tie the code to a process document that will go stale.
- **Fix:** Describe the modules in domain terms instead, e.g. "Runs the latency-budget experiment: …".
- **Decision:** — _(pending)_

#### N6 — Threshold sweep values appear only in the figure

- **File:** `training/router/budget_experiment.py:347-385`
- **Issue:** The printed report shows only tuned τ, so the per-threshold numbers can't be read in text form.
- **Fix:** Optionally print a compact per-threshold on-time/latency line per score.
- **Decision:** — _(pending)_

## Tests

- **Added:**
  - `test_latency_policies.py` has 18 tests. The hand-computed branch tests cover both policies, `outcomes`, on-time zeroing and metrics, with inclusive boundaries. There are oracle choice, fallback and tie-break tests, and a seeded property test that the oracle dominates. The predictor, score and determinism tests are also here.
  - `test_evaluation.py` has 3 new tests for the generic bootstrap.
  - `test_budget_plot.py` has 3 smoke tests.
  - `test_budget_experiment.py` has 1 ephemeral-Postgres integration test that runs twice and checks determinism, completeness, finiteness and oracle dominance.
- **Result:** All 35 tests in these four files pass, and `ruff check router tests` is clean.
- **Gaps:**
  - Nothing tests that tuning uses train rows, or the tie-break (S6).
  - The "hand-built" bootstrap test uses a constant diff, so it doesn't exercise the percentile computation. The wrapper-equivalence test and the byte-identical spec 02 re-run cover that path well enough.
  - The integration test's synthetic `mean_confidence` is uniform noise, so it checks that the cascade branches but not whether it behaves sensibly. That's acceptable at the prototype bar.
  - The plot smoke test can't catch readability problems like S5.
- **Methodology spot checks (read-only):**
  - Train/test rows are 80,000/20,000, 200 condition rows per photo.
  - Predictor MAE is 6.47 ms on train vs 6.60 ms on test, so tuning τ on train rows with an in-sample latency model adds negligible optimism.
  - Budget-only using the *true* offload latency scores only 0.0003–0.0101 higher on on-time accuracy than the predicted-latency version (largest gaps at 300–400 ms). The prediction error costs at most about 1 point, which supports S4.

## Recommended Decisions

- **S1** — Accept — The sentence is factually wrong by a factor of ~9 points, and the fix is a one-sentence rewrite.
- **S2** — Accept — It's cheap (the arrays exist already), I've confirmed it strengthens the headline, and it replaces a weak magnitude argument with the formal interval the findings themselves call for.
- **S3** — Accept — The current mechanism text attributes the result to something the data doesn't support. The offload-dominance explanation is measurable and changes how the raw/learned tie should be read.
- **S4** — Accept — The headline claim needs its main caveat quantified, and doing so makes the phone/server recommendation more precise.
- **S5** — Accept — The figure is a spec deliverable, and half its panels can't be read. The fix is axis limits in one function.
- **S6** — Accept — This guards the no-leakage acceptance criterion with a small unit test. An argument swap would otherwise go unnoticed.
- **N1** — Accept — These are trivial text corrections, and the findings claim to report numbers exactly.
- **N2** — Accept — A direct coding-guidelines violation, and a mechanical fix.
- **N3** — Accept — Removes duplication of a gotcha-prone constant at low risk.
- **N4** — Accept — A one-line replacement that keeps the tie-break semantics and removes a duplicated loop.
- **N5** — Accept — Trivial, and stops docstrings from going stale.
- **N6** — Decline — The figure already satisfies the AC. Printing 21 × 2 × 6 extra lines adds noise to the report for little value.
