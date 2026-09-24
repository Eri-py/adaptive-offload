# Review — Feature Router Experiment

Reviewed range: `9fe98b5..9ba09ac` on `feature/router-frame-features`.

## Verdict

The code does what the spec asks, and it does it correctly. The data loading, the
two-stage leakage guards, the cascade scoring, the photo-level bootstrap and
the determinism all check out. The 27 new tests pass, ruff and mypy are clean on
the new files, and a read-only rerun on the real database reproduces every
number in `findings.md` exactly. The problem is the write-up, which is the key output. The
headline in `findings.md` says all four routers are "worse than always-offload with 95% confidence
… a stronger, more negative result than the baseline routers". Most of that
comes from how stage 2 was trained, not from what the features can do. Stage
2 is an unweighted 0/1 classifier, but the two kinds of mistake cost very
different amounts. On the train rows, a correct LOCAL pick gains 0.088 utility
on average, and a wrong one loses 0.220. The routers pick local on 42–60% of
test rows. The same cost-blind classifier setup on the old five features scored
0.6574 in `router.baseline`, and `decide_first_linear` scores 0.6575, which is
the same result, not a stronger one. When stage 2 is made cost-aware, `decide_first_linear` ties
always-offload (CI straddles 0). The gradient-boosted variants lose for a
different reason, which is that they memorise photo identity (98–99% train
accuracy against 53–56% test accuracy). "Don't resume phone/server work yet" still holds,
because no variant, cost-aware or not, clears the bar. But the explanation, the
comparison with the earlier routers, and the suggested next steps need to be
rewritten before anyone relies on them, or before spec 03 cites them. That is
B1. Everything else is polish.

## Acceptance Criteria

- **Diagnostics over all 5,000 frames — MET.** `training/router/feature_experiment.py:79` computes Spearman
  over `load_frame_table`'s output. The rerun confirmed that the table holds 5,000 frames.
  The results are printed at `feature_experiment.py:128-129` and recorded in
  `findings.md:108-121`.
- **No held-out test photo in any training data — MET.** Stage 1 trains only on
  frames that were never simulated (`two_stage.py:106`), which is stricter than
  the spec requires. Stage 2 trains only on train-photo rows (`two_stage.py:220`). Both are
  asserted against the frame sets that actually reached `.fit()`
  (`tests/router/test_two_stage.py:132-153`).
- **Decide-first shown next to local/offload/oracle with headroom share — MET.**
  `evaluation.py:83-105` and `feature_experiment.py:137-151`.
- **Cascade scoring (escalate means offload accuracy plus local+offload latency; accept means
  local) — MET.** `evaluation.py:42-47` and `evaluation.py:63-68`. Tested at
  `tests/router/test_evaluation.py:64-81`.
- **95% photo-resampled bootstrap CI and a usefulness verdict per router — MET.**
  `evaluation.py:119-148`. It resamples unique `frame_id`s and computes a ratio of
  sums, so a photo drawn twice counts twice, which is correct.
- **Identical numbers on rerun — MET.** Seeds are set at `feature_experiment.py:81`
  and `115` and at `two_stage.py:62-70`. My rerun matches `findings.md` to the
  printed digit.
- **`findings.md` section with diagnostics, results, verdicts, compute times and
  recommendation — PARTIAL.** Every required element is present
  (`findings.md:83-209`). But the interpretation and the recommendation's rationale
  misattribute the negative result (B1). The cascade-ceiling discussion also
  leaves out a number that matters (S2).
- **Existing router scripts unchanged — MET.** `baseline.py`, `analyze.py` and
  `dataset.py` are not in the diff. Per learnings, `router.baseline` reprints
  0.6812 / 0.6811.

## Scope

The diff matches the plan's Files Changed table exactly: five new modules, five
new test files, and a `findings.md` edit. Nothing is outside scope. Workflow
artifacts are ignored per the rules.

## Blockers

#### B1 — findings.md blames the features and the photo count for the negative result, but the stage-2 loss and GBT memorisation explain most of it

- **File:** `training/router/findings.md:148-156`, `training/router/findings.md:181-200`
- **Issue:** The section presents "worse than always-offload with 95% confidence" as a
  stronger negative result than the baselines. It blames stage-2 overfitting on 400 photos
  and recommends more train photos. Checks I ran read-only against the real DB found three things:
  1. **The two errors cost very different amounts.** A correct LOCAL pick gains 0.088 utility on
     average, and a wrong one loses 0.220. Yet stage 2 is fit on 0/1 labels and
     picks LOCAL at p > 0.5, when the break-even point is about 0.71. It picks
     local on 42–60% of test rows.
  2. **Cost-aware stage 2 changes the linear verdict.** With `sample_weight = |margin|`
     or margin regression, `decide_first_linear` gets 0.6807–0.6810, CI
     [-0.0030, +0.0018] / [-0.0020, +0.0014], which is a tie. The same setup
     improves `cascade_linear` to 0.660, but it stays significantly negative.
  3. **Linear stage 2 is not overfitting, and GBT is.** Linear train/test accuracy
     is 0.570/0.560 (decide-first) and 0.644/0.604 (cascade). GBT is
     0.982/0.555 and 0.992/0.526, because `predicted_gap` takes only 400 distinct
     values in train, one per photo, so it works as a photo ID. See S1.

  Also, `decide_first_linear` (0.6575) matches `router.baseline`'s logistic
  classifier (0.6574), so this result is not "more negative than the baselines". It
  repeats the same classifier-loss artifact. The earlier "~0.68" line
  (`findings.md:17`) hides that.
- **Fix:** Rewrite the interpretation paragraph and the recommendation's rationale:
  - State the cost asymmetry and the cost-aware result (a linear tie; the cascade and GBT still
    below always-offload).
  - Limit the overfitting claim to the GBT variants and name its cause (per-photo stage-1
    outputs).
  - Drop "stronger, more negative than the baselines". Say instead that it
    reproduces the earlier logistic router's 0.6574.
  - Replace "more than 400 train photos" with "cost-aware stage 2 / margin regression"
    as the first next step.

  Keep the bottom line: no router clears the bar, so don't resume
  `features/server-served-benchmark/`. Add one line saying spec 03
  (`03-latency-budget-router`) reframes the objective and supersedes this
  recommendation. Ideally, make the cost-aware stage 2 a fifth/sixth row in
  `feature_experiment.py` rather than only prose, so the claim can be reproduced.
- **Decision:** Accepted — addressed in "Address B1: cost-aware stage-2 routers and a corrected explanation"

## Suggestions

#### S1 — GBT stage 2 memorises photos through the per-photo stage-1 outputs

- **File:** `training/router/two_stage.py:67-70`, `training/router/two_stage.py:48`
- **Issue:** `predicted_gap` and `p_local_good_enough` are constant within a photo,
  which gives 400 unique values in 80,000 train rows. `HistGradientBoostingClassifier`
  splits on them to learn each photo's own outcome: 98–99% train accuracy against 53–56% test.
  Its automatic early stopping uses a random validation split over rows, not photos, so it
  cannot catch this. The same thing would happen with 4,000 photos.
- **Fix:** Constrain the GBT stage 2. Options:
  - Set a large `min_samples_leaf`, for example at least one photo's rows × 20.
  - Add `monotonic_cst` on the two stage-1 columns.
  - Turn off early stopping.
  - Or use a photo-grouped validation split.

  At minimum, describe the mechanism in `findings.md` instead of "400 photos is too few".
- **Decision:** — _(pending)_

#### S2 — Quantify the cascade's own ceiling, which is above always-offload

- **File:** `training/router/findings.md:159-168`
- **Issue:** The note says the cascade's ceiling is below the shown oracle but gives no
  number. Its reasoning is also garbled: "an accepted row can never earn the (uncharged)
  offload latency the oracle implicitly assumes". On the test rows, the cascade ceiling,
  max(local, escalated) per row, is **0.7091**. Always-escalate is 0.6591 and
  always-offload is 0.6812. So the cascade has real headroom of about 0.028 over
  always-offload, which readers can't tell from the text. The reason it is lower is simply
  that escalating adds about 74 ms of local latency (≈0.022 utility) to every offloaded
  row.
- **Fix:** Print the cascade ceiling in the report (for example
  `np.maximum(local_utility, escalated_utility).mean()` in `evaluation.py`, shown as an
  extra column for cascade rows), and replace the prose with those numbers.
- **Decision:** — _(pending)_

#### S3 — The row-alignment fix depends on two copies of the same filter, and no test covers it

- **File:** `training/router/feature_experiment.py:92-94`, `training/router/two_stage.py:221`
- **Issue:** Picks are positional arrays. Correct scoring depends on
  `run_experiment` and `run_router` independently applying the same `isin` filter to the
  same frame in the same order. If either changes, the numbers go silently wrong, which is
  the bug Task 5 already hit once. No test would fail, because the integration test
  checks only determinism and finiteness.
- **Fix:** Have `RouterResult` carry the `test_rows` it predicted on, or return picks as a
  `pd.Series` indexed like them, and score against that. Or add a test that
  asserts `run_router`'s test-row `frame_id` order equals the scoring frame's.
- **Decision:** — _(pending)_

#### S4 — The CI covers test-photo sampling only, from a single train/test split

- **File:** `training/router/feature_experiment.py:81`
- **Issue:** The bootstrap resamples the 100 test photos but keeps one fixed split and
  one fixed fit, so variance from which photos land in train is not captured. Here that
  doesn't change the verdict, since the linear tie and the GBT losses are large. But any
  future "useful" verdict near the boundary would rest on one split.
- **Fix:** Mention it in `findings.md` as a caveat. Optionally, repeat over a few split seeds
  and report the spread.
- **Decision:** — _(pending)_

## Nitpicks

#### N1 — The misalignment comment is 7 lines long and names the wrong mechanism

- **File:** `training/router/feature_experiment.py:85-91`
- **Issue:** The guidelines ask for comments of one line, or two at most. The comment also says the
  risk is "pandas's index-based Series arithmetic", but the scorers use positional
  `np.where` (`evaluation.py:58-68`). The index is ignored, which is why the error was silent.
- **Fix:** Shorten it to a line like `# Same filter/order as run_router's test rows; picks
  are positional.` (moot if S3 is applied).
- **Decision:** — _(pending)_

#### N2 — Module and dataclass docstrings run long and describe the implementation

- **File:** `training/router/two_stage.py:1-12`, `training/router/two_stage.py:84-90`, `training/router/evaluation.py:126-133`
- **Issue:** Several docstrings restate implementation details (for example "Vectorised: each frame's
  sum …", or which test inspects a field). That goes against the "why, not what" and brevity guidelines.
- **Fix:** Cut each one down to the one-line purpose plus any non-obvious reason.
- **Decision:** — _(pending)_

#### N3 — The frame table is loaded twice per run

- **File:** `training/router/frame_dataset.py:87`, `training/router/feature_experiment.py:75-77`
- **Issue:** `load_simulated_rows` calls `load_frame_table` again after
  `run_experiment` has already loaded it. The result is correct, just a redundant query.
- **Fix:** Optionally let `load_simulated_rows` accept an already-loaded frame table.
- **Decision:** — _(pending)_

## Tests

- **Added:** 27 tests across five files, all passing on my run.
  - hand-computed utility, cascade and headroom cases
  - bootstrap determinism, with the useful and not-useful directions
  - leakage guards that assert on the frame sets actually passed to `.fit()`
  - the single-class stage-2 fallback
  - loader joins and scoping (ephemeral Postgres)
  - an end-to-end run done twice
- **Good:** The leakage guards check what really reached `.fit()` instead of
  re-deriving the filter. The alternating-gap fixtures avoid flaky single-class splits.
- **Gaps:**
  - Nothing guards the pick/row alignment (S3).
  - The bootstrap has no test for the photo-weighted mean, for example unequal row counts per photo, where
    a frame-mean-of-means would give a different answer. This is minor, because the implementation is correct.
  - Nothing tests the diagnostics' handling of constant or NaN feature columns. Spearman returns
    NaN for these, and sorting by `abs()` would then place them unpredictably. There are none in the real
    data.
- **Fragile:** The integration test's finiteness check would not catch misaligned
  scoring (S3).

## Recommended Decisions

- **B1** — Accept — The key deliverable currently misstates why the routers lose and points the next step at the wrong fix. A prose rewrite, plus optionally cost-aware stage-2 rows, is cheap, and spec 03 builds on this conclusion.
- **S1** — Accept — The GBT variants are measuring photo memorisation, not features. At minimum, the findings text should say so. A constrained GBT is a small change.
- **S2** — Accept — One extra number (0.7091) changes how the cascade result reads, and it clears up garbled reasoning.
- **S3** — Accept — This already caused one silent bug. Carrying the test rows on `RouterResult` removes the duplicate filter for a few lines of code.
- **S4** — Decline — It doesn't change any current verdict, and spec 03 supersedes this experiment. A one-line caveat can go into the B1 rewrite if desired.
- **N1** — Accept — The comment is inaccurate as well as long, and the fix is trivial.
- **N2** — Decline — Style only, in research code that spec 03 will largely replace.
- **N3** — Decline — A redundant read-only query with no correctness impact.
