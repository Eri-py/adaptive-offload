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
