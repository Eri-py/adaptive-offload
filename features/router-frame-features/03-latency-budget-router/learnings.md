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
