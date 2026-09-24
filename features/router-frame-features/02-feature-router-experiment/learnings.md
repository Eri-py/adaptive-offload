# Learnings — feature-router-experiment

## Task 1 — Frame-level data loading

- The `label` Postgres enum type (used by both `simulation_results.label` and
  `model_inference.model_path`) stores the enum *member names* (`'LOCAL'`/
  `'OFFLOAD'`), not the lowercase `Label` values (`"local"`/`"offload"`) —
  confirmed by migration `0002`'s docstring and `postgresql.ENUM("LOCAL",
  "OFFLOAD", name="label", ...)`. Raw SQL joining on `model_inference.model_path`
  must compare against the string literals `'LOCAL'`/`'OFFLOAD'`, not
  `Label.LOCAL.value`.
- `load_simulated_rows` composes `router.dataset.load_training_data` (SQL)
  with `load_frame_table` (also SQL) via a pandas `merge` in Python, rather
  than writing one larger SQL query — the per-frame feature columns are
  merged onto the simulated rows on `frame_id == file_name`, then `file_name`
  is dropped so only `frame_id` survives. This avoids re-deriving `gap` or
  duplicating `scene_complexity` (already present on `load_training_data`'s
  output via its own `scene_complexity` join), and keeps `load_frame_table`
  as the single place the frame-features join logic lives.
- All three loader functions are backed by two SQL queries + one merge, all
  going through the existing `datagen.persistence`/`router.feature_store`
  helper functions in tests (`store_complexity_scores`, `store_model_inference`,
  `store_features`, `create_run`/`store_results`) rather than raw ORM
  inserts or hand-written SQL — mirrors `test_persistence.py`/
  `test_feature_store.py`'s seeding style, so no new seeding pattern was
  introduced.
- `frame_features`'s 11 columns split cleanly into the task's two exported
  6-column constants (`IMAGE_COLUMNS` = `scene_complexity` + the 5 image
  stats; `CONFIDENCE_COLUMNS` = `detection_count` + the 5 confidence/box
  stats) — kept a third, unexported `_FRAME_FEATURE_COLUMNS` (the 11 minus
  `scene_complexity`) since that's the exact set `load_simulated_rows` needs
  to merge in without re-adding a duplicate `scene_complexity` column.

## Task 2 — Feature diagnostics

- `router.diagnostics.compute_feature_diagnostics` is pure (no DB access) —
  it takes any DataFrame with a `gap` column (in practice `load_frame_table`'s
  output) plus the feature column names to check, so it's testable with a
  small synthetic DataFrame instead of seeding Postgres, and reusable for
  both spec 01's frame features and the pre-existing `scene_complexity`.
- Followed the router test suite's existing convention (from
  `test_confidence.py`/`test_image_stats.py`) of avoiding `import pytest` —
  plain `def test_...()` functions only — since importing pytest here has
  previously pulled `_pytest`'s numpy integration into a stub incompatible
  with this venv.
- `scipy.stats.spearmanr` returns a result object whose `.correlation`/
  `.pvalue` unpack fine via tuple assignment (`rho, p_value = spearmanr(...)`)
  under the repo's existing blanket `scipy.*` mypy override
  (`ignore_missing_imports = true`), so no per-call `# type: ignore` was
  needed — matches how the task description said to check existing handling
  rather than adding one.
- To order by |rho| descending without a lambda-in-`sort_values` (which mypy
  strict flags less cleanly here), built the DataFrame unordered first, then
  reindexed by `result["spearman_rho"].abs().sort_values(ascending=False).index`.

## Task 3 — Evaluation and bootstrap

- `router.evaluation` has no pandas-stubs installed (`pip show pandas-stubs`
  finds nothing), and `[tool.mypy.overrides]` already sets
  `ignore_missing_imports = true` for `pandas.*`, so `pd.Series`/`pd.DataFrame`
  type hints resolve to effectively untyped members under mypy strict — no
  per-call `# type: ignore` needed for `.to_numpy()`, `np.where(...)` on a
  `pd.Series`, or `np.maximum` of two Series (all pass through unchecked).
  Only bare `np.ndarray` (no type args) trips `strict`'s `[type-arg]` check,
  hence `npt.NDArray[np.str_]` for the `picks` parameters.
- The bootstrap is fully vectorised without ever re-touching individual rows
  per resample: precompute each frame's sum of (router − offload) per-row
  differences and its row count once (`np.add.at` scatter-sum keyed by
  `np.unique(frame_ids, return_inverse=True)`'s inverse index), then each of
  the 2,000 resamples is just `frame_sums[draws].sum(axis=1) /
  frame_counts[draws].sum(axis=1)` over an `(n_resamples, n_frames)` draw
  matrix from a single `rng.integers(...)` call — no Python-level loop over
  resamples at all. Runs in well under a second even at the real dataset's
  ~100 frames, so it'll comfortably clear "seconds" at ~20k rows too.
- Chose exact float equality (`denominator == 0`) for the headroom-share
  zero-denominator check rather than `math.isclose` — the case that actually
  occurs is oracle-average exactly equal to offload-average (every row's
  oracle pick *is* offload, so `oracle_utility` and `offload_utility` are
  literally the same per-row values before averaging), not a near-zero
  floating-point residue from unrelated arithmetic, so exact comparison is
  the right check and easy to hit exactly in tests.
- Kept `oracle_utility` and `escalated_utility` as separate public functions
  (not just inlined into `summarize`/`score_cascade`) since the plan's
  Task 4 needs `escalated_utility` (or the equivalent comparison) again to
  compute the cascade's ACCEPT/ESCALATE label — reusing it there avoids a
  second reimplementation of "offload accuracy charged local + offload
  latency."

## Task 4 — Two-stage routers

- `frame_dataset.load_simulated_rows` already merges the 11 `frame_features`
  columns (plus `scene_complexity` from `load_training_data`) onto every
  simulated row, so stage 1's outputs never need a frame-table merge at
  prediction time — `predict_stage1` just reads `models.feature_columns`
  straight off whatever DataFrame it's given (the per-frame table for
  training, or `simulated_rows` for inference). Confirmed this by re-reading
  `load_simulated_rows`'s implementation rather than assuming a join was
  still needed.
- `Stage1Models`/`Stage2Model`/`RouterResult` each carry the *actual*
  `frame_id`/`file_name` set that reached `.fit()` (`trained_frame_ids`),
  rather than the router recomputing the simulated/train/test filter a
  second time for tests to check against. The two leakage-guard tests then
  assert directly on those returned sets (`train_stage1`'s return for the
  stage-1 guard; `run_router`'s `RouterResult.stage2_trained_frame_ids` for
  the stage-2 guard) instead of re-deriving the filter logic in the test —
  a change to the actual filtering would break both the guard test and any
  reuse of `trained_frame_ids` elsewhere, so it can't silently drift.
- `_StageRegressor`/`_StageClassifier` (the `Pipeline | HistGradientBoosting*`
  unions) needed an explicit `: TypeAlias` annotation
  (`from typing import TypeAlias`) — mypy strict rejects a bare
  `X = A | B` module-level assignment as "not valid as a type" when used
  later as a parameter/return annotation, even though inlining the same
  union directly in a signature (as `router/baseline.py` already does)
  works fine. `TypeAlias` is the fix; a `type X = A | B` (PEP 695) statement
  would also work under `python_version = "3.12"` but wasn't needed here.
- Single-class stage-2 labels (e.g. every train-photo cascade label is
  ACCEPT) make `LogisticRegression`/`HistGradientBoostingClassifier.fit()`
  raise `ValueError` — chose to detect this before fitting
  (`labels.unique()` length check) and short-circuit to a constant pick
  returned for every test row, recorded on `Stage2Model` as
  `constant_pick` with `classifier=None`. `predict_stage2` then branches on
  `classifier is None` instead of trying to fit/predict and catching the
  exception.
- `np.full(n, "ACCEPT", dtype=np.str_)` silently truncates to 1 character
  (`'A'`) because unsized `np.str_`/`str` dtype defaults to `<U0`/`<U1`, not
  the string's actual length — a real footgun, easy to hit here since the
  constant-pick fallback returns a small fixed-length array. Fixed by
  sizing the dtype explicitly from the string itself:
  `np.full(n, pick, dtype=f"<U{len(pick)}")`. By contrast,
  `np.asarray(existing_str_array, dtype=np.str_)` (used for the classifier
  branch, converting `.predict()`'s `object`-dtype output) infers the
  correct width from the array's actual contents and does *not* truncate —
  confirmed both behaviors directly in a REPL before relying on either.
- Built the leakage-guard/determinism/label test fixtures with an
  **alternating gap sign by frame index** (even index → offload better, odd
  → local better) rather than a single unconstrained RNG draw, so that
  *any* contiguous slice used later (the non-simulated pool, or the
  simulated train/test subsets) is guaranteed to contain both classes for
  stage 1's `gap <= 0` classifier and, via fixed local/offload latencies
  chosen so the escalate threshold (`gap > 0.036` at `DEFAULT_LAMBDA =
  0.3`) lines up with the sign, for the cascade label too — avoids flaky
  single-class failures without re-seeding until a mix happens to appear.

## Task 5 — Experiment script (integration)

- **Row-order alignment between `picks` and the scoring DataFrame is not
  automatic**, and got it wrong on the first pass. `router.dataset
  .frame_level_split`'s `test_split` is `simulated_rows.iloc[test_idx]`
  re-indexed 0..n-1, where `test_idx` comes from `GroupShuffleSplit` and is
  *not* guaranteed to preserve `simulated_rows`'s original row order.
  Meanwhile `run_router` internally computes its own test subset as
  `rows[rows["frame_id"].isin(test_frame_ids)].reset_index(drop=True)` —
  i.e. filtered in `simulated_rows`'s *original* order. If the orchestrator
  scores `result.picks` against `frame_level_split`'s `test_split` directly,
  the two can have different row orders while both having the same length,
  so `score_decide_first`/`score_cascade`'s `np.where(picks == ...)` (a
  positional NumPy array, no index) silently pairs each pick with the wrong
  row — no exception, just wrong numbers. Fixed by having the orchestrator
  re-derive its scoring DataFrame the same way `run_router` derives its test
  rows (`simulated_rows[simulated_rows["frame_id"].isin(test_frame_ids)]
  .reset_index(drop=True)`) instead of reusing `frame_level_split`'s
  `test_split`, so both are built by the identical filter over the identical
  source ordering. Only used `frame_level_split` itself to obtain the
  train/test `frame_id` *sets* (which don't care about row order).
- Reused `test_two_stage.py`'s alternating-gap-sign fixture strategy for the
  integration test's Postgres-seeded synthetic dataset (10 frames outside
  `simulation_results` for stage 1's pool, 20 simulated frames — split
  roughly 16 train / 4 test at `test_size=0.2` — with 2 simulation rows
  each). Same reasoning applies unchanged when seeding through
  `datagen.persistence`/`router.feature_store` instead of building
  DataFrames directly: alternating sign by absolute frame index keeps both
  the decide-first label and the (separately, internally computed) cascade
  label balanced across any 80/20 split, so stage 2 exercises its real
  classifier rather than the single-class constant-pick fallback.
- `ExperimentResult` (the orchestration function's return value) is a frozen
  dataclass with a `pd.DataFrame` field (`diagnostics`). Comparing two
  instances with plain `==` blows up with pandas's "truth value of a
  DataFrame is ambiguous" `ValueError`, because a dataclass's generated
  `__eq__` tuple-compares all fields and `bool()`s each pairwise result.
  The determinism test instead compares `diagnostics` via `.equals()` and
  compares everything else (`train_frame_count`/`test_frame_count`, and the
  `routers` list — whose `RouterEvaluation`/`RouterSummary`/`BootstrapResult`
  fields are all `str`/`float`/`bool`, no DataFrame) with ordinary `==`.
- `f"{value:.2%}"` on `float("nan")` (the `headroom_share` formatting in
  `_print_report`) does not raise — it prints `nan%` — so no special-casing
  was needed in the report formatter for the zero-headroom case Task 3
  already produces as `nan`.

## Task 6 — Run the experiment and write findings

- Real-data result is a clean negative, stronger than the earlier baseline
  routers': all four routers' 95% bootstrap CIs (resampling the 100 held-out
  photos) sit entirely *below* zero, not just overlapping it. The earlier
  three baseline routers in `findings.md` were statistical ties with
  always-offload; these four are worse than always-offload with 95%
  confidence. `decide_first_linear` (CI `[-0.0471, -0.0024]`) is the least
  bad; the cascade variants are worst (`[-0.0898, -0.0169]` and
  `[-0.0868, -0.0214]`), plausibly stage 2 overfitting on only 400 train
  photos.
- Two full runs of `../.venv/bin/python -m router.feature_experiment` from
  `training/` against the real Postgres database produced byte-identical
  stdout (`diff` clean) — confirms the seeding (split seed 42, bootstrap
  seed 42, model `random_state=42`) is deterministic end-to-end against real
  data, not just the synthetic integration test.
- `python -m router.baseline` still runs clean and reprints
  always-offload = 0.6812 and utility-regression router = 0.6811 exactly
  matching the existing findings.md prose. Its printed logistic-router avg
  utility is 0.6574 (HistGB 0.6481) — findings.md's prose only ever said
  "~0.68 (comparable, no clear win)" for logistic, which is a loose
  description rather than a precise figure. `git diff main --stat -- training
  /router/baseline.py training/router/dataset.py training/router/analyze.py`
  shows these files are pure additions on this branch (no line changed
  since `main`), so this isn't a regression from this feature's work — the
  original prose was just imprecise, not wrong about the conclusion
  (logistic is comparable to, not clearly better than, always-offload).
- `router.baseline`'s `.joblib` files under `training/router/models/` and
  `feature_experiment`'s own `__pycache__` dirs are correctly covered by the
  root `.gitignore`'s `*.joblib` and standard Python ignore rules —
  `git status --short --ignored training/router/` shows them as `!!`
  (ignored), so running both scripts leaves `git status` showing only the
  intended `training/router/findings.md` edit.
