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
