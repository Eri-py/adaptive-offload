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
