# Learnings — frame-feature-extraction

## Task 1 — `frame_features` table and migration

- `database/tests/common/test_models.py` has exactly one test function that
  round-trips every table in `models.py` in a single Postgres session/fixture
  instance. Adding a new table means extending that same test (rename it to
  keep "all N tables" accurate), not adding a new test function — there's
  only one test file in `database/tests/`, so `pytest` reporting "1 passed"
  after adding a table is expected, not a sign something was skipped.
- Offline render (`alembic upgrade 0002:0003 --sql` from `database/`, via
  `../.venv/bin/python -m alembic ...`) is a good pre-check but only
  validates DDL shape/ordering against the model — it never opens a real
  connection, so it can't catch the enum `create_type` class of bug that
  migration `0002`'s docstring describes. Not relevant here (no enum
  columns), but worth remembering for any future migration that adds one.

## Task 3 — Feature functions

- `np.ndarray` (unparameterized) as a function return/param annotation trips
  mypy strict's `type-arg` check. The existing codebase's fix is to
  annotate with `cv2.typing.MatLike` instead wherever the array is
  image-shaped (`image_stats.py` and its tests do this) — `np.ndarray` bare
  is only tolerated in the 9 pre-existing errors the plan calls out
  (`router/baseline.py`, `router/analyze.py`, two datagen test files); don't
  copy that pattern into new files even though a sibling test file
  (`test_complexity.py`) does it.
- `ultralytics.engine.results.Results.__init__` takes a raw `boxes` tensor
  shaped `[x1, y1, x2, y2, conf, cls]` per row (pixel `xyxy`, not
  normalised) and wraps it in `Boxes` internally — useful for building a
  synthetic `Results` in tests without a real model:
  `Results(orig_img=np.zeros((h, w, 3), uint8), path=..., names={...},
  boxes=torch.tensor([[x1, y1, x2, y2, conf, cls], ...]))`. `boxes.conf` and
  `boxes.xywhn` are then computed from that tensor and `orig_shape`.
- `torch.tensor(...)` defaults to float32, so a `Results` built this way
  gives confidences/areas with ~1e-7 rounding error against the input
  floats — compare rounded in tests, not with exact `==`.
- Real-image verification (`load_model(config.LOCAL_MODEL_WEIGHTS_PATH,
  "cpu")` then `model(path, device="cpu", verbose=False)`, adapted through
  `from_results`, run twice per image) on
  `000000000139.jpg`, `000000000285.jpg`, `000000000632.jpg` from
  `training/data/coco/val2017/`: all three images' confidence features
  matched exactly between the two runs (detection counts 13/1/7).

## Task 5 — Extraction step (integration)

- The plan text's `predict(image_path) -> Results`-like callable is
  ambiguous on its own; per the task's accompanying implementation note it
  actually returns the *sequences* (`tuple[Sequence[float], Sequence[float]]`
  of confidences and normalised box areas), not a `Results` object — that's
  what lets `test_extract_features.py` stub predictions with plain lists
  and never import `ultralytics`/`torch`.
- To keep "time the model call" and "time the confidence derivation"
  separate per the plan, `confidence.py`'s `from_results` was split: a new
  `sequences_from_results(results) -> tuple[list[float], list[float]]`
  holds the `boxes.conf`/`boxes.xywhn` extraction, and `from_results` is now
  just `sequences_from_results` + `confidence_features`. `from_results`'s
  signature and behavior are unchanged, so none of Task 3's tests needed
  updating. `extract_features.py`'s real predictor calls
  `sequences_from_results` directly (inside the timed "model" segment,
  since it's cheap tensor-to-list conversion tied to the model call); the
  orchestration function calls `confidence_features` itself as the
  separately-timed "confidence" segment.
- `extract_features()` computes the missing-image check from the *pending*
  set only (`known_frames - already_stored`), not every known frame — an
  image already covered by a stored `frame_features` row is never touched
  again, so it's fine for it to be absent from `--folder` on a later run
  (e.g. a partial local image cache). Only pending frames' images are
  required to exist.
- Batching (`BATCH_SIZE = 100`) calls `store_features()` per batch inside
  the same loop that does image stats/model/confidence timing — the batch
  flush itself isn't part of any of the three timed segments, matching the
  plan's "times the image statistics, the model call and the confidence
  derivation separately" (three segments, not four).

## Review fix — S1 (timing summary group cost)

- `TimingSummary` gained a `confidence_group_seconds` field
  (`model_seconds + confidence_seconds`) so the confidence-features group's
  full per-frame cost is explicit, not just its two component segments.
  `image_stats_seconds` already *was* the image-features group's full cost
  (one segment), so no new field was needed on that side.
- The printed line's labels were changed from `image_stats=`/`model=`/
  `confidence=` to `image_features=`/`confidence_features (model+derive)=`
  (with `model=`/`confidence=` breakdown kept in parentheses) to match the
  spec's own terminology ("image features" / "confidence features").
- No test in `test_extract_features.py` asserted on `TimingSummary`'s field
  values (only `summary is not None`/`is None`), so adding the field
  required no test changes.
