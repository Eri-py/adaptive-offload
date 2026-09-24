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
