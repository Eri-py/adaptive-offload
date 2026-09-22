# Review — Real Model Inference for the Data-Gen Simulator

## Verdict

The implementation meets the spec. YOLOv8n (CPU) and YOLOv8x (GPU) run for real. Accuracy is the IoU >= 0.5 same-class match rate against COCO ground truth. Results are cached per (dataset, file_name, model) in a new `model_inference` table and reused across runs. Conditions now change latency only. Checks run during this review: `database/` 1/1 passed, `training/` 93/93 passed, and `ruff` and `mypy` are clean in both packages. I also ran the real builders against six val2017 images; every COCO category name matches `model.names` exactly, so name-based matching is sound.

There is one blocker. The first inference call on each model includes one-time warmup: about 709 ms, against about 15 ms for the same image once warm. That warmup time is written into the permanent cache as the "real" latency for whichever frame the run processes first, and `store_model_inference` never overwrites it. It needs fixing before the first real run fills the cache. Everything else is polish or hardening.

## Acceptance Criteria

1. **Empty cache: every sampled frame gets real YOLOv8n (CPU) and YOLOv8x (GPU) inference, and the results are persisted: MET.** `_compute_missing_model_inference` (`training/datagen/cli/run_simulation.py:261-319`) runs both models on every pool image that is not fully cached. That covers every image that can be sampled, and more (see Scope). Results are flushed through `store_model_inference` (`training/datagen/persistence.py:~118-147`). The real builders are `training/datagen/simulate/inference.py:64-112`: `YOLO("yolov8n.pt")` on `"cpu"` and `YOLO("yolov8x.pt")` on `"cuda"`. `main()` builds each model once (`run_simulation.py:~385-386`). I checked this by hand against real val2017 images. Latency was above 0 and accuracy stayed in [0, 1] and varied between images, but see B1 for the first frame's latency.
2. **Second run: cached values reused, no inference re-run: MET.** The `missing_records` filter (`run_simulation.py:289-294`) skips files that already have both labels. The test is `test_run_simulation_reuses_known_model_inference_on_second_call` (`training/tests/datagen/cli/test_run_simulation.py`): the fakes are called 0 times on the second run and the row count stays at `POOL_SIZE * 2`.
3. **Same accuracy for a frame across every condition; only latency varies: MET.** `apply_condition_overhead` (`inference.py:146-197`) passes both accuracies through unchanged and adds overhead to the real base latency only. The per-frame lookup is at `run_simulation.py:218-219`. Tested by `test_condition_never_changes_accuracy_for_a_given_frame` and `test_apply_condition_overhead_passes_accuracy_through_unchanged`.
4. **Accuracy is the fraction of ground-truth boxes matched by a same-class prediction at IoU >= 0.5: MET.** `score_accuracy` / `_iou` are at `inference.py:200-246`. Ground truth is parsed with the correct `[x, y, w, h]` to `(x_min, y_min, x_max, y_max)` conversion (`training/datagen/simulate/ground_truth.py:665-678`). A frame with no ground truth scores 1.0, as the plan decided. Well unit-tested. (See S2 on crowd annotations.)
5. **`compute_label` runs unchanged on the real values: MET.** The `compute_label` call and its arguments are unchanged (`run_simulation.py:235-241`). `labeling.py` has no code changes; one stale docstring line remains (N3).

Other requirements:
- **Latency overhead formula:** kept the same shape, with the six config coefficients retained (`training/datagen/config.py:33-51`).
- **Synthetic accuracy coefficients:** all nine removed.
- **Progress logging:** the inference pass logs "Computed inference for X/Y images." every batch (`run_simulation.py:311-317`).

## Scope

The diff matches the plan's Files Changed table and Tasks 1–8. Two items fall outside or beside the plan:

- **`.gitignore` `*.pt`:** not in the plan. The orchestrator added it after Task 5 found that weights download into the working directory. The change is sensible and is noted in the implementation report. Note that the pattern is global, not limited to `training/`.
- **Inference runs on the whole pool, not just sampled frames:** the plan chose this deliberately (it mirrors complexity scoring). It goes beyond the spec's "for each sampled frame" wording, but covers it. On full val2017 that is about 5000 × ~45 ms ≈ 4 min of inference per new pool, paid once because of caching. This is acceptable. S1 covers a crash this choice leaves open.

Task 4's success criterion `grep -rn "stub_inference" training/` returns nothing is **not** met: `training/datagen/simulate/labeling.py:3` still names it (N3).

## Blockers

#### B1 — First-call model warmup is cached permanently as a frame's "real" latency

- **File:** `training/datagen/simulate/inference.py:72-82` (also `:98-106`)
- **Issue:** The wall-clock window around the first `model(...)` call also covers one-time lazy initialisation (and CUDA context setup for the offload model). I measured the same image at 708.9 ms local / 705.5 ms offload cold, and 15.2 ms / 17.4 ms warm, which is about 46x. `_compute_missing_model_inference` persists that value, and `store_model_inference` never overwrites an existing row. So the first uncovered frame of every invocation (in sorted val2017 order, `000000000139.jpg`) keeps a latency roughly 700 ms too high in the cache for good. Every future run that samples that frame gets distorted latencies and possibly a flipped label. The learnings file noticed the warmup but treated it as harmless.
- **Fix:** In each builder, after `model.to(...)`, run one warmup inference that is thrown away before returning the closure. A small dummy array is enough, for example `model(np.zeros((640, 640, 3), dtype=np.uint8), device=..., verbose=False)`. If a real run has already filled the cache, delete the affected `model_inference` rows.
- **Decision:** Accepted — addressed in "Address B1: discard model warmup before caching real inference latency"

## Suggestions

#### S1 — Sampling can pick a frame with no inference result, which crashes with a bare `KeyError`

- **File:** `training/datagen/cli/run_simulation.py:181-183`, `:218-219`
- **Issue:** `stratified_sample` draws from `all_complexity`, which holds *every* `scene_complexity` row for the dataset in the DB plus the current pool. Inference results exist only for the current `image_records` plus the cached `model_inference` rows. If complexity was scored for a larger or different pool under the same `--dataset` name, a sampled frame has no inference entry. Examples: an earlier `score-complexity` run over full val2017, followed by `run-simulation` with a subset annotations file. `all_model_inference[frame_id]` then raises `KeyError` after the run row is already created. Under the old stub this was harmless because it never needed the image.
- **Fix:** Pass `stratified_sample` only the complexities of frames that have both labels in `all_model_inference`, and filter to the current pool, e.g. `{f: c for f, c in all_complexity.items() if f in pool_names}`. Or fail early with a clear message naming the mismatch. Add a test that seeds `scene_complexity` with an extra file not in the pool.
- **Decision:** Accepted — addressed in "Address S1: guard sampling against frames with no cached inference"

#### S2 — COCO `iscrowd=1` regions are counted as ground-truth boxes to match

- **File:** `training/datagen/simulate/ground_truth.py:668-678`
- **Issue:** val2017 has 446 crowd annotations spread over 411 images (about 8% of the pool). A crowd box covers a whole group of objects. Standard COCO evaluation ignores these regions, and a detector almost never produces one box matching a crowd box at IoU >= 0.5. So these boxes pull accuracy down on those frames, for both models, and push both toward the same score. The spec says "each ground-truth box", so this follows the spec's wording, but it is almost certainly not what was meant.
- **Fix:** Skip entries with `entry.get("iscrowd", 0) == 1` in `load_ground_truth`, and add a one-line test. Mention the change in the spec or plan as a clarification.
- **Decision:** — _(pending)_

#### S3 — `persistence.py` now imports `ultralytics`/`torch` indirectly

- **File:** `training/datagen/persistence.py:17`, `training/datagen/simulate/inference.py:32-33`
- **Issue:** `persistence` imports `DetectionResult` from `inference`, which imports `ultralytics` at module level. Importing `datagen.persistence` now takes about 0.9 s (measured with `-X importtime`; `ultralytics` alone is 0.73 s). It also ties `relabel_run`, `preview_sample` and every DB test to the heavy CV stack. `inference.py` also mixes three concerns: pure scoring, the overhead formula, and the YOLO builders. That goes against the guideline "one file, one concern".
- **Fix:** Move the YOLO builders and `_to_boxes` into their own module (e.g. `simulate/yolo_inference.py`), leaving `DetectionResult`, `RunInferenceFn`, `score_accuracy` and `apply_condition_overhead` free of `ultralytics`. The simpler option is to move the `ultralytics` imports inside the builder functions.
- **Decision:** — _(pending)_

#### S4 — No test checks that ground truth reaches the inference functions keyed by `image_id`

- **File:** `training/tests/datagen/cli/test_run_simulation.py` (every `run_simulation` call passes `ground_truth={}`)
- **Issue:** The `ground_truth.get(record.image_id, [])` lookup (`run_simulation.py:~301`) is never exercised with real data. If it used `file_name` instead, or dropped the boxes, every test would still pass and every real frame would silently score 1.0. This is exactly the kind of research-output corruption the testing bar asks tests to catch.
- **Fix:** Add one test with a non-empty `ground_truth` dict for some pool `image_id`s. Use a fake whose returned accuracy depends on `len(ground_truth_boxes)`, or that records the boxes it received. Assert that the persisted `model_inference.accuracy` / `simulation_results` values reflect it.
- **Decision:** — _(pending)_

#### S5 — Weight files resolve against the working directory and download silently when missing

- **File:** `training/datagen/simulate/inference.py:72`, `:98`
- **Issue:** `YOLO("yolov8n.pt")` looks for the file in the working directory. If you run `run-simulation` from the repo root, or anywhere other than `training/`, it silently downloads about 136 MB of weights into that directory. That cuts against the guideline "`run-simulation` never fetches data itself … fails clearly rather than trying to top up anything missing". It also means the weights actually used can vary with where the command was run from.
- **Fix:** Resolve the weights from a fixed location (e.g. `Path(__file__).resolve().parents[2] / "yolov8n.pt"`, or `config` constants). Optionally fail with a clear message if the file is missing, instead of downloading it.
- **Decision:** — _(pending)_

#### S6 — Cached inference rows don't record which weights produced them

- **File:** `database/common/models.py:92-104`
- **Issue:** `model_path` records only LOCAL or OFFLOAD. If the weights are ever changed (e.g. to YOLOv8s, or different confidence settings), the old rows are silently reused as if the new model had produced them. The `simulation_runs` config snapshot does not record the model either. That goes against the guideline "every simulation run logs enough to reproduce it".
- **Fix:** Add a `model_name` column (e.g. `"yolov8n"`) to the row, or to `RunConfig`, and filter on it when reading the cache.
- **Decision:** — _(pending)_

## Nitpicks

#### N1 — The two builders are near-duplicates

- **File:** `training/datagen/simulate/inference.py:64-112`
- **Issue:** `build_local_inference_fn` and `build_offload_inference_fn` differ only in the weights file and the device. Each also sets the device twice: `model.to(device)` and `device=` on every call.
- **Fix:** Factor out a `_build_inference_fn(weights: str, device: str)` and have both public builders delegate to it. This is also the single place to add the B1 warmup.
- **Decision:** — _(pending)_

#### N2 — Multi-line comments go against the comment guideline

- **File:** `training/datagen/cli/run_simulation.py:71-81`, `run_simulation.py:221-227`, `training/datagen/simulate/inference.py:27-31`, `training/datagen/config.py:33-39`
- **Issue:** `.claude/coding-guidelines.md` says comments should be one line, two at most, with no multi-line block comments. These are 5–11 line blocks (the flush-size comment grew from 9 to 11 lines, and the config header is 7).
- **Fix:** Cut each to one or two lines stating the why, e.g. `# mypy can't see YOLO in ultralytics' runtime-built __all__.`
- **Decision:** — _(pending)_

#### N3 — Stale `stub_inference` reference left in `labeling.py`

- **File:** `training/datagen/simulate/labeling.py:3`
- **Issue:** The docstring still says "Consumes the four `stub_inference` outputs". Task 4 left this for Task 7, which never picked it up, so the plan's `grep` success criterion fails.
- **Fix:** Reword it to "the four per-row latency/accuracy values from `inference.apply_condition_overhead`".
- **Decision:** — _(pending)_

#### N4 — A partial cache hit re-runs both models and diverges from the persisted value

- **File:** `training/datagen/cli/run_simulation.py:301-306`
- **Issue:** If a file has only one label cached, both models run again. The new value for the already-cached label is used in memory for this run, but the insert skips it, so the DB keeps the old value. This run's `simulation_results` then don't match what a rerun with the same seed would produce. Rare, since it needs a crash between two per-label inserts that are committed together.
- **Fix:** Run only the missing label(s), and take the known one from `known_model_inference`.
- **Decision:** — _(pending)_

#### N5 — Exact float equality in a test

- **File:** `training/tests/datagen/simulate/test_inference.py:91`
- **Issue:** `higher - base == 20.0` depends on floating-point arithmetic cancelling exactly. It passes today with seed 42 but is fragile.
- **Fix:** Use `pytest.approx(20.0)`.
- **Decision:** — _(pending)_

## Tests

- **Added:**
  - `test_ground_truth.py` (9): parsing, bbox conversion, missing image, and file- and key-level errors.
  - `test_inference.py` (13): IoU scoring edge cases, overhead determinism, accuracy passthrough, and direction of the latency effect.
  - `test_persistence.py` (+3): cache round-trip, skipping already-known pairs, empty dataset.
  - `test_run_simulation.py` (+2): accuracy stays constant across conditions; the cache is reused on a second run.
  - `database/tests/common/test_models.py`: extended to round-trip `ModelInference`.
- **Changed:** 10 existing `run_simulation` call sites now inject fakes. Two assertions were adjusted with sound reasons: `resolve_image` is now called `POOL_SIZE * 2` times, and the progress-log check only looks at "Scored" lines.
- **Results:** everything above passes against real ephemeral Postgres (`database/` 1/1, `training/` 93/93).
- **Gaps:**
  - No test passes non-empty `ground_truth` through `run_simulation` (S4).
  - No test covers a complexity-only frame being sampled (S1).
  - No test for `_to_boxes`. Checked by hand here, and the real class names match COCO exactly.
  - No test for the inference-pass progress logging or for the partial-cache path.
- **Real YOLO path:** checked by hand only, as the plan intended. My re-run confirmed plausible values and exposed B1.

## Recommended Decisions

- **B1** — Accept — Two-line warmup fix. Without it, a bad latency that is never overwritten goes into the permanent cache on the first real run.
- **S1** — Accept — Cheap filter. Turns a confusing mid-run `KeyError` (after the run row is created) into correct behaviour for a realistic pool/dataset-name mismatch.
- **S2** — Accept — Matches standard COCO evaluation and removes a systematic accuracy drop on about 8% of frames. Record it as a spec clarification.
- **S3** — Accept — Keeps the DB layer and lightweight CLIs free of torch, and restores one concern per file. Mostly moving code around.
- **S4** — Accept — The single untested path whose breakage would silently corrupt every accuracy value.
- **S5** — Accept — Removes a silent 136 MB network fetch that depends on the working directory, from a tool the guidelines say must never fetch.
- **S6** — Decline — The weights are hardcoded and not expected to change this semester. Revisit if the model choice becomes a variable.
- **N1** — Accept — Removes duplication and gives B1's warmup a single place to live.
- **N2** — Accept — Straightforward guideline compliance.
- **N3** — Accept — A one-line docstring fix that closes an unmet plan success criterion.
- **N4** — Decline — Needs an unlikely partial-commit state, and the effect is limited to one run's rows for one frame.
- **N5** — Accept — A trivial change that removes a fragile assertion.
