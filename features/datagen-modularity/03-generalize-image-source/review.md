# Review — Generalize run-simulation's Image Data Source

## Verdict

The implementation does what the surviving spec asks. `run-simulation` now
takes `--annotations`/`--images`/`--dataset` as required arguments, validates
both paths before any database work, and threads the dataset name through
every `scene_complexity` and `simulation_runs` write. The three post-plan
commits (paths moved into `config.py`, download machinery removed, `presets.py`
relocated) landed cleanly: there is no stale import, no reference to
`sync_coco_cache`/`download_missing_images`/`requests` anywhere in `training/`
source or tests, and `datagen.sampling.presets` is gone from every importer.
Verified locally with Postgres reachable: `pytest -q` → **68 passed, 0 failed,
0 skipped** (77 at `d3ddf3a` minus the 9 tests deleted with the download/sync
code), `ruff check .` clean, `mypy .` clean over 40 source files. CLI behavior
spot-checked: `--help` lists all three as required, and `--preset baseline`
alone exits 2 with an argparse error before reaching `get_engine()`.

No blockers. The findings below are three things worth fixing — a missing test
for the feature's headline behavior, incomplete shape validation on the
annotations file, and a set of now-dead COCO defaults that the removal commit
left behind — plus two stale package docstrings and some comment polish.

## Acceptance Criteria

1. **Invoked without `--annotations`/`--images`/`--dataset` → clear error, no
   COCO fallback — MET.** `run_simulation.py:262-279` declares all three
   `required=True`. Observed: exit code 2, `error: the following arguments are
   required: --annotations, --images, --dataset`. `get_engine()`
   (`run_simulation.py:286`) is never reached.

2. **Valid args → scores uncovered images, persists `(<name>, file_name,
   score)`, proceeds through the pipeline — MET.** `run_simulation.py:283-293`
   wires `load_image_index(args.annotations)` and
   `functools.partial(resolve_image_path, images_dir=args.images)` into
   `run_simulation(..., dataset=args.dataset)`; `resolved_dataset`
   (`run_simulation.py:120`) replaces every former `config.DATASET_NAME` use at
   `:124`, `:148`, `:152`, and `:180`. Correct by reading, but no test
   exercises a non-default `dataset` — see S1.

3. **Bad `--annotations` (missing / invalid JSON / no `images` array) → clear,
   specific error before any scoring or DB work — PARTIAL.** The three named
   cases are handled with path-naming messages at `image_source.py:44-62` and
   covered by `test_image_source.py:39-59`; `load_image_index` is called at
   `run_simulation.py:283`, before `get_engine()` at `:286`. But an `images`
   value that exists in the wrong *shape* still escapes as a bare
   `TypeError`/`KeyError` — see S2.

4. **Missing `--images` folder → clear error before any DB work — MET.**
   `_require_images_dir` (`run_simulation.py:230-238`) raises
   `FileNotFoundError(f"No such images folder: {images_dir}")`, called at
   `:282` before `get_engine()` at `:286`. Covered by
   `test_run_simulation.py:78-88`.

5. **`sync-coco-cache` unchanged — SUPERSEDED, not failed.** Commit `6ce53e5`
   deleted the tool, its tests, the `requests` dependency, and its
   `[project.scripts]` entry under explicit user direction after the plan was
   written. The removal is complete and self-consistent: `pyproject.toml:11-16`
   and `:21-26` carry no trace of it, and a repo-wide grep for
   `sync_coco_cache|download_missing_images|fetch_image_bytes|FetchFn|
   COCO_VAL2017_BASE_URL|requests` hits nothing outside historical
   `features/*.md` planning docs. The convention is documented at
   `.claude/coding-guidelines.md:123-127`.

6. **`coco.py` renamed, every importer updated — MET.** Done in the prior reorg
   commit; the module is `datagen/sourcing/image_source.py` and no source or
   test file references `datagen.coco`. (Stale `coco.cpython-312.pyc` /
   `sync_coco_cache.cpython-312.pyc` bytecode remains in `__pycache__/`, but
   it is gitignored and not importable — Python won't load a `__pycache__`
   `.pyc` whose source is gone.)

## Scope

The three planned tasks match the plan's Files Changed table exactly. Beyond
them, the diff contains the three user-directed post-plan changes
(`4189266`, `6ce53e5`, `f0748fc`) and their mechanical follow-through:
`config.py` gained the path constants, `conditions.py:8,15` and
`preview_conditions.py:18` follow `presets.py` to its new home,
`test_conditions.py`/`test_preview_conditions.py`/`test_presets.py` follow the
same import, and `.claude/coding-guidelines.md:123-133` documents the new
no-fetch convention and drops `sync_coco_cache` from the entry-point list.
Nothing in the diff is outside either the plan or those directed changes.
`stub_inference`, `score_complexity`, `preview_sample`, `relabel_run`, and
`ImageRecord.image_id` are all untouched, as the spec required.

## Blockers

None.

## Suggestions

#### S1 — No test exercises a non-default `dataset`, the feature's headline behavior

- **File:** `training/tests/datagen/cli/test_run_simulation.py:117,133,194`
- **Issue:** Every DB-backed test still calls `run_simulation()` without
  `dataset=` and asserts against `config.DATASET_NAME`, so the whole point of
  `--dataset` — that scores and runs land under the *caller's* dataset key
  rather than `coco_val2017` — has zero coverage. The plan deliberately left
  the existing tests alone (reasonable), but never added the positive case. A
  regression here (e.g. one of the four `resolved_dataset` substitutions
  reverting to `config.DATASET_NAME`) would silently write research rows under
  the wrong key and no test would catch it, which is exactly the class of
  failure `.claude/coding-guidelines.md:160-164` says must be tested ("data-gen
  logging").
- **Fix:** Add one test alongside the existing ones that passes
  `dataset="test_dataset_xyz"` and asserts `run.dataset == "test_dataset_xyz"`
  and `session.query(SceneComplexity).filter_by(dataset="test_dataset_xyz")
  .count() == POOL_SIZE`, plus that no rows exist under `config.DATASET_NAME`.
  ~15 lines reusing `_write_fake_pool`/`_make_resolve_image`.
- **Decision:** Accepted — addressed in "Address S1: add test for non-default dataset in run_simulation"

#### S2 — `load_image_index` validates the `"images"` key but not its shape

- **File:** `training/datagen/sourcing/image_source.py:52,58-66`
- **Issue:** The spec asks for a clear error when the annotations file "doesn't
  contain an `images` array **in the expected shape**". Only the key's presence
  is checked. Probed against the current code:
  - top-level scalar (`42`) → `TypeError: argument of type 'int' is not iterable`
  - `{"images": "nope"}` → `TypeError: string indices must be integers`
  - `{"images": null}` → `TypeError: 'NoneType' object is not iterable`
  - `{"images": [{"id": 1}]}` → `KeyError: 'file_name'`
  - `{"images": [1, 2]}` → `TypeError: 'int' object is not subscriptable`

  None of these name the annotations path or say what's wrong, which is the
  opposite of the fail-fast contract the rest of the module honors. Relatedly,
  the `data: dict[str, list[Any]]` annotation at `:52` is unsound — `json.load`
  can return any JSON type, and mypy is only satisfied because the annotation
  asserts otherwise.
- **Fix:** After the parse, check `isinstance(data, dict)` and
  `isinstance(data.get("images"), list)` and raise `ValueError` naming the path
  for each; in the comprehension, raise a `ValueError` naming the path and the
  offending entry index when an entry isn't a dict or lacks `id`/`file_name`.
  Annotate the parse result as `Any` (or `object`) so the narrowing is real.
- **Decision:** Accepted — addressed in "Address S2: validate annotations JSON shape in load_image_index"

#### S3 — The COCO default paths and `run_simulation()`'s implicit pool fallback are now dead, and contradict the spec's "no defaults" rule

- **File:** `training/datagen/config.py:21-23`, `training/datagen/sourcing/image_source.py:37,69`, `training/datagen/cli/run_simulation.py:105-110`
- **Issue:** `DATASET_DIR`/`ANNOTATIONS_PATH`/`IMAGES_DIR` now have exactly one
  consumer: the default parameter values of `load_image_index` and
  `resolve_image_path`. Those defaults in turn are only reachable through
  `run_simulation()`'s `image_records is None` / `resolve_image is None`
  fallbacks at `run_simulation.py:105-110` — which `main()` never takes (it
  always injects both) and which no test takes either. Their last real
  consumer, `sync-coco-cache`, was deleted in `6ce53e5`. So this is leftover
  scaffolding from the pre-feature design, and it directly contradicts spec
  requirement "There is no default dataset, annotations path, or images path
  anymore." Worse, the fallbacks are asymmetric: a caller who passes
  `image_records=` but forgets `dataset=` silently writes that pool's scores
  under `coco_val2017`, mixing two datasets in one `scene_complexity` key —
  the exact collision `--dataset`'s help text says it prevents.
- **Fix:** Make `annotations_path` and `images_dir` required parameters of
  `load_image_index`/`resolve_image_path`, drop the two `None` fallbacks in
  `run_simulation()` (make `image_records`/`resolve_image` required keyword
  args — `main()` and all six integration tests already pass them), and delete
  `DATASET_DIR`/`ANNOTATIONS_PATH`/`IMAGES_DIR` from `config.py` along with
  their explanatory comment block. Keep `config.DATASET_NAME` — it is still a
  live default for `preview_sample.py:72`. If a default pool is wanted for
  interactive use, that is a separate decision worth making explicitly rather
  than inheriting.
- **Decision:** Accepted — addressed in "Address S3: remove dead COCO default paths and run_simulation's unused fallback"

#### S4 — `sourcing/__init__.py` still calls the package "image acquisition"

- **File:** `training/datagen/sourcing/__init__.py:1`
- **Issue:** Reads `"""Image acquisition: annotation-index loading and local
  image-file resolution."""`. "Acquisition" is precisely what `6ce53e5`
  removed, and `image_source.py:10-13` now says the opposite ("acquiring the
  data is the caller's responsibility, not this module's"). First thing a
  reader of the package sees, and it is wrong.
- **Fix:** Reword to something like `"""Image sourcing: COCO-format
  annotation-index loading and local image-file resolution (never fetches)."""`
- **Decision:** — _(pending)_

#### S5 — `sampling/__init__.py` still advertises condition presets

- **File:** `training/datagen/sampling/__init__.py:1`
- **Issue:** Reads `"""Frame/condition sampling: complexity scoring, condition
  presets and sampling, frame sampling."""`, but `presets.py` moved out to
  `datagen/presets.py` in `f0748fc` precisely because a preset is config data,
  not sampling logic. The docstring points at a module that is no longer in
  this package.
- **Fix:** Drop "condition presets and" — e.g. `"""Frame/condition sampling:
  complexity scoring, condition-vector sampling, frame sampling."""`
- **Decision:** — _(pending)_

## Nitpicks

#### N1 — Five-line comment block in `config.py` exceeds the repo's comment rule

- **File:** `training/datagen/config.py:13-17`
- **Issue:** `.claude/coding-guidelines.md:152-156` says "One line, under ~100
  chars. Two lines only when truly needed. No multi-line block comments." This
  is a five-line block plus a further two-line block at `:19-20`. The content
  is genuinely *why*-flavored (it explains why the paths live here rather than
  in `image_source.py`), so it isn't waste — it's just over the stated budget.
- **Fix:** Compress to two lines, or drop it entirely if S3 is accepted (the
  constants go away with it).
- **Decision:** Accepted — resolved for free by S3 (the flagged comment block and the constants it explained were deleted entirely; confirmed `config.py` no longer contains it). No separate commit needed.

#### N2 — `config.py:27` still describes the frame pool as val2017-specific

- **File:** `training/datagen/config.py:27`
- **Issue:** `# Frames sampled (stratified by scene complexity) from the val2017
  pool.` — `FRAME_COUNT` now applies to whatever pool `--images`/`--annotations`
  name, which is the entire point of the feature.
- **Fix:** Replace "the val2017 pool" with "the run's image pool".
- **Decision:** Accepted — addressed directly (trivial wording fix, folded into "Address N1-N3: fix stale val2017/comment wording left after S3").

#### N3 — Two more val2017 assumptions in `run_simulation.py`'s prose

- **File:** `training/datagen/cli/run_simulation.py:70-71,91`
- **Issue:** The `COMPLEXITY_SCORE_FLUSH_BATCH_SIZE` comment justifies 200
  against "the ~5,000-image val2017 pool", and `run_simulation()`'s docstring
  says `image_records`/`resolve_image` "default to the real COCO val2017 pool"
  — technically still true today, but it documents the fallback S3 recommends
  deleting.
- **Fix:** Generalize the batch-size rationale to "the image pool"; update or
  remove the docstring sentence to match whatever is decided for S3.
- **Decision:** Accepted — addressed directly (trivial wording fix, folded into "Address N1-N3: fix stale val2017/comment wording left after S3").

#### N4 — Four CLI docstrings still show `python -m` invocations the guidelines forbid

- **File:** `training/datagen/cli/preview_conditions.py:6` (also
  `preview_sample.py:6`, `score_complexity.py:6`, `relabel_run.py:6`)
- **Issue:** `.claude/coding-guidelines.md:128-133` — edited by this feature —
  says to run these by console-script name, "not `python -m
  datagen.cli.<name>`". Task 2 fixed exactly this in `run_simulation.py:7`,
  leaving the other four inconsistent with both the guideline and their
  sibling. Pre-existing, not introduced here.
- **Fix:** Swap each for its `[project.scripts]` name (`preview-conditions`,
  `preview-sample`, `score-complexity`, `relabel-run`).
- **Decision:** — _(pending)_

#### N5 — `run_simulation.py`'s first line references a dead task numbering

- **File:** `training/datagen/cli/run_simulation.py:1`
- **Issue:** `"""Standalone data-gen simulator script — wires Tasks 3-10
  together.` refers to an earlier feature's plan task numbers that mean nothing
  to a reader of the code. Same for `test_run_simulation.py:3`. Pre-existing.
- **Fix:** Drop the "Tasks 3-10" clause; the docstring's next paragraph already
  describes what it wires together.
- **Decision:** — _(pending)_

## Tests

**Observed run** (Postgres reachable, DB-backed tests executed — not skipped):
`pytest -q` → 68 passed, 0 failed, 0 skipped, in 2.34s. `ruff check .` and
`mypy .` both clean (40 source files). The 77 → 68 delta is exactly the 9 tests
deleted with the download/sync machinery in `6ce53e5` (6 in
`test_image_source.py`, 3 in the deleted `test_sync_coco_cache.py`) — no test
was lost silently.

**Added by this feature:** three `load_image_index` validation tests
(`test_image_source.py:39-59`) and two `_require_images_dir` tests
(`test_run_simulation.py:78-88`). All are `tmp_path`-based with no network or
real-dataset access, matching the file's existing convention.

**Coverage gaps:**
- The `dataset` parameter's effect on persisted rows is untested (S1) — the
  most significant gap, since it's this feature's core behavior.
- `main()`'s argparse wiring has no test; it was verified by hand instead
  (`--help` and the missing-args exit-2 path, both confirmed above). Acceptable
  at the prototype bar — argparse `required=True` is not code worth mocking —
  but it does mean the `functools.partial` binding at `run_simulation.py:284`
  and the `dataset=args.dataset` passthrough at `:292` rest on inspection
  alone.
- Malformed-shape annotations (S2) are untested because they are unhandled; any
  fix for S2 should come with cases for the five shapes listed there.

**Nothing fragile.** The two flush-batching tests (`test_run_simulation.py:215`,
`:272`) assert exact batch sequences `[6, 6, 6, 2]` against a monkeypatched
batch size — tight, but deterministic and deliberately so.

## Recommended Decisions

- **S1** — Accept — the feature's core behavior is untested, and a silent
  dataset-key mixup is exactly the research-output corruption the repo's
  testing bar names as must-cover; ~15 lines.
- **S2** — Accept — the spec explicitly asks for "expected shape" validation,
  the current failures are bare `TypeError`/`KeyError` with no path named, and
  the fix is a handful of lines in the function that already owns this contract.
- **S3** — Accept — `sync-coco-cache`'s removal orphaned these defaults, and
  leaving them lets a caller silently write a foreign pool's scores under
  `coco_val2017`; removing them is what the spec asked for in the first place.
- **S4** — Accept — one-line fix, and the word "acquisition" directly
  contradicts a convention this same feature just wrote into the guidelines.
- **S5** — Accept — one-line fix; the docstring points at a module that left
  the package.
- **N1** — Accept — fold into S3's cleanup; free if S3 is taken.
- **N2** — Accept — one-word fix, and the stale "val2017" undercuts exactly the
  generalization this feature delivered.
- **N3** — Accept — same reason as N2, and it keeps the docstring honest about
  whatever S3 decides.
- **N4** — Decline — pre-existing, outside this feature's file scope, and worth
  doing as one sweep across all four CLIs rather than as a drive-by on the one
  file that happened to appear in this diff.
- **N5** — Decline — pre-existing cosmetic debt from an earlier feature; not
  worth a commit on its own, fold it in the next time that docstring is edited
  for a real reason.
