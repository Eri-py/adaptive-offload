# Learnings — 03-generalize-image-source

## Task 1 — Generalize `image_source.py`'s validation and error messages

- Module docstring rewrite keeps the COCO-download helper (`download_missing_images`,
  `fetch_image_bytes`, `COCO_VAL2017_BASE_URL`) explicitly called out as
  "the one COCO-specific piece still living here" rather than pretending
  the whole module is dataset-agnostic — `ANNOTATIONS_PATH`/`IMAGES_DIR`
  defaults and the download helpers are genuinely COCO val2017-specific;
  only `load_image_index`/`resolve_image_path`'s *logic* (given an explicit
  path/dir) works for any COCO-format dataset. `ImageRecord`'s docstring
  ("One COCO val2017 image's id...") was directly inconsistent with that
  generalization, so it was brought in line too ("One image's id and file
  name, from a COCO-format annotations file") — this was in scope per the
  task's own note, not a drive-by.
- `load_image_index` validation order: existence check first
  (`FileNotFoundError`, names the path), then JSON parse wrapped in
  `try/except json.JSONDecodeError` re-raised as `ValueError` (names the
  path, includes the original exception text via `from exc` for the
  underlying cause), then a `"images" not in data` check (`ValueError`,
  names the path). Two distinct exception types for two distinct failure
  modes, per this repo's established convention (missing file vs. bad
  content) — matches the pattern already used in `score_complexity.py`'s
  folder validation from feature 02.
- `resolve_image_path`'s error message dropped both the `"COCO image"`
  framing and the `sync_coco_cache` suggestion — the message is now just
  `Image {file_name!r} not found under {images_dir}.` with no remediation
  suggestion, since once this function is used for a non-COCO dataset
  there's no single correct "go run this command" fix to suggest.
- Grep swept for stale `"COCO image"`/`sync_coco_cache` references across
  `training/`: the remaining hits (`sync_coco_cache.py` itself,
  `test_sync_coco_cache.py`, the module docstring's own mention of
  `python -m datagen.cli.sync_coco_cache`, and `run_simulation.py`'s
  docstring) are all legitimate — `sync_coco_cache` is still the correct,
  real CLI for populating the COCO val2017 cache; only the misleading
  suggestion *inside `resolve_image_path`'s error message* (which fires for
  any dataset, not just COCO) needed to go.
- Test updates: the two existing `FileNotFoundError` tests that matched on
  `"datagen.cli.sync_coco_cache"` now match on the file name instead
  (`"000000000099.jpg"`), which is present in the new message and remains a
  meaningful assertion (confirms the bad path is named). Added three new
  tests for `load_image_index` — missing annotations file, invalid JSON,
  and JSON missing `"images"` — each asserting the specific exception type
  and that the message names the annotations path, reusing the file's
  existing fake-annotations-writing pattern (`tmp_path`-based, no real
  `training/data/coco/` access).
- One `ruff` line-length fix needed after the first pass: the
  `JSONDecodeError` f-string exceeded 100 chars on one line; wrapped the
  `raise ValueError(...)` onto multiple lines.
- Final results: `pytest tests/datagen/sourcing/test_image_source.py -v` —
  13/13 passed (10 baseline + 3 new). Full `training/` suite: 54 passed
  (non-DB tests) + 21 errors, all 21 pre-existing `OperationalError:
  connection to server at "127.0.0.1", port 5432 failed` — these require a
  live Postgres instance, which per CLAUDE.md is never started by the
  agent; unrelated to this change (no DB code touched) and present before
  this task's edits too. `ruff check .` and `mypy .` both clean (42 source
  files, no issues).

## Task 2 — Add required `--annotations`/`--images`/`--dataset` args to `run-simulation`

- `dataset: str | None = None` added to `run_simulation()`'s keyword-only
  params, positioned right after `resolve_image` (before the existing
  run-shape tunables). Resolved via `resolved_dataset = dataset if dataset
  is not None else config.DATASET_NAME`, same shape/position pattern as
  `resolved_frame_count` etc. All four `config.DATASET_NAME` call sites
  inside the function body (`get_known_complexity`, both
  `store_complexity_scores` calls, `RunConfig(dataset=...)`) now use
  `resolved_dataset` instead.
- `_require_images_dir(images_dir: Path) -> None` added as a standalone
  module-level function (not inlined in `main()`), raising
  `FileNotFoundError(f"No such images folder: {images_dir}")` when
  `not images_dir.is_dir()` — directly unit-testable without argparse or a
  DB connection, matching this repo's fail-fast/name-the-bad-path
  convention.
- `main()` gains three required argparse arguments (`--annotations`,
  `--images` both `type=Path`; `--dataset` `type=str`), then calls
  `_require_images_dir(args.images)` before building
  `image_records = image_source.load_image_index(args.annotations)` and
  `resolve_image = functools.partial(image_source.resolve_image_path,
  images_dir=args.images)` (added `import functools`). All three —
  `image_records`, `resolve_image`, `dataset=args.dataset` — are passed
  into the `run_simulation(...)` call. Confirmed `resolve_image_path`'s
  real signature is `(file_name: str, *, images_dir: Path = IMAGES_DIR)`,
  so the `functools.partial` binds `images_dir` as a keyword arg cleanly.
- Module docstring rewritten to show the new invocation
  (`run-simulation --preset baseline --annotations <path> --images
  <folder> --dataset <name>`) and to state that `dataset` is now also
  injectable, not just `image_records`/`resolve_image`. Left the
  `sync_coco_cache` mention intact (still accurate — Task 1's learnings
  already covered why that reference is legitimate). Also tightened
  `main()`'s `ArgumentParser(description=...)` to mention the annotations
  file/images folder/dataset name, since argparse's per-argument `help=`
  text alone doesn't summarize the whole command the way the top-level
  description should.
- Grep swept `training/` and `.claude/coding-guidelines.md` for stale
  `python -m datagen.cli.run_simulation` / bare `run-simulation --preset
  baseline` invocation examples outside this feature's own plan docs (which
  intentionally keep the old text as history) — the only other hit,
  `coding-guidelines.md:127`, uses `run-simulation --preset baseline` only
  to illustrate the console-script-naming convention, not as a full runnable
  invocation, so it was left alone (not in this task's file scope either).
- Two new unit tests in `test_run_simulation.py`:
  `test_require_images_dir_accepts_existing_directory` (passes `tmp_path`,
  no exception) and `test_require_images_dir_raises_for_nonexistent_path`
  (asserts `FileNotFoundError` and that the missing path's string appears in
  the message). No existing test in the file was modified — none of them
  pass `dataset=`, so `resolved_dataset` still resolves to
  `config.DATASET_NAME` and every existing `run.dataset ==
  config.DATASET_NAME` / `filter_by(dataset=config.DATASET_NAME)` assertion
  is unaffected.
- Postgres was checked and confirmed **not reachable**
  (`Connection refused` on `localhost:5432`, both via a raw `/dev/tcp` probe
  and by the DB-backed tests themselves) — per CLAUDE.md this was not
  started. Full `training/` suite: 56 passed (54 pre-existing non-DB passes
  + the 2 new `_require_images_dir` tests) + 21 errors, all the same
  pre-existing `OperationalError: connection to server at "127.0.0.1", port
  5432 failed` as Task 1 — none of the 7 DB-backed tests already in
  `test_run_simulation.py` could be executed to confirm their assertions
  still pass; their source is unmodified (only the import line changed to
  add `_require_images_dir`) and the `resolved_dataset` fallback logic keeps
  their existing `config.DATASET_NAME` expectations intact by construction,
  but this is not the same as an observed green run — flagging this
  explicitly per this feature's Postgres-reachability convention.
  `run-simulation --help` (activated venv) confirmed listing `--annotations`,
  `--images`, `--dataset` as required; `run-simulation --preset baseline`
  alone exits 2 with an argparse "required: --annotations, --images,
  --dataset" error and never reaches `get_engine()`/DB connection code.
  `ruff check .` and `mypy .` both clean (still 42 source files touched by
  mypy, no issues).

## Review fix — S1

- Added `test_run_simulation_stores_rows_under_caller_supplied_dataset` to
  `training/tests/datagen/cli/test_run_simulation.py` (placed right before
  the existing reproducibility test), the first test in the file to pass
  `dataset=` to `run_simulation`. It reuses `_write_fake_pool` /
  `_make_resolve_image` exactly like the neighboring tests and asserts three
  things against a custom `"test_dataset_xyz"` key: `run.dataset ==
  custom_dataset`, `SceneComplexity` row count under that dataset equals
  `POOL_SIZE`, and — the part that would actually catch a `resolved_dataset`
  regression reverting to the default — zero `SceneComplexity` rows exist
  under `config.DATASET_NAME`.
- Postgres was reachable this time (unlike the Task 1 note above where it
  was refused) — ran the real suite, not just a compile check: `pytest -q`
  in `training/` gives 69 passed (previously 68), confirming the new test
  and all 7 existing DB-backed tests actually pass against live Postgres,
  not just by inspection.
- `ruff check .` and `mypy .` both clean in `training/` after the change.

## Review fix — S2

- `load_image_index`'s parse result is now annotated `data: Any` (was
  `dict[str, list[Any]]`, which mypy accepted unconditionally regardless of
  what `json.load` actually returned — an unsound narrowing the finding
  called out). Real narrowing now happens via explicit `isinstance` checks
  in sequence: `isinstance(data, dict)` → `ValueError` naming the path and
  `type(data).__name__` if false; then the existing `"images" not in data`
  key check (unchanged); then `isinstance(data["images"], list)` →
  `ValueError` naming the path and the actual type if false. After that,
  mypy treats `images` as `list[Any]` for the comprehension-turned-loop,
  which is why the annotation swap is what makes the narrowing sound rather
  than just silent — confirmed by `mypy .` passing clean with no `Any`
  leaking past the checks (a manual re-check: without the `isinstance(data,
  dict)` branch, `mypy` still passed today only because the old annotation
  lied to it; the fix means it passes because the checks are real).
- The list comprehension became an explicit `for` loop over
  `enumerate(images)` so each entry can be validated (`isinstance(entry,
  dict)` and `"id"`/`"file_name"` both present) before indexing into it,
  raising `ValueError` naming the annotations path, the `images[<index>]`
  position, and `entry!r` for the offending value — matches the finding's
  requested wording (name the path and the offending entry index).
- Order preserved: existence (`FileNotFoundError`) → JSON parse
  (`ValueError` on `JSONDecodeError`) → top-level-is-dict (`ValueError`,
  new) → `"images"` key present (`ValueError`, unchanged) →
  `"images"`-is-a-list (`ValueError`, new) → per-entry shape (`ValueError`,
  new). All five previously-passing tests in
  `test_image_source.py` (missing file, invalid JSON, missing `"images"`
  key, plus the two `resolve_image_path` tests) needed zero changes — the
  new checks only add failure paths, they don't touch the ones already
  covered.
- Added three new tests to `test_image_source.py`, following the file's
  existing `_write_fake_annotations`/`tmp_path` pattern directly (each
  writes its own malformed JSON rather than reusing the helper, since the
  helper always produces a valid shape): non-dict top-level (`42`),
  non-list `"images"` value (`"nope"`), and an `images` entry missing
  `"file_name"` (`[{"id": 1}]`) — the third asserts the message contains
  `images[0]` to confirm the offending index is actually named, not just
  that *a* `ValueError` was raised.
- One `ruff` line-length fix needed on the first pass: the new
  not-a-dict `ValueError` f-string exceeded 100 chars on one line; wrapped
  it across three lines.
- Postgres was reachable — ran the real suite, not just a compile check:
  `pytest -q` in `training/` gives 72 passed (69 before this fix + 3 new),
  confirming the new tests and all existing DB-backed tests pass against
  live Postgres. `ruff check .` and `mypy .` both clean in `training/`
  after the change (40 source files, no issues).
