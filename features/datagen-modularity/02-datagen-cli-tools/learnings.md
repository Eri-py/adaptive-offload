# Learnings — 02-datagen-cli-tools

## Task 1 — Split presets out of config

- The move was mechanical: `ConditionPresetRanges`/`PRESETS` copied verbatim
  into `training/datagen/presets.py`, removed from `config.py` (which also
  meant dropping the now-unused `from typing import TypedDict` import from
  `config.py` — ruff would have flagged it as unused otherwise).
- A full-repo grep for `PRESETS`/`ConditionPresetRanges` turned up two
  references beyond the task's own `Files` list that still needed fixing:
  `training/tests/datagen/test_conditions.py` (`from datagen.config import
  PRESETS`) and `training/tests/datagen/test_run_simulation.py` (two
  `config.PRESETS[...]` call sites, at the run-assertion in the main
  integration test and in the `_expected_condition_ranges` helper). Both
  were already listed in the task's `Files`, so no scope surprise — just
  confirms the grep sweep step is worth doing even when the plan's file
  list looks complete.
- `stub_inference.py` and `sampling.py` docstrings mention `datagen.config`
  generically but never reference `PRESETS`/`ConditionPresetRanges` — left
  untouched.
- Test split: `test_config.py` keeps only the frame/condition-count shape
  assertion; the preset-shape assertion (`test_every_preset_defines_all_four_ranges`)
  moved to a new `test_presets.py` as-is (not further split — it was already
  one cohesive assertion over all axes/presets). Total test count stayed at
  48 (same as baseline) — nothing added or removed, just relocated.
- `ruff check .` and `mypy .` both clean after the move; full `training/`
  suite passes 48/48 in ~2.3s (real ephemeral-Postgres fixture, no mocks).

## Task 2 — Complexity-scoring CLI

- `score_folder(folder: Path) -> dict[str, float]` is the core function
  (file name -> score, in glob order via `sorted(folder.iterdir())`),
  wrapping `complexity.scene_complexity()`. `main()` is a thin `argparse`
  wrapper with a required `--folder` arg, printing `name\tscore` lines. No
  `get_engine()`/`.env` loading at all — unlike `run_simulation.py`, this
  tool truly has zero DB/config dependency, per the spec's "no dependency
  on any other pipeline stage" requirement.
- Matched extensions case-insensitively (`path.suffix.lower() in
  IMAGE_EXTENSIONS`) rather than exact-case globbing (`*.jpg` etc.) — a
  real-world image folder is not guaranteed to use lowercase extensions,
  and the task's `.jpg`/`.jpeg`/`.png`/`.bmp` list reads as "these image
  types," not "these exact byte sequences."
  `folder.iterdir()` + a suffix filter also composes more simply than
  unioning four separate `glob()` calls (and avoids double-matching on
  case-insensitive filesystems).
- Two clear, distinct error types matching the two failure modes:
  `FileNotFoundError` when the folder itself doesn't exist (checked via
  `Path.is_dir()`, which is `False` for both "missing" and "exists but is a
  file" — either way "no such folder" is the accurate message), `ValueError`
  when the folder exists but the extension filter finds nothing. Both
  messages include the folder path so a failure is actionable without a
  stack trace.
- Test file follows `test_complexity.py`'s no-`import pytest` convention
  (plain `def test_...()`, same mypy/`_pytest`-stub-incompatibility reason
  noted there) and its synthetic-image helpers (`_blank_image`,
  `_checkerboard_image`, built with numpy + written via `cv2.imwrite`),
  trimmed to just the two needed here — no noise-image helper since this
  test isn't asserting anything about noise specifically.
- Smoke-tested the actual `python -m datagen.score_complexity --folder
  <path>` invocation by hand (not just the core function via pytest)
  against a real temp folder with a blank PNG and a random-noise JPG, plus
  a missing-folder case — confirmed the printed `name\tscore` lines and the
  non-zero exit code with a clear traceback on the error path.
- `ruff check .` and `mypy .` clean; full `training/` suite passes 52/52
  (48 baseline + 4 new: scores-every-image, ignores-non-matching-files,
  missing-folder-error, empty-folder-error).

## Task 3 — Stratified-sample preview CLI

- `preview_sample(engine, dataset, frame_count, bucket_count, seed) ->
  list[tuple[str, float]]` is the core function — it returns
  `(file_name, complexity_score)` pairs, not just `list[str]`, even though
  `sampling.stratified_sample` itself only returns file names. Pairing the
  score in at the source (one `known_complexity[file_name]` lookup per
  selected frame, right where the scores dict is already in scope) means
  `main()` doesn't need to re-fetch or re-derive scores to print them "for
  context" per the task description — it just unpacks tuples. This also
  keeps `main()` a truly thin wrapper (`get_engine()` -> call core function
  -> print loop) rather than partially reimplementing the core function's
  logic to get at the scores, which would have been the alternative if the
  core function returned bare file names.
- Error-on-empty-dataset check lives in the core function (`if not
  known_complexity: raise ValueError(...)`), not duplicated in `main()` —
  `main()` calls the core function directly rather than re-checking, unlike
  `score_complexity.py`'s pattern where `score_folder` itself both raises
  and is the sole read path. Same shape here: one function owns the check,
  the CLI just surfaces whatever it raises via the normal traceback/exit
  code (consistent with how `score_complexity.main()` doesn't catch its
  own core function's errors either).
- Test asserts the no-run/no-result-row success criterion by querying
  `SimulationRun`/`SimulationResult` directly via a `Session` (same pattern
  as `test_run_simulation.py`'s `_fetch_results` helper, inlined here since
  it's only used once) rather than through a `persistence` read function —
  there's no `persistence.list_runs`-type read yet at this point in the
  plan (Task 5 adds `get_run_results`, which needs a `run_id` this test
  doesn't have), so a direct `select(SimulationRun)`/`select
  (SimulationResult)` count is the cleanest read available without
  reaching ahead of the plan.
- Confirmed via a direct `pg_database` query against
  `POSTGRES_ADMIN_URL` after the full suite run that no `test_%` database
  was left behind — `ephemeral_postgres_database`'s `finally`-based
  drop held even across this task's new tests.
- `ruff check .` and `mypy .` clean; full `training/` suite passes 55/55
  (52 baseline + 3 new: matches-stratified-sample-directly,
  creates-no-run-or-result-rows, raises-clear-error-for-unscored-dataset).

## Task 4 — Condition-vector preview CLI

- `preview_conditions(preset_name: str, count: int, seed: int) ->
  list[tuple[float, float, float, float]]` is the core function — a lookup
  into `presets.PRESETS` (raising a clear `ValueError` listing valid names
  on a `KeyError`) composed with `conditions.sample_condition_vectors`.
  Genuinely the simplest of the four preview/scoring CLIs so far: no
  `Engine`, no `get_engine()`, no `.env` loading — `main()` is a pure
  argparse-in/print-out wrapper, matching `conditions.py`'s own "no I/O, no
  database access" docstring claim. Unlike `preview_sample.py`, there's no
  `from dotenv import load_dotenv` or `common.db` import at all.
- `argparse`'s own `choices=sorted(presets.PRESETS)` on `--preset` means an
  unknown preset name typed at the CLI is rejected by argparse itself
  (exit code 2, usage message) before `preview_conditions` is ever called —
  the core function's own `ValueError` path only fires for a programmatic
  caller (or a test) that bypasses `main()` and calls `preview_conditions`
  directly with a bad name. Both paths were smoke-tested by hand: `python
  -m datagen.preview_conditions --preset baseline --count 3 --seed 42`
  printed three tab-separated 4-value rows; `--preset not-real` was
  rejected by argparse with the valid-choices list, before the function
  ran at all.
- Test file needs no `postgres_engine` fixture and no `pytest.ini`/conftest
  DB setup at all, unlike every other CLI test file in this feature so
  far — confirmed by running the full suite and seeing this file's three
  tests complete in the same pass with no ephemeral-database fixture in
  their call stack.
- `ruff check .` and `mypy .` clean; full `training/` suite passes 58/58
  (55 baseline + 3 new: matches-sample-condition-vectors-directly,
  matches-for-every-preset, raises-clear-error-for-unknown-preset).

## Task 5 — get_run_results + re-labeling CLI

- `get_run_results(engine, run_id) -> list[ResultRow]` slots into
  `persistence.py` right next to `create_run`/`store_results` — a plain
  `select(SimulationResult).where(run_id == ...).order_by(frame_id)` mapped
  back to `ResultRow` field-by-field (the reverse of `store_results`'s
  `ResultRow` -> `SimulationResult` mapping). Ordered by `frame_id` for the
  same "deterministic output" reason as `get_known_complexity`'s
  `file_name` ordering. Deliberately doesn't distinguish "run exists with
  zero results" from "run doesn't exist" — both return `[]` — per the task
  description; that distinction is the caller's job (here, `relabel_run`'s
  `ValueError`).
- Almost missed the `.env`-loading convention: `preview_sample.py`'s
  established DB-touching-CLI pattern loads `training/.env` via
  `load_dotenv` inside `main()` only (never in the core function), and my
  first draft of `relabel_run.py` skipped this entirely, copying
  `get_engine()` without it. Caught immediately by hand-invoking `python -m
  datagen.relabel_run --run-id <id> --lambda 1.0` against a manually seeded
  run and hitting `RuntimeError: DATABASE_URL is not set` even though
  `training/.env` has it — the pytest fixtures never exercise `main()`
  directly (they call the core function with an injected engine), so this
  gap wouldn't have been caught by the test suite alone. Confirms the value
  of smoke-testing the actual CLI invocation, not just the core function
  via pytest, for every DB-touching CLI in this feature.
- To get a real flip case for the test (and the by-hand smoke test), needed
  a row where offload wins at a low λ but local wins at a higher λ — that
  requires offload to have *both* higher latency and higher accuracy than
  local (a straight comparative advantage on latency in either direction
  won't flip under increasing λ, since larger latency is monotonically
  worse as its penalty weight grows). Landed on local=(100ms, 0.60) vs.
  offload=(800ms, 0.70): OFFLOAD wins at λ=0.1 (0.62 > 0.595...), LOCAL
  wins at λ=1.0 (0.50 > -0.10). A second row with a much larger latency gap
  (local=50ms/0.90 vs. offload=2000ms/0.50) stays LOCAL at both λ values,
  giving an exact expected flip count of 1/2 to assert against — not just
  "at least one flipped."
- Seeded each test row's stored `label` by calling `labeling.compute_label`
  at the stored λ directly in the test setup (rather than hardcoding the
  enum value) — mirrors how the real pipeline computes labels, and it's
  what actually caught that my first flip-scenario arithmetic sketch (done
  by hand in a comment) was consistent with `compute_label`'s real output,
  not just my mental math.
- No new stray database: same `pg_database`-query spot-check as Task 3,
  run after the full suite plus the by-hand CLI smoke test (which seeds and
  then manually cleans up its own run against whatever `DATABASE_URL`
  `training/.env` points at, not an ephemeral fixture database).
- `ruff check .` and `mypy .` clean; full `training/` suite passes 64/64
  (58 baseline + 6 new: get_run_results round-trip, empty-for-no-results,
  empty-for-nonexistent-run, relabel_run reports-flips-correctly,
  persists-nothing, raises-clear-error-for-run-with-no-results).

## Task 6 — Split COCO acquisition: coco.py refactor + cache-population CLI

- `resolve_image_path` lost its `fetch`/`base_url` parameters entirely — it's
  now `resolve_image_path(file_name, *, images_dir=IMAGES_DIR) -> Path`,
  raising `FileNotFoundError` (naming `python -m datagen.sync_coco_cache` in
  the message) on a miss. `download_missing_images(image_records, *,
  images_dir=IMAGES_DIR, base_url=COCO_VAL2017_BASE_URL,
  fetch=fetch_image_bytes) -> list[str]` inherits the old function's
  temp-file-then-rename atomic-write block verbatim (just moved and put
  inside a loop over `image_records`, `continue`-ing past any file whose
  `local_path.exists()` already), and returns the file names it actually
  downloaded (a subset of the input, in input order) rather than paths —
  `sync_coco_cache` only needs counts, and returning names keeps the
  function honest about "which ones did I touch" without forcing every
  caller to also want `Path` objects back.
- Confirmed by reading (not assuming) that `run_simulation.py` line 95
  (`resolve_image if resolve_image is not None else
  coco.resolve_image_path`) never calls `resolve_image_path(..., fetch=...)`
  — it only ever references the function as a default callable value, so
  dropping `fetch` from the signature doesn't break this call site. No edit
  needed there. Likewise confirmed `test_run_simulation.py` never imports or
  calls `coco.resolve_image_path` directly — every test in that file injects
  its own `_make_resolve_image(tmp_path, call_count)` closure instead — so it
  needed no changes either. Ran the full suite specifically watching
  `test_run_simulation.py`'s 7 tests to confirm this rather than taking the
  grep at face value.
- `sync_coco_cache.py` follows `score_complexity.py`/`preview_conditions.py`
  pattern, not `preview_sample.py`/`relabel_run.py`'s: no database access,
  so no `get_engine()` and (per that established split) no `load_dotenv`
  call in `main()` either — this tool genuinely has zero dependency on
  `DATABASE_URL` or any other config, same reasoning as those two DB-free
  CLIs. First draft added a `load_dotenv` call out of habit (copying the
  DB-touching CLIs' `main()` shape) before catching that there was nothing
  in `training/.env` this tool actually needed — removed it and the now-
  unused `dotenv` import.
- Core function `sync_coco_cache(*, annotations_path=coco.ANNOTATIONS_PATH,
  images_dir=coco.IMAGES_DIR, base_url=coco.COCO_VAL2017_BASE_URL,
  fetch=coco.fetch_image_bytes) -> tuple[int, int]` takes all four as
  keyword params with real-path defaults specifically so tests can override
  every one of them (fake annotations file, fake images dir, fake fetch)
  without needing to construct a fake `coco` module or monkeypatch — same
  "core function takes everything as a parameter, `main()` supplies the real
  defaults via `argparse`" split as every other CLI in this feature, except
  here `main()` takes zero CLI arguments at all (there's nothing to
  parametrize — it always syncs the one real COCO cache) so it's just
  `sync_coco_cache()` with no args, printing the two counts.
- Split `test_coco.py`'s original 5 download-behavior tests (all on
  `resolve_image_path`) into `download_missing_images` equivalents (adding
  one new one — downloads-only-the-missing-subset-from-a-mixed-list, since
  the old single-file-at-a-time tests never exercised a mixed list) plus 2
  new fail-clearly tests for `resolve_image_path` (missing file, missing
  images_dir entirely — both raise `FileNotFoundError`). Net test count for
  the file: 6 -> 10, all reusing the same tmp_path/injected-Mock-fetch
  pattern as before, still never touching the real network or
  `training/data/coco/`.
- mypy caught one thing worth noting: `test_sync_coco_cache.py`'s
  `_FAKE_IMAGES` is a `list[dict[str, object]]` (mixed `int`/`str` values),
  so `images_dir / record["file_name"]` failed to type-check (`Path.__truediv__`
  doesn't accept `object`) even though it's always a `str` at runtime — fixed
  with an explicit `str(record["file_name"])` at the one call site that needed
  it, rather than trying to give `_FAKE_IMAGES` a narrower type.
- Smoke-tested `python -m datagen.sync_coco_cache` against the real,
  already-fully-downloaded `training/data/coco/` cache (5,000 images): printed
  `Already cached: 5000` / `Newly downloaded: 0` with no network activity,
  confirming the real annotations file plus a fully-populated cache round-trips
  cleanly. Also smoke-tested `resolve_image_path` against a real but
  nonexistent file name, confirming the `FileNotFoundError` message names
  `python -m datagen.sync_coco_cache`.
- `ruff check .` and `mypy .` clean; full `training/` suite passes 71/71
  (64 baseline - 6 old `test_coco.py` download tests + 10 new `test_coco.py`
  tests + 3 new `test_sync_coco_cache.py` tests). `test_run_simulation.py`'s
  7 tests are among the passing 71 and required zero changes.
