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
