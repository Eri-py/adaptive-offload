# Review — Data-Gen Config Split and CLI Tools

## Verdict

The implementation satisfies the spec. All eight acceptance criteria are met,
the diff matches the plan's Files Changed table exactly (no scope drift), and
`ruff check .`, `mypy .`, and `pytest` (71 passed) are all clean in
`training/` as of this review. No blockers.

On the three points flagged for particular attention:

1. **`coco.resolve_image_path`'s behavior change against the fully-cached
   local pool:** no effect on a real `run_simulation.py` run. The new
   signature (`coco.py:63`, `file_name: str, *, images_dir: Path = IMAGES_DIR`)
   is still structurally compatible with `run_simulation.py:68`'s
   `Callable[[str], Path]` annotation, and the only call site
   (`run_simulation.py:127`) passes exactly one positional argument and never
   supplied `fetch`/`base_url`, so nothing at that site changes. The resolver
   is invoked only for images not already in `known_complexity`
   (`run_simulation.py:124-128`), and for every such image the file exists, so
   the function returns the same path it returned before. The behavior only
   diverges on the genuinely-missing-file case, which is the intended change.
   One consequence worth being aware of (not a defect): on a fresh checkout
   the pipeline now hard-fails at the first uncached image instead of
   self-healing by downloading, so `python -m datagen.sync_coco_cache` becomes
   a mandatory prerequisite. The failure happens in the complexity-scoring
   loop *before* `create_run` (`run_simulation.py:173`), so no partial
   `simulation_runs`/`simulation_results` state is left behind — only
   batched-but-committed `scene_complexity` rows, which are resumable by
   design. `run_simulation.py`'s own module docstring was not updated to
   reflect this (S3).

2. **`relabel_run.py` never persists anything:** confirmed. `relabel_run`
   (`relabel_run.py:31-60`) calls only `persistence.get_run_results` and
   `labeling.compute_label`. `get_run_results` (`persistence.py:89-118`) opens
   a `Session`, issues a single `SELECT`, and never calls `commit()` — the
   context manager rolls back and closes on exit, and no ORM objects escape
   the session (each row is copied into a frozen `ResultRow` dataclass inside
   the block, so there is no later attribute access that could trigger a
   lazy-load or an implicit flush). `main()` (`relabel_run.py:62-91`) only
   builds an engine and prints. `create_run`/`store_results` are not imported.
   `test_relabel_run.py:101` verifies this against real Postgres by
   re-reading the run afterwards and comparing row-for-row, plus a
   `SimulationRun` count check.

3. **Presets split — leftover consumers:** none. A repo-wide grep for
   `config.PRESETS`, `PRESETS`, and `ConditionPresetRanges` (excluding
   `features/` workflow docs) returns hits only in `datagen/presets.py`,
   `datagen/conditions.py:15`, `datagen/preview_conditions.py`,
   `datagen/run_simulation.py:85,87,89,227`, and the corresponding tests — all
   pointing at `datagen.presets`. `datagen/stub_inference.py:14` still imports
   from `datagen.config`, but only for stub-model coefficients, which stayed
   there correctly. I also diffed the moved block against `7b0a652`'s
   `config.py`: the `ConditionPresetRanges` TypedDict and all four presets'
   values are byte-identical; the only delta is a two-line comment hoisted
   into `presets.py`'s module docstring.

## Acceptance Criteria

1. **Presets in their own module, separate from tunables** — MET.
   `training/datagen/presets.py:12` (`ConditionPresetRanges`) and
   `presets.py:21` (`PRESETS`); `config.py` retains only run-shape and
   stub-model tunables and now points at `datagen.presets` in its docstring
   (`config.py:1-10`). Verified value-identical to the pre-split definitions.

2. **Standalone complexity scoring over an arbitrary folder** — MET.
   `score_complexity.py:25` (`score_folder`) globs `folder.iterdir()`,
   filters on `IMAGE_EXTENSIONS` case-insensitively (`score_complexity.py:22`),
   and calls `complexity.scene_complexity` per file. No COCO logic, no DB, no
   `.env`. `test_score_complexity.py:33` runs it against a synthetic
   `tmp_path` folder (not the COCO cache) and asserts a score per image.

3. **Standalone stratified-sample preview, no writes** — MET.
   `preview_sample.py:30` composes `get_known_complexity` with
   `stratified_sample` and returns `(file_name, score)` pairs;
   `test_preview_sample.py:42` asserts it matches `stratified_sample` called
   directly, and `test_preview_sample.py:53` asserts zero `SimulationRun` and
   `SimulationResult` rows exist afterwards against real Postgres.

4. **Standalone condition-vector preview** — MET.
   `preview_conditions.py:22` looks the preset up in `presets.PRESETS` and
   delegates to `conditions.sample_condition_vectors`; no engine, no dotenv,
   no DB import at all. `test_preview_conditions.py:19,27` assert equality
   with a direct `sample_condition_vectors` call for every preset.

5. **Standalone re-labeling from persisted values only** — MET.
   `relabel_run.py:31` reads via `get_run_results` and recomputes with
   `labeling.compute_label` using the row's stored latency/accuracy values;
   nothing re-scores, re-samples, or re-runs stub inference (no imports of
   `complexity`, `sampling`, `conditions`, or `stub_inference` in the module).
   `test_relabel_run.py:81` asserts a correct flip under a changed λ against a
   seeded run.

6. **Missing cache file fails clearly, no silent download** — MET.
   `coco.py:63-77` raises `FileNotFoundError` naming
   `python -m datagen.sync_coco_cache`; the download code is gone from that
   path entirely (there is no `fetch` parameter left to call), and
   `run_simulation.py:95` uses it as the default resolver with no exception
   handling around `run_simulation.py:127`, so the error propagates.
   `test_coco.py:56,64` cover the missing-file and missing-directory cases.
   Note the criterion is vacuous for the complexity-scoring entry point:
   `score_complexity.py` only scores files it actually finds by globbing, so
   it can never encounter a "needed file that is missing" — it does raise
   clearly for a missing folder (`score_complexity.py:29`) and an empty one
   (`score_complexity.py:37`). The failure is unit-tested at the `coco.py`
   level but not at the pipeline level (see S4).

7. **Cache-population entry point downloads exactly the missing files** — MET.
   `coco.py:80-112` skips any `local_path.exists()` and never passes those to
   `fetch`, preserving the temp-file-then-rename atomic write;
   `sync_coco_cache.py:28` composes `load_image_index` with it and returns
   `(already_cached, newly_downloaded)`. `test_coco.py:108` (mixed
   cached/missing list) and `test_sync_coco_cache.py:29,50` cover it,
   including asserting already-cached bytes are untouched.

8. **Full pipeline produces identical rows** — MET, by inspection rather than
   by a literal before/after database comparison. Every computation input is
   provably unchanged: the preset values are byte-identical to the pre-split
   ones, `resolve_image_path` returns the same path for every cached file,
   and `stub_inference.py`, `labeling.py`, `sampling.py`, and `complexity.py`
   are untouched by this diff. `test_run_simulation.py`'s seven tests —
   including the reproducibility test at `test_run_simulation.py:143` —
   required no changes and pass. Worth noting for the record: no actual
   pre-change-vs-post-change run was diffed in the database, so this rests on
   the argument above, which I consider sound.

## Scope

The diff matches the plan's Files Changed table exactly. Every one of the 22
code/test files touched appears in that table, and no file in the table is
missing. Nothing outside `training/datagen/` and `training/tests/datagen/`
was modified — `database/` and `server/` are untouched, as the plan stated.
No new dependencies, no schema changes, no migration. The only other paths in
the diff are the workflow artifacts (`implementation.md`, `progress.md`,
`learnings.md`, `implementation-report.md`), which are out of review scope.

## Blockers

None.

## Suggestions

#### S1 — `get_run_results`'s ordering is not a total order

- **File:** `training/datagen/persistence.py:102`
- **Issue:** `.order_by(SimulationResult.frame_id)` is not deterministic for a
  real run: a run has one row per (frame, condition) pair, so at the default
  500 × 50 shape there are 50 rows sharing each `frame_id`, and Postgres may
  return those 50 in any order between calls. The docstring at
  `persistence.py:92` claims "Ordered by `frame_id` for deterministic output,"
  which overstates what the query guarantees. Flip *counts* are unaffected
  (order-independent), but `relabel_run`'s printed report can come back in a
  different row order for the same run on two invocations. The existing test
  (`test_persistence.py:143`) only uses two rows with distinct `frame_id`s, so
  it can't catch this.
- **Fix:** Add the surrogate key as a tiebreaker —
  `.order_by(SimulationResult.frame_id, SimulationResult.id)` (`id` exists at
  `database/common/models.py`, `SimulationResult.id`, autoincrement) — which
  also restores per-frame insertion order. Adjust the docstring to match.
- **Decision:** — _(pending)_

#### S2 — `relabel_run`'s report can't identify individual rows at real scale

- **File:** `training/datagen/relabel_run.py:89`
- **Issue:** Each printed line is `frame_id`, stored label, recomputed label.
  On a real run that's 25,000 lines in which each `frame_id` appears 50 times
  with nothing distinguishing the condition vectors, so a user who sees a flip
  cannot tell *which* condition flipped — which is the main thing you'd want
  from a λ sweep. `ResultRow` already carries all four condition values
  (`persistence.py:42-45`), so the information is in hand and simply not
  printed.
- **Fix:** Have `relabel_run` return (and `main` print) the condition columns
  alongside the labels — e.g. bandwidth/latency/packet-loss/device-load, or at
  minimum a stable per-row index. A small `NamedTuple` return type would keep
  the widened tuple readable.
- **Decision:** — _(pending)_

#### S3 — `run_simulation.py`'s module docstring still claims the pipeline downloads images

- **File:** `training/datagen/run_simulation.py:9`
- **Issue:** The docstring says the script "downloads/scores any COCO val2017
  pool images not yet covered by the `scene_complexity` table." After this
  change the pipeline never downloads — it fails with `FileNotFoundError` and
  points at `sync_coco_cache`. This is the first thing a reader of the entry
  point sees, and it now describes the opposite of the actual failure mode.
- **Fix:** Reword to "scores any COCO val2017 pool images not yet covered …
  (images must already be cached locally; run `python -m
  datagen.sync_coco_cache` first)".
- **Decision:** — _(pending)_

#### S4 — no pipeline-level test for the missing-image failure

- **File:** `training/tests/datagen/test_run_simulation.py`
- **Issue:** Acceptance criterion 6 names "the main simulator pipeline"
  explicitly, but the fail-clearly behavior is only tested at the `coco.py`
  unit level (`test_coco.py:56,64`). Every `run_simulation` test injects its
  own `resolve_image` closure, so no test exercises the pipeline reaching a
  missing file.
- **Fix:** Add one test that passes
  `resolve_image=lambda name: coco.resolve_image_path(name, images_dir=tmp_path)`
  with a record whose file is absent, and asserts `FileNotFoundError`
  propagates out of `run_simulation`.
- **Decision:** — _(pending)_

## Nitpicks

#### N1 — bool added to an int counter

- **File:** `training/datagen/relabel_run.py:87`
- **Issue:** `flip_count += flipped` relies on `bool` being an `int` subclass;
  it reads as a type error at a glance even though it is valid and passes
  mypy strict.
- **Fix:** `if flipped: flip_count += 1`, or compute the count with a `sum(...)`
  generator the way `test_relabel_run.py:97` already does.
- **Decision:** — _(pending)_

#### N2 — `preview_sample` prints no selected-vs-requested count

- **File:** `training/datagen/preview_sample.py:104`
- **Issue:** When the scored pool is too small to fill every bucket,
  `stratified_sample` comes back short. `run_simulation.py:143-150` logs an
  explicit warning for exactly this case; the preview tool just prints fewer
  lines, leaving the user to count them.
- **Fix:** Print a trailing `Selected {len(selected)} of {frame_count}
  requested.` line after the loop.
- **Decision:** — _(pending)_

#### N3 — base-url test asserts only one of three fetch calls

- **File:** `training/tests/datagen/test_sync_coco_cache.py:69`
- **Issue:** `fetch.assert_any_call(...)` checks a single URL out of the three
  downloads the test triggers, so a bug that mangled the URL for all but the
  first record would pass.
- **Fix:** Assert against `fetch.call_args_list` (or add
  `assert fetch.call_count == 3` plus the other two `assert_any_call`s).
- **Decision:** — _(pending)_

## Tests

**Added:** `test_presets.py` (1), `test_score_complexity.py` (4),
`test_preview_sample.py` (3, real Postgres), `test_preview_conditions.py` (3),
`test_relabel_run.py` (3, real Postgres), `test_sync_coco_cache.py` (3);
`test_persistence.py` gained 3 `get_run_results` tests; `test_coco.py` was
rewritten from 6 to 10 tests for the `resolve_image_path` /
`download_missing_images` split. `test_config.py`, `test_conditions.py`, and
`test_run_simulation.py` were repointed at `datagen.presets`.

I ran `../.venv/bin/pytest -q` in `training/`: **71 passed**, 0 skipped, in
2.44s. `ruff check .` and `mypy .` both clean (34 source files).

Quality is good for the prototype bar: the DB-backed tests use the real
ephemeral-Postgres fixture rather than mocks, the download tests inject a fake
`fetch` and never touch the network or `training/data/coco/`, and the
persistence-side-effect assertions in `test_relabel_run.py:101` and
`test_preview_sample.py:53` directly verify the spec's "must not persist"
requirement rather than asserting on mock call counts. The flip fixture in
`test_relabel_run.py:63-79` derives its stored labels by calling
`labeling.compute_label` rather than hardcoding enum values, which keeps the
test honest about the real formula.

Gaps:

- No pipeline-level missing-file test (S4).
- `get_run_results`'s ordering is only exercised with two rows having distinct
  `frame_id`s, so the non-total ordering in S1 is invisible to the suite.
- No test invokes any CLI's `main()` — the `.env` loading, argparse wiring,
  and print formatting are covered only by the implementer's manual smoke
  tests (documented in `learnings.md`). That's a reasonable prototype-bar
  tradeoff, but it is why the `load_dotenv` omission described in
  `learnings.md` Task 5 escaped the suite; a regression there would still
  escape it today.

## Recommended Decisions

- **S1** — Accept — a one-line `order_by` addition removes real
  non-determinism from a report the research loop reads, and the surrogate key
  is already there.
- **S2** — Accept — without condition context the re-labeling report doesn't
  answer the question it exists to answer at the real 500 × 50 shape; the data
  is already on `ResultRow`.
- **S3** — Accept — the docstring now states the opposite of the actual
  behavior at the pipeline's main entry point; cheap to fix, actively
  misleading if left.
- **S4** — Decline — the resolver's failure is already unit-tested and
  `run_simulation` wraps the call in no `try`/`except`, so a pipeline-level
  test would assert that the language propagates exceptions; low value for the
  prototype bar.
- **N1** — Decline — valid Python that passes mypy strict; changing it is pure
  taste.
- **N2** — Decline — the preview's whole output *is* the selection, so a short
  sample is visible in a way it isn't during a full run; the pipeline warning
  exists for a different reason.
- **N3** — Decline — `base_url` forwarding is a single pass-through parameter
  and the existing assertion covers it; tightening buys nothing real.
