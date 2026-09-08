# Review — Data-Gen Simulator

## Verdict

The implementation is complete, well-structured, and matches the plan almost
exactly: all three tables live in `server/common/`, the migration is authored
but unapplied, `training/datagen/` is split one-module-one-concern with pure
functions and injected dependencies, and `ruff`/`mypy --strict` are clean
across both packages (verified). Nine of ten acceptance criteria are met.
Two blockers stand in the way of the data actually being fit for feature 02,
though, and both are the "silently corrupts research output" kind the
coding guidelines single out. First, the default λ is scaled down twice —
once in the constant's stated units, once again by `latency_ms / 1000` in
`labeling.py` — leaving the latency term ~100× weaker than the accuracy
noise, so the win/loss label is ~99% determined by accuracy alone and barely
responds to bandwidth, RTT, or device load at all (measured: on `baseline`,
96% of rows label `OFFLOAD`, and the latency term flips only 5 labels in
4,000). Second, the stratified frame sample is only deterministic when no two
frames share a complexity score; real COCO scores do tie (2 tied pairs in a
1,500-image sample), and `get_known_complexity`'s unordered `SELECT` means
run 2 can order those ties differently from run 1, breaking the reproducibility
criterion. Both fixes are small and local.

## Acceptance Criteria

- **One row per (sampled frame × sampled condition vector), tagged with the
  run id, every field populated** — MET. Crossing loop at
  `training/datagen/run_simulation.py:134-167`, persisted at
  `training/datagen/persistence.py:94-117`; asserted at
  `training/tests/datagen/test_run_simulation.py:121-136`.
- **`simulation_runs` record holds preset name, per-axis ranges, frame count,
  condition-vector count, seed, λ** — MET.
  `training/datagen/run_simulation.py:117-131`, verified field-by-field at
  `training/tests/datagen/test_run_simulation.py:99-113`. (Caveat: the stored
  `frame_count` is the actual sample size, not the configured target — see N1.)
- **Rows from two different-preset runs are never conflated** — MET by
  construction: `run_id` is a UUID PK per invocation with a FK from every
  result row (`server/common/models.py:39-41`, `:59-61`) and `preset_name` is
  on the run record. No test exercises the two-different-presets case (S5).
- **Two runs with the same preset/config/seed produce identical frames,
  condition vectors, and values** — PARTIAL. Condition vectors and stub values
  are deterministic (`training/datagen/conditions.py:36-41`,
  `training/datagen/stub_inference.py:60-64`, per-row seed at
  `training/datagen/run_simulation.py:143`), and the integration test at
  `training/tests/datagen/test_run_simulation.py:185-195` passes — but only
  because its synthetic pool is built to avoid tied scores (its own docstring,
  `:41`, says so). With tied scores the frame sample can differ between runs.
  See B2.
- **Config-module-only changes are reflected under a new run id** — MET. Every
  tunable resolves from `datagen.config` at
  `training/datagen/run_simulation.py:82-92`; `main()` passes no overrides
  (`:183-185`); presets are data in `training/datagen/config.py:77-104`.
- **Scene-complexity values span a wide range** — MET. Scored 1,500 real
  val2017 images with `training/datagen/complexity.py:31`: min 0.0013, median
  0.093, max 0.322, 1,498 distinct values. Genuine spread, no clustering.
- **An already-scored frame is reused, not recomputed** — MET. The skip is at
  `training/datagen/run_simulation.py:98-99`, keyed off the
  `scene_complexity` table read at `training/datagen/persistence.py:52-58`;
  `training/tests/datagen/test_run_simulation.py:173` asserts the second run
  resolves zero images.
- **Sampled condition vectors span each axis's configured range** — MET. LHS
  via `scipy.stats.qmc` at `training/datagen/conditions.py:36-41`; coverage
  asserted at `training/tests/datagen/test_conditions.py:51`.
- **The label matches the configured utility function applied to both paths** —
  MET as literally specified. `training/datagen/labeling.py:16-39` implements
  `accuracy − λ·(latency_ms/1000)` and picks the higher score. The formula is
  right; the default λ fed into it is not (B1).
- **A migration exists but is not applied, defining all three tables** — MET.
  `server/migrations/versions/0001_create_simulation_tables.py:29-66` creates
  `scene_complexity`, `simulation_runs`, `simulation_results` with columns
  matching `server/common/models.py`. Nothing in the branch applies it.

## Scope

Every file in the plan's Files Changed table is present, and every module does
what its row says. Additions beyond the table:

- `server/common/testing.py`, `server/tests/conftest.py`,
  `training/tests/conftest.py` — the ephemeral-Postgres fixture from the
  corrective Task 1b. In scope: user-directed mid-execution, and the plan's
  SQLite-based testing decision was explicitly superseded. `testing.py` living
  in `server/common/` is what the guidelines prescribe for helpers shared by
  `api/` and `tests/`.
- `CLAUDE.md` — the scoped create/drop-test-database exception. In scope:
  explicitly granted by the user, and documented at the point of use.
- `.claude/coding-guidelines.md`, `HANDOFF.md`, `CLAUDE.md` — "harness" →
  "simulator" terminology (commit `0ad402a`), and
  `.claude/commands/generateFeatureSpec.md` — dropping the spec-only PR step
  (commit `849b90c`). Both predate implementation and are workflow/doc changes
  unrelated to this feature. Not harmful, but they ride along in a feature
  branch whose PR is about the simulator; worth knowing they're in the diff.

Nothing touches `server/api/`, the app, or feature 02's territory, as planned.
`training/data/` (the ~1.6GB COCO cache) is correctly gitignored.

## Blockers

#### B1 — Default λ is scaled twice, making latency nearly irrelevant to the label

- **File:** `training/datagen/config.py:27` and `training/datagen/labeling.py:17`
- **Issue:** `config.py:24-27` documents `DEFAULT_LAMBDA = 0.005` as the weight
  for a millisecond-valued latency term ("Latency is in milliseconds ... so
  this is scaled down"), but `labeling.py:17` divides by 1000 again before
  applying it. The effective weight is 5e-6 per ms, so the latency half of the
  utility contributes ~0.001 while the accuracy half differs by ~0.097 — the
  label is decided by accuracy (and its noise) alone. Measured across all four
  presets at 4,000 rows each: the latency term changes the label in 5/4000
  (`baseline`), 22/4000 (`network-stress`), 0/4000 (`device-stress`),
  49/4000 (`degraded-network-idle-device`). `baseline` labels 96% `OFFLOAD`,
  `device-stress` 99.7%. The dataset feature 02 trains on therefore encodes
  almost no relationship between network/device conditions and the right
  routing decision — which is the entire point of the feature.
- **Fix:** Pick λ so the two terms are comparable at realistic latencies —
  e.g. `DEFAULT_LAMBDA = 0.3` with the existing `/1000` (0.3 per second of
  latency, so a 300 ms gap moves utility by 0.09, on par with the accuracy
  gap), and correct the config comment to state the per-second convention.
  Add a test asserting the latency term actually flips labels under the
  default λ, so this can't silently regress. Both are config/test changes the
  spec explicitly permits.
- **Decision:** Accepted — addressed in "Address B1: default lambda scaled twice, latency near-irrelevant to label"

#### B2 — Frame sampling is non-deterministic when complexity scores tie

- **File:** `training/datagen/sampling.py:40`
- **Issue:** `sorted(scores.items(), key=lambda item: item[1])` sorts on score
  only. Python's sort is stable, so tied frames keep their dict-insertion
  order — and that order differs between runs: run 1 builds the mapping from
  `new_scores` in annotation-file order, run 2 from
  `get_known_complexity`'s `SELECT` with no `ORDER BY`
  (`training/datagen/persistence.py:55-57`), whose row order Postgres does not
  guarantee. When a tied pair straddles a `rng.choice` draw, the two runs
  select different frames. Ties are real, not theoretical: 2 tied pairs among
  1,500 scored val2017 images (~7 expected across the full 5,000 pool), and
  with 100-of-1000 drawn per bucket each tied pair has a ~18% chance of
  splitting. Reproduced directly: the same score mapping in reversed insertion
  order yields a different sample (`img_042.jpg` vs `img_041.jpg`). This
  breaks the reproducibility acceptance criterion, silently, and the
  integration test can't catch it because its fixture pool is built to avoid
  ties (`training/tests/datagen/test_run_simulation.py:41`).
- **Fix:** Make the sort total: `key=lambda item: (item[1], item[0])`. Add
  `.order_by(SceneComplexity.file_name)` to `get_known_complexity` as a second
  line of defence, and add a sampling test with a deliberately tied pair fed
  in two different insertion orders.
- **Decision:** Accepted — addressed in "Address B2: frame sampling non-deterministic on tied complexity scores"

## Suggestions

#### S1 — The dataset is never persisted on runs or results, so feature 02's join must hardcode it

- **File:** `server/common/models.py:53-74`, `training/datagen/run_simulation.py:123-130`
- **Issue:** `scene_complexity` is keyed `(dataset, file_name)` — deliberately,
  so other datasets can be added without a schema change (plan Task 1) — but
  neither `simulation_results` nor `simulation_runs` records which dataset a
  row's `frame_id` belongs to. `config.DATASET_NAME` exists but is only used
  at write time. Feature 02's training query has to hardcode
  `dataset = 'coco_val2017'` to join, and the day a second dataset is added,
  existing rows are unattributable.
- **Fix:** Add a `dataset` column to `simulation_runs` (populated from
  `config.DATASET_NAME`) and to the migration — one column, and the join
  becomes unambiguous. Cheaper now than after rows exist.
- **Decision:** Accepted — addressed in "Address S1: add dataset column to simulation_runs"

#### S2 — The full-pool complexity pass persists nothing until it finishes

- **File:** `training/datagen/run_simulation.py:96-103`
- **Issue:** The loop scores every not-yet-known pool image into an in-memory
  dict and only calls `store_complexity_scores` after the loop completes. On
  the first real run that's 5,000 images including any downloads; a network
  error, a corrupt file, or a Ctrl-C at image 4,999 discards all of it and the
  next run starts from zero. It also reads against the guidelines' "writes each
  row straight to its Postgres table as it's produced".
- **Fix:** Flush to Postgres in batches (e.g. every 200 images) inside the
  loop. `store_complexity_scores` already skips existing file names, so a
  resumed run picks up exactly where it left off.
- **Decision:** Accepted — addressed in "Address S2: flush complexity scores in batches during scoring pass"

#### S3 — Downloaded images are written non-atomically and a truncated file is cached forever

- **File:** `training/datagen/coco.py:81`
- **Issue:** `local_path.write_bytes(image_bytes)` writes in place, and
  `resolve_image_path` treats any existing path as a valid cache hit
  (`:76-77`). An interrupted or partial write leaves a truncated JPEG that
  every future run accepts without re-fetching; `cv2.imread` will often decode
  it partially rather than fail, so it silently becomes a wrong complexity
  score. On the user's slow connection, a 5,000-image first run is exactly
  where this happens.
- **Fix:** Write to `local_path.with_suffix(local_path.suffix + ".part")` and
  `Path.replace()` into place once the write completes.
- **Decision:** Accepted — addressed in "Address S3: write downloaded images atomically"

#### S4 — No artificial delay is applied, though the spec says conditions are simulated by delay

- **File:** `training/datagen/stub_inference.py:66-78`
- **Issue:** The spec requires "Network and device conditions are simulated
  in-process (artificial delay applied based on each condition vector's
  bandwidth/latency/packet-loss or device-load value)". The implementation
  models latency analytically and never sleeps. That is the better engineering
  call — sleeping through 25,000 rows would add hours of wall clock and change
  nothing about the data — but it's a real divergence from the spec text that
  nothing records.
- **Fix:** Amend the spec requirement to say latency is modeled analytically
  rather than by real delay, and note why in `learnings.md`. No code change.
- **Decision:** Accepted — addressed in "Address S4: amend spec, latency modeled analytically not via real delay"

#### S5 — No test covers the two-different-presets attribution criterion

- **File:** `training/tests/datagen/test_run_simulation.py:139`
- **Issue:** The reproducibility test runs the same preset twice. The
  acceptance criterion about two runs on *different* presets never being
  conflated has no coverage, and it's the criterion protecting against the most
  costly failure mode (a mixed-preset training set that looks fine).
- **Fix:** Add a test that runs `baseline` then `network-stress` against the
  same engine and asserts each run's rows carry the right `run_id`, that
  neither run's row count leaks into the other, and that each run record's
  `condition_ranges` matches its own preset.
- **Decision:** Accepted — addressed in "Address S5: add two-different-presets attribution test"

## Nitpicks

#### N1 — A short sample is recorded as the run's frame count with no warning

- **File:** `training/datagen/sampling.py:51-52`, `training/datagen/run_simulation.py:125`
- **Issue:** When a bucket has fewer items than its share, `stratified_sample`
  takes what's there and returns a list shorter than `target_count` (documented
  at `:34-37`). `run_simulation` then stores `frame_count=len(sampled_frames)`,
  so a run that quietly produced 460 frames instead of 500 looks correct in the
  run record. Can't trigger at the real 500-of-5000 scale, but it makes a
  misconfiguration invisible.
- **Fix:** Log a warning when `len(sampled_frames) < resolved_frame_count`,
  noting both numbers.
- **Decision:** Accepted — addressed in "Address N1: warn when sampled frame count falls short of target"

#### N2 — Avoidable `type: ignore[literal-required]` in the condition sampler

- **File:** `training/datagen/conditions.py:39-40`
- **Issue:** Iterating `_AXES` to index the `TypedDict` needs two
  `# type: ignore` comments. `run_simulation.py:117-122` solves the identical
  problem by naming the literal keys explicitly, and comments why.
- **Fix:** Build `lower_bounds`/`upper_bounds` from the four literal keys, as
  `run_simulation.py` does, and drop both ignores.
- **Decision:** Accepted — addressed in "Address N2: drop avoidable type: ignore comments in conditions.py"

#### N3 — The label column stores enum names, not the `local`/`offload` values

- **File:** `server/common/models.py:71`, `server/migrations/versions/0001_create_simulation_tables.py:62`
- **Issue:** `Enum(Label)` persists member *names*, so the column holds
  `'LOCAL'`/`'OFFLOAD'` while the spec, config, and `Label`'s values all say
  `local`/`offload`. Feature 02 writes raw SQL against this column and will
  have to match the uppercase form. Harmless once known, surprising if not.
- **Fix:** Either pass `values_callable=lambda e: [m.value for m in e]` to
  `Enum(...)` (and match it in the migration), or leave as-is and note the
  stored form in `learnings.md` for feature 02.
- **Decision:** Accepted — addressed in "Address N3: document that label column stores enum names" (schema left as-is per the reviewer's own recommendation — documentation is sufficient)

#### N4 — No index on `simulation_results.run_id`

- **File:** `server/migrations/versions/0001_create_simulation_tables.py:49-66`
- **Issue:** Every feature-02 query filters by `run_id` over 25,000 rows per
  run, growing with each run. Postgres does not index FK columns automatically.
- **Fix:** Add `index=True` to the mapped column and an `op.create_index` in
  the migration — it's unapplied, so it can still be edited in place.
- **Decision:** — _(pending)_

#### N5 — No progress output during the first-run 5,000-image scoring pass

- **File:** `training/datagen/run_simulation.py:97-101`
- **Issue:** The first real invocation scores (and possibly downloads) 5,000
  images in a silent loop before printing anything; the only output is
  `Created run <id>` at the very end. Hard to tell a slow run from a hung one.
- **Fix:** Print a line every N images with the count scored so far.
- **Decision:** — _(pending)_

## Tests

40 tests were added across the two packages, and the split is right: pure
functions get unit tests, everything touching the schema gets a real ephemeral
Postgres database. I ran the 33 non-DB tests (all pass, 0.51s) plus
`ruff check` and `mypy --strict` on `training/` — all clean. The DB-backed
tests need a running Postgres, which I did not start.

Coverage is genuinely good where it matters: determinism and per-axis range
coverage for both samplers, monotonicity for all three stub-model directions,
the labeling formula including exact and near ties, complexity-score
de-duplication and per-dataset scoping, and an end-to-end run asserting exact
row counts, full field population, FK linkage, and zero re-scoring on a second
run.

The gaps line up exactly with the two blockers, which is why neither was
caught:

- **Nothing tests the default λ.** `test_labeling.py` passes hand-chosen λ
  values (0.5, 1.0, 0.01) that make the latency term meaningful, so the
  formula is verified while `config.DEFAULT_LAMBDA`'s inertness is not. A test
  asserting the label distribution isn't degenerate under the shipped defaults
  would have caught B1 immediately.
- **The fixture pool is built to avoid tied scores**
  (`test_run_simulation.py:34-54`, docstring at `:41`). That's a reasonable
  choice for a happy-path test, but it means the reproducibility test can only
  pass — it structurally cannot exercise B2. A tie case belongs in
  `test_sampling.py`, fed in two different insertion orders.
- No two-different-presets test (S5), and no test that a short sample is
  detectable (N1).

Nothing in the suite is fragile — no sleeps, no network, no shared state
between tests (the fixture is function-scoped, giving each test its own
database), and the ephemeral database is dropped in a `finally` so a failing
test can't leak one.

## Recommended Decisions

- **B1** — Accept — the label column is the feature's whole output, and right
  now it barely responds to the condition axes; the fix is a constant plus a
  comment, both of which the spec explicitly says live in config.
- **B2** — Accept — one-line total-ordering fix that restores a stated
  acceptance criterion, with ties confirmed present in the real pool.
- **S1** — Accept — one column, and adding it after rows exist means either a
  backfill or permanently ambiguous history.
- **S2** — Accept — the first real run is the longest operation in the project,
  and losing it to a transient error costs hours on this connection.
- **S3** — Accept — cheap, and a silently corrupt cached image produces a wrong
  complexity score that propagates into frame selection and every stub accuracy
  value derived from it.
- **S4** — Accept — spec text and implementation should not disagree on the
  record; the code is right, so amend the spec.
- **S5** — Accept — cross-run contamination is the failure mode that would
  quietly invalidate feature 02's results, and the test is a short one given
  the fixtures already exist.
- **N1** — Accept — a one-line warning, and it converts an invisible
  misconfiguration into a visible one.
- **N2** — Accept — the repo already has the cleaner pattern two files over;
  no reason to keep two suppressions.
- **N3** — Accept — but the documentation half is sufficient; feature 02 only
  needs to know which form is stored.
- **N4** — Decline — at 25,000 rows per run a sequential scan is milliseconds,
  and the prototype bar doesn't call for index tuning ahead of a measured
  problem. Revisit if run counts grow.
- **N5** — Accept — trivial, and it makes the one genuinely long-running
  operation observable.
