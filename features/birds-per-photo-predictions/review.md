# Review — Birds Per-Photo Predictions

## Verdict

The implementation does what the spec asked: `bird_predictions` is owned by
`database/`, introduced by an authored-but-unapplied migration `0004`, and the
three-module seam (`predictions.py` / `store.py` / `summary.py`) cleanly
separates GPU work, persistence and table maths. Replace-by-(dataset, split,
model) is real and tested against real Postgres; the summary is test-split-only,
deterministic by explicit tie-breaking, and tracked as one standalone MLflow run;
`findings/birds.md` now carries reproducible committed-checkpoint figures that
are internally consistent (the agreement cells sum to 1.0 within rounding and
reconstruct both accuracies, and every escalated count matches
`n * percent // 100` for n=5,794). I independently confirmed Task 8's gate:
`ruff check` clean in both packages, `mypy` clean in `database/` and exactly the
nine pre-existing `ndarray` errors in `training/coco/`, and `pytest tests/birds`
64 passed.

One blocker: `predictions.main()` loads each checkpoint inside the model loop, so
if `small.pt` exists and `large.pt` does not, the small model's 11,788 rows are
written before the run fails — acceptance criterion 12 says a missing checkpoint
must write no records. The existing test only covers the case where both
checkpoints are absent, so it passes without exercising this. Everything else is
polish: the summary's "no records yet" errors don't name the command that fixes
them (the one requirement-level, non-criterion shortfall), and the two design
smells `learnings.md` flagged for me are both real and both worth fixing, since
the private-name import from the frozen `evaluate.py` is the exact pattern the
mid-execution amendment added `split_labels()` to avoid.

The report's claims hold up where I could check them. The amendment is in the
plan (`23edb5b`) and its scope matches the diff; the `.env` defect is fixed in
both entry points and pinned by a test; the migration is authored only and
`evaluate.py` is untouched by the diff. The two claims I cannot re-verify from
the repo are the 23,576-row count and the agreement with `evaluate.py`'s
0.7798/0.8699 — both need the live database and a GPU, and the captured output
they were checked against lived in a scratchpad that is now gone (see N8).

## Acceptance Criteria

1. **Records for both models across all three splits, 23,576 rows — MET.**
   `predictions.py:113-118` loops both models × `_SPLITS`;
   `test_predictions.py:121-147` asserts one record per model per photo for every
   split against a disposable database. The real 5,400/594/5,794-per-model count
   is recorded in `progress.md` Task 6 and `findings/birds.md:30-33`; not
   re-verifiable from the diff alone.
2. **Each record carries split, photo, true/predicted species, correctness, max
   probability, top-2 margin — MET.** `database/common/models.py:132-149`,
   mirrored column-for-column by
   `database/migrations/versions/0004_create_bird_predictions_table.py:30-45`,
   populated at `store.py:29-41`, round-tripped at
   `database/tests/common/test_models.py:75-86,156-165`.
3. **Record identifies the checkpoint's training run — MET.**
   `predictions.py:115` resolves it per model via `tracking.read_run_id` on the
   sidecar; `store.py:39` persists it; `test_store.py:32` asserts it round-trips.
   The column is nullable by design and a null is accepted silently — see S5.
4. **Re-running replaces only that model and split — MET.** `store.py:22-33`
   deletes exactly (dataset, split, model_name) then inserts in one transaction;
   `test_store.py:36-47` asserts the other split's and the other model's rows
   survive.
5. **Summary computed from the test split only — MET.** `summary.py:21` pins
   `SPLIT = "test"` and `summary.py:139` passes it to `read_predictions`, which
   filters on split (`store.py:52-54`); train/val rows cannot reach the tables.
6. **Agreement breakdown reports all eight figures — MET.** `summary.py:30-48`
   and `_agreement` at `summary.py:73-88`; hand-checked in
   `test_summary.py:40-50`; printed at `summary.py:148-158`.
7. **Confidence separation for both measures, both models — MET.**
   `summary.py:126-132` builds the 2×2 AUC dict, `summary.py:160-164` prints it
   as a model × measure table; `test_summary.py:53-63` hand-checks three of the
   four cells.
8. **Cascade: eight points, both measures, endpoints equal each model — MET.**
   `summary.py:23` (the eight percents), `_cascade` at `summary.py:98-116`;
   `test_summary.py:75-102` hand-checks rows and asserts the 0%/100% endpoints;
   `findings/birds.md:129,136` shows 0% = 0.7798 (small) and 100% = 0.8699
   (large).
9. **Derived accuracy agrees with `evaluate.py` — MET (on recorded evidence).**
   `progress.md` Task 6 and `findings/birds.md:33-36` both state 0.7798/0.8699
   match the independent figures. Requires the live database to re-check.
10. **Recorded against a single run covering both models — MET.**
    `summary.py:204-208` opens one `tracking.run("prediction-summary", ...)` and
    logs agreement, both models' AUCs and every cascade row through `_metrics`
    (`summary.py:178-196`). The run carries no pointer back to the two
    checkpoints — see S6.
11. **Identical output across two runs — MET.** `summary.py:100-102` sorts
    stably on (confidence, numeric photo id), `summary.py:69` orders the join;
    `test_summary.py:111-133` asserts reversed and shuffled input produce
    identical cascade rows and identical formatted text. `progress.md` Task 6
    records byte-identical consecutive runs.
12. **Missing dataset / checkpoint / GPU stops with an actionable message and
    writes no records — PARTIAL.** Dataset: MET, `predictions.py:109` calls
    `split_labels()` before any GPU or checkpoint work and the error names
    `python -m birds.dataset.download` (`test_predictions.py:150-158`). GPU: MET
    (`predictions.py:110`, `test_predictions.py:172-177`). Checkpoint: the
    message is right (`test_predictions.py:161-169`) but "writes no records"
    fails when only the second checkpoint is missing — see B1.
13. **`findings/birds.md` rewritten from this feature's numbers — MET.**
    `findings/birds.md:96-137` is a new reproducible section;
    `findings/birds.md:11-19` and `:21-37` state the figures come from the
    committed checkpoints and name both commands; the unreproducible caveat is
    retained but now correctly scoped to flowers and the pilot
    (`findings/birds.md:56-61`).
14. **No migration run by the feature; `evaluate.py` unchanged — MET.**
    `git diff main...HEAD` does not touch `training/birds/evaluation/evaluate.py`;
    migration `0004` is authored only and its docstring points at who applies it
    (`0004_create_bird_predictions_table.py:6-12`); `progress.md` Task 6 records
    that execution stopped and the user applied it.

## Scope

The diff matches the plan's Files Changed table exactly — all twelve rows,
including the two (`training/birds/dataset/splits.py` and its test) added by the
approved mid-execution amendment in `23edb5b`. The only other files are this
feature's own workflow artifacts (`spec.md`, `implementation.md`, `progress.md`,
`learnings.md`, `implementation-report.md`), which I ignore as process records.

Nothing outside scope. The plan's further deferrals hold: no
`[project.scripts]` entry was added to `training/pyproject.toml`, no index
beyond the primary key, no backfill of any existing table, no COCO change, no
server or app change.

Not in the diff and correctly so: `HANDOFF.md`, which still lists this feature
as the next step to do — that is `refreshDocs`'s job, not this plan's (noted as
N6 only so it isn't forgotten).

## Blockers

#### B1 — A missing second checkpoint writes the first model's records before failing

- **File:** `training/birds/evaluation/predictions.py:113`
- **Issue:** `main()` loads each checkpoint inside the model loop
  (`for name in _MODEL_NAMES: model = _load_checkpoint(name, device)`), so with
  `small.pt` present and `large.pt` absent the small model's three splits —
  11,788 rows — are committed by `record_split` before `_load_checkpoint("large")`
  raises. Acceptance criterion 12 requires that a missing checkpoint "writes no
  records", and this is a realistic state (retraining one model, or recording
  before the large model has finished training).
  `test_predictions.py:161-169` monkeypatches `WEIGHTS_DIR` to a `tmp_path` so
  *both* checkpoints are missing and the run fails on `small` first, which is why
  the suite is green.
- **Fix:** Before the loop, assert both checkpoint paths exist — e.g. iterate
  `_MODEL_NAMES` and raise the same `FileNotFoundError` that `_load_checkpoint`
  raises for the first missing one (a path check, not a load, so GPU memory is
  unaffected). Extend the test to the asymmetric case: one checkpoint present,
  the other absent, and assert the table is still empty.
- **Decision:** Accepted — addressed in "Address B1: check both checkpoints exist before writing records"

## Suggestions

#### S1 — The summary's missing-records errors don't name the command that fixes them

- **File:** `training/birds/evaluation/summary.py:123`
- **Issue:** Running `python -m birds.evaluation.summary` before any records
  exist raises `ValueError("no stored predictions to summarize")`, and running it
  with only one model recorded raises `"small and large models cover different
  photos; re-record both"` (`summary.py:65`). Neither names
  `python -m birds.evaluation.predictions`. The spec's Requirements bullet is
  explicit that "a missing prerequisite stops the run with an actionable message,
  naming the command that fixes it, consistent with how the existing birds entry
  points behave" — and `splits.py:65-70` and `evaluate.py:58-61` both name their
  fixing command. This is the one requirement-level shortfall that no acceptance
  criterion covers.
- **Fix:** Append `"Run \`python -m birds.evaluation.predictions\` first."` to
  both messages, and assert the command string in a test the way
  `test_splits.py:95` does.
- **Decision:** Accepted — addressed in "Address S1: name the fixing command in the summary's errors"

#### S2 — `ENV_PATH` re-derives a path `birds.config` already exports, in the wrong module

- **File:** `training/birds/evaluation/predictions.py:105`
- **Issue:** `ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"`
  counts three `.parent` hops to reach `training/`, although `birds/config.py:9`
  already defines `TRAINING_DIR = Path(__file__).resolve().parents[1]` and
  `predictions.py:14` already imports from that module — the same constant
  `shared/tracking.py:27` uses for the same `.env` file. The fragility this
  creates is then papered over by a test that pins the hop count
  (`test_predictions.py:187-191`). It also sits mid-module between two function
  definitions rather than with `DATASET_NAME` at `predictions.py:20`, and because
  it lives here, `summary.py:17` has to import it from the torch- and
  `evaluate`-importing `predictions` module just to find its own `.env` — which
  undercuts the plan's "summary needs no GPU" seam.
- **Fix:** Put `ENV_PATH = TRAINING_DIR / ".env"` in `birds/config.py` beside
  `WEIGHTS_DIR`, import it in both `main()`s, and drop the hop-count test (or
  move it to `config`).
- **Decision:** Accepted — addressed in "Address S2: move ENV_PATH to birds.config"

#### S3 — `predictions.py` imports two private names out of the frozen `evaluate.py`

- **File:** `training/birds/evaluation/predictions.py:16`
- **Issue:** `from birds.evaluation.evaluate import _MODEL_NAMES, _load_checkpoint`
  reaches into another module's privates — the exact pattern the approved
  amendment added `splits.split_labels()` to avoid for `CubDataset._records` —
  and couples the new module to one the plan deliberately froze, so any future
  refactor of `evaluate.py` silently breaks recording. Two concrete consequences
  are already visible: `test_predictions.py:166` has to monkeypatch
  `evaluate_module.WEIGHTS_DIR` to test a `predictions` code path, and the
  model-name tuple now exists twice (`evaluate.py:16` as `_MODEL_NAMES`,
  `summary.py:25` as `_MODELS`) on top of `config.MODEL_WEIGHTS`' keys.
- **Fix:** Promote `MODEL_NAMES: tuple[ModelName, ...]` to `birds/config.py` and
  have `evaluate.py`, `predictions.py` and `summary.py` all use it; move
  `_load_checkpoint` to a public `load_checkpoint` (its natural home is
  `birds/modeling/build.py`, next to `build_model`) and import it from there in
  both entry points.
- **Decision:** — _(pending)_

#### S4 — `predictions` and `store` import each other, worked around with a local import

- **File:** `training/birds/evaluation/predictions.py:94`
- **Issue:** `store.py:10` imports `PhotoPrediction` from `predictions.py`, so
  `record_split` has to defer `from birds.evaluation.store import write_predictions`
  into the function body to break the cycle. A function-local import is a
  marker that the module boundary is wrong, and it means `record_split`'s
  dependency is invisible at the top of the file.
- **Fix:** Move `PhotoPrediction` into its own small module (e.g.
  `birds/evaluation/records.py`) that both import, which makes the dependency
  one-directional: `predictions` → `store` → `records`. Moving `record_split` and
  `main` into a dedicated entry-point module would also work and matches the
  plan's own description of `predictions.py` as "pure compute, no database".
- **Decision:** — _(pending)_

#### S5 — A null `training_run_id` is recorded silently

- **File:** `training/birds/evaluation/predictions.py:115`
- **Issue:** `tracking.read_run_id` returns `None` when tracking is off or the
  sidecar is absent, and the run then writes 23,576 rows that identify no
  checkpoint, with nothing printed. Traceability is the specific gap the spec
  says made the pilot numbers unreproducible ("recognised as stale after a
  retrain"), and `evaluate.py:132-133` already sets the convention of printing
  `note: {name} results are not attached to a training run.` for exactly this
  case.
- **Fix:** In the model loop, print that same note when `run_id is None`, so a
  run that produces untraceable rows says so.
- **Decision:** — _(pending)_

#### S6 — The tracked summary run carries no link to the checkpoints it summarizes

- **File:** `training/birds/evaluation/summary.py:206`
- **Issue:** The MLflow run's params are only `dataset`, `split` and `photos`, so
  a tracked summary cannot be tied to the two checkpoints behind it or recognised
  as stale after a retrain — which is what the per-record `training_run_id` was
  added for. The column is already in the frame `read_predictions` returns, so
  the information is in hand and discarded.
- **Fix:** Log each model's distinct `training_run_id` as a param or tag (e.g.
  `small_training_run_id` / `large_training_run_id`), and note in the output when
  a model's rows carry more than one — that would mean a partially refreshed
  table.
- **Decision:** — _(pending)_

#### S7 — `split_labels()` re-derives the whole split on every call

- **File:** `training/birds/evaluation/predictions.py:95`
- **Issue:** `record_split` calls `split_labels()` and discards five-sixths of
  the result, so a full run re-reads the three CUB index files and re-runs the
  seeded per-species shuffle seven times (once in `main()`, once per
  model × split), on top of the `_read_index()` that `make_data_loader` does
  internally through `_load_split`. Harmless but wasteful, and it recomputes an
  RNG-derived split repeatedly in the one place that must not drift.
- **Fix:** Resolve `split_labels()` once in `main()` into a
  `dict[Split, list[tuple[str, int]]]` and pass the split's pairs into
  `record_split` as a parameter.
- **Decision:** — _(pending)_

## Nitpicks

#### N1 — Doubled "and" in the table-list docstring

- **File:** `database/common/models.py:8`
- **Issue:** The enumeration now reads "... `model_inference` (...), **and**
  `frame_features` (...), **and** `bird_predictions` (...)" — the new clause was
  appended after the list's existing closing "and".
- **Fix:** Drop the first "and" so the list closes once, before
  `bird_predictions`.
- **Decision:** — _(pending)_

#### N2 — The order-mismatch error reports counts that are equal in the case it fires for

- **File:** `training/birds/evaluation/predictions.py:64`
- **Issue:** The length mismatch is already caught at `predictions.py:48-49`, so
  by the time this error raises, the usual cause is a wrong *order* — and the
  message's `"({len(true_labels)} collected vs {len(expected_labels)} expected)"`
  then prints two identical numbers, which reads like a bug in the message
  rather than a diagnosis. `test_predictions.py:60-64` matches only on "does not
  match", so the misleading half is untested.
- **Fix:** Report the first disagreeing position and its two labels instead of
  the counts (the counts are still useful for the short-loader case at
  `test_predictions.py:72-76`, so keep both, clearly separated).
- **Decision:** — _(pending)_

#### N3 — "last two sections" doesn't point at two sections

- **File:** `findings/birds.md:16`
- **Issue:** The reproducible figures live in one top-level section ("Birds,
  committed checkpoints") with two subsections, and "Interpretation" follows it,
  so "the last two sections" names the wrong thing either way.
- **Fix:** Name the section: "the **Birds, committed checkpoints** section
  below".
- **Decision:** — _(pending)_

#### N4 — The numeric-photo-id assumption is repeated in three places, documented in one

- **File:** `training/birds/evaluation/summary.py:69`
- **Issue:** `ids.astype(int)` appears at `store.py:57`, `summary.py:69` and
  `summary.py:100`; only the first carries the comment explaining why. The table
  keys on `dataset` precisely so other datasets can share it (plan, Approach),
  and the first non-numeric photo id would fail with an opaque pandas cast error
  from inside a sort key.
- **Fix:** Extract the ordering into one small helper used by all three call
  sites, carrying the existing comment, so the assumption lives in one place and
  can raise a clear message.
- **Decision:** — _(pending)_

#### N5 — `predict_photos` takes a device but never moves the model to it

- **File:** `training/birds/evaluation/predictions.py:36`
- **Issue:** The function moves images (`images.to(device)`) but not the model,
  so it relies on the caller having done it — true for `_load_checkpoint`, but
  `evaluate.py:27` (`mean_latency_ms`) takes the same `(model, device)` pair and
  does call `model.to(device)`, so the two neighbours differ (`predictions.py:50` is `model.eval()` alone). A caller following
  the other convention gets a device-mismatch error from inside the loop.
- **Fix:** Either call `model.to(device)` beside the existing `model.eval()`, or
  say in the docstring that the model must already be on `device`.
- **Decision:** — _(pending)_

#### N6 — `HANDOFF.md` still lists this feature as the next thing to do

- **File:** `HANDOFF.md:96`
- **Issue:** Next-step 1 is "Per-photo birds prediction records", which this
  branch delivers, and `HANDOFF.md:111-112` quotes birds' confidence AUC as
  "0.844–0.854", now the pilot's figure — the committed small model is
  0.8324/0.8443 (`findings/birds.md:117`). Out of this plan's scope by design;
  flagged so the next `refreshDocs` catches it rather than the next reader.
- **Fix:** Leave to `refreshDocs`; no change on this branch.
- **Decision:** — _(pending)_

#### N7 — The committed-checkpoint section doesn't say which recipe trained those checkpoints

- **File:** `findings/birds.md:96`
- **Issue:** The only training settings in the file are under "Pilot settings",
  which the new section correctly disclaims — leaving the committed checkpoints'
  recipe unstated. `learnings.md` flags this as unsettled. In fact
  `birds/config.py:40-43` matches the listed recipe exactly (small 30/1e-3/64,
  large 15/1e-4/32), so one line closes the gap honestly.
- **Fix:** Add a sentence to the committed-checkpoint section pointing at
  `birds/config.py`'s `TRAIN_SETTINGS` as the recipe of record, noting it matches
  the pilot's listed settings but was not re-verified against the checkpoints.
- **Decision:** — _(pending)_

#### N8 — The canonical summary output isn't committed anywhere

- **File:** `findings/birds.md:21`
- **Issue:** The figures in the findings file were checked against output
  captured to a session scratchpad, which no longer exists, so re-verifying a
  number without a GPU and the live database means trusting the file against
  itself. The spec's point is reproducibility, and the commands are named, so
  this is a convenience gap rather than a spec gap.
- **Fix:** Note the date and commit of the run that produced the tables, or
  commit the captured summary text beside the findings file so a later run can be
  diffed against it.
- **Decision:** — _(pending)_

## Tests

**Added:** 25 test functions across five files — `test_predictions.py` (10),
`test_summary.py` (10), `test_store.py` (3), `test_splits.py` (+2), and
`database/tests/common/test_models.py` extended to six tables. Verified
independently: `pytest tests/birds` is 64 passed; `ruff check` clean in
`training/` and `database/`; `mypy` clean in `database/` and exactly the nine
pre-existing `coco/` `ndarray` errors in `training/`.

**Strong where it matters.** The parts that would silently corrupt research
output are genuinely covered: the hand-computed probability/margin values
(`test_predictions.py:40-53`), batch-size invariance (`:56-57`), the alignment
guard in all three of its failure modes (`:60-76`), replace-by-(split, model)
against real Postgres (`test_store.py:36-47`), numeric id ordering
(`:50-55`), and the cascade arithmetic row by row including the deliberately
unclamped >1 recovery (`test_summary.py:75-93`). Determinism is tested the right
way — reversed *and* shuffled input must produce identical output, compared on
the formatted text, not just the dataclasses (`test_summary.py:111-133`).

**Gaps.**

- The missing-checkpoint test only covers both checkpoints absent, which is why
  B1 is invisible to the suite. The asymmetric case is the one that loses data.
- `test_predictions.py:187-191` exists only to pin a hop count that S2 would
  make unnecessary. It is the right reflex after the Task 6 defect, but the
  better fix is to remove the duplication it is guarding.
- `separation[("large", "top2_margin")]` is the one AUC cell never asserted
  (`test_summary.py:58-62` covers the other three).
- Nothing exercises `main()` in `predictions.py` end to end with stub models —
  `record_split` is covered and the three error paths are covered, but the
  model × split loop itself is not, which is the other reason B1 slipped through.
- `_metrics`' non-finite filter (`summary.py:196`) is reached by the NaN tests
  only through `summarize`, never asserted on the metrics dict handed to MLflow.

**Fragility:** none of note. `test_summary.py:126-133` leans on
`sample(frac=1, random_state=1)`, which is pinned and fine. The Postgres tests
use the ephemeral-database fixture, within `CLAUDE.md`'s scoped exception.

## Recommended Decisions

- **B1** — Accept — An explicit acceptance criterion fails in a state that
  occurs whenever one model is retrained before the other, the data loss is
  silent, and the fix is a path check plus one test.
- **S1** — Accept — The spec asks for it in so many words, every neighbouring
  entry point already does it, and it is a two-string change.
- **S2** — Accept — Removes a duplicated path derivation, moves the constant to
  the module that already owns `TRAINING_DIR`, and lets `summary.py` stop
  importing the torch-heavy `predictions` module for a filename.
- **S3** — Accept — This is the pattern the amendment was approved to avoid, it
  couples the new code to a module the plan froze, and it is already distorting
  the tests.
- **S4** — Accept — Cheap once S3 is in flight, and a function-local import to
  dodge a cycle will outlive anyone's memory of why it is there.
- **S5** — Accept — One `print`, matching `evaluate.py`'s existing note, and it
  protects the traceability the whole table exists to provide.
- **S6** — Accept — The run ids are already in the frame; logging them is what
  makes a tracked summary auditable after a retrain.
- **S7** — Accept — Small, local, and removes six redundant re-derivations of
  the seeded split from the one code path that must never let the split drift.
- **N1** — Accept — One-word docstring fix in the file that documents the table
  inventory.
- **N2** — Accept — The message fires exactly when its numbers are useless;
  reporting the first disagreeing index costs two lines.
- **N3** — Accept — A wrong cross-reference in the file the write-up draws on.
- **N4** — Accept — One helper, three call sites, and the assumption stops being
  folklore.
- **N5** — Accept — Make it `model.to(device)` to match `mean_latency_ms`; a
  docstring note is the acceptable cheaper option.
- **N6** — Decline — Correctly outside this plan's scope; `refreshDocs` owns
  `HANDOFF.md`, and editing it here would be the scope drift this review is
  meant to catch.
- **N7** — Accept — One sentence, and it converts a `learnings.md` loose end
  into a statement a reader can check.
- **N8** — Decline — The spec's requirement (name the commands) is met, and
  committing generated output invites it to go stale against the database it
  came from.
