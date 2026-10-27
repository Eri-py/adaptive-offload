# Birds Per-Photo Predictions — Implementation Plan

## Summary

Record both bird models' per-photo predictions and confidence for all 11,788
CUB-200-2011 photos across the three splits into a new Postgres table, then
derive the agreement, confidence-separation and cascade tables from those
records on the test split and rewrite `findings/birds.md`'s comparison and
cascade sections from that output.

## Approach & Key Decisions

- **Two entry points, one per deliverable.** `python -m birds.evaluation.predictions`
  writes the records; `python -m birds.evaluation.summary` reads them back and
  prints the tables. This matches the existing birds convention of a module with
  a `main()` (`birds.train`, `birds.evaluation.evaluate`, `birds.dataset.download`)
  rather than a registered console script, which `training/pyproject.toml`
  reserves for the COCO CLI tools. Splitting them means a re-derived summary
  never re-runs 11,788 inferences.
- **A three-module seam in `birds/evaluation/`:** `predictions.py` turns a model
  plus a loader into records (pure compute, no database), `store.py` owns the
  reads and writes, `summary.py` owns the tables. This keeps the parts that need
  a GPU, a database, and neither respectively testable in isolation — the store
  against the ephemeral-Postgres fixture, the other two against stub models and
  hand-built records.
- **Plain `String` columns for the split and model name, not Postgres enums.**
  `database/` must not import from `training/`, so the `ModelName`/`Split`
  `Literal` types in `birds/config.py` and `birds/dataset/splits.py` stay on the
  training side. Strings also sidestep the `postgresql.ENUM` pitfall migration
  `0003`'s docstring flags from migration `0002`. The table keys on `dataset`
  like its three siblings in `database/common/models.py`, even though this spec
  serves one dataset.
- **Primary key is (dataset, photo, model), with split as an ordinary column.**
  A photo belongs to exactly one split, so putting split in the key would permit
  contradictory rows; keeping it a column still makes the required
  replace-by-(model, split) a single delete.
- **Deterministic tie-breaking is explicit, not incidental.** The cascade sorts
  photos by confidence, and equal confidences would otherwise order arbitrarily
  and break the spec's identical-output requirement, so every sort breaks ties on
  the photo identifier. Loaders run unshuffled over the existing seed-42 split.
- **The executor must stop for the migration.** Per `CLAUDE.md`'s Infrastructure
  rule, no task runs the migration or touches the real database; Task 6 pauses
  and asks the user. Tests need no migration — `common.testing`'s fixture applies
  `Base.metadata.create_all`, so adding the model in Task 1 is enough for every
  test task.
- **`birds/evaluation/evaluate.py` is not touched.** Its test-split accuracies
  instead serve as the independent cross-check that the derived accuracies agree.

## Out of Scope

Carried from the spec: no change to `evaluate.py`'s behaviour; no per-photo or
device latency and no cost model for escalation; no escalation threshold, router
or cascade policy; no latency-budget evaluation or latency sweep; no retraining
or change to either model, the recipe, or the split definition; no change to
attempt 1 (COCO) code, tables or findings; nothing phone-side or server-side; no
stored images or full probability vectors; no second dataset; no running of
migrations and no database create/drop.

Further deferred by this plan: no console-script entry in
`training/pyproject.toml`; no backfill or migration of any existing table; no
index beyond the primary key.

## Dependencies and Configuration

- **New migration:** `database/migrations/versions/0004_create_bird_predictions_table.py`,
  authored but not applied. **The user runs it** (`CLAUDE.md`, Infrastructure).
- **New packages:** none. `torch`, `pandas`, `scikit-learn` and `sqlalchemy` are
  already dependencies of `training/`.
- **Configuration:** none new. `DATABASE_URL` (via `common.db.get_engine`) and
  `POSTGRES_ADMIN_URL` (test fixture) are already in use.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `database/common/models.py` | edit | Add the `BirdPrediction` table model; update the module docstring's table count | `database/` owns every table model |
| `database/migrations/versions/0004_create_bird_predictions_table.py` | add | Authored migration matching that model | Schema changes go through Alembic |
| `database/tests/common/test_models.py` | edit | Round-trip coverage for the new table | Matches existing per-table coverage |
| `training/birds/evaluation/predictions.py` | add | Per-photo records from a model and loader, plus the recording entry point | Pure compute, no database |
| `training/birds/evaluation/store.py` | add | Write records with replace semantics; read them back for the summary | Single owner of this table's access |
| `training/birds/evaluation/summary.py` | add | Agreement, confidence-separation and cascade tables, printed and tracked | Reads records; needs no GPU |
| `training/tests/birds/evaluation/test_predictions.py` | add | Record computation and entry-point error paths | Would silently corrupt research output |
| `training/tests/birds/evaluation/test_store.py` | add | Write, read and replace against real Postgres | Persistence is on the testing bar |
| `training/tests/birds/evaluation/test_summary.py` | add | Table maths on hand-computed records, and determinism | The numbers that get published |
| `findings/birds.md` | edit | Replace the pilot comparison and cascade sections | Spec requirement |

## Tasks

### Task 1 — Table model and migration

- **Objective:** Add the `BirdPrediction` table model and its authored migration.
- **Files:** `database/common/models.py`,
  `database/migrations/versions/0004_create_bird_predictions_table.py`,
  `database/tests/common/test_models.py`
- **Details:** Add a model following the conventions of the existing
  `FrameFeatures`/`ModelInference` classes in the same module (one-line class
  docstring, `_utcnow` default for the timestamp). Columns: `dataset`,
  `photo_id`, `model_name` (the composite primary key, all `String`), `split`
  (`String`), `true_label` and `predicted_label` (`Integer`), `is_correct`
  (`Boolean`), `max_probability` and `top2_margin` (`Float`),
  `training_run_id` (`String`, **nullable** — `shared.tracking.read_run_id`
  returns `None` when tracking is off or the sidecar is absent), and
  `computed_at`. Update the module docstring, which currently says "Five
  tables". Author migration `0004` with `down_revision = "0003"`, mirroring
  `0003`'s docstring shape including the note on who applies it; no enum
  columns, so migration `0002`'s `postgresql.ENUM` pitfall does not apply.
- **Success criteria:**
  - The new model's columns and primary key match the migration exactly
  - `database/tests/common/test_models.py` round-trips a row through the
    ephemeral-Postgres fixture and passes
  - The migration is authored only — nothing in this task applies it

### Task 2 — Per-photo record computation

- **Objective:** Turn a model and an unshuffled loader into per-photo records.
- **Files:** `training/birds/evaluation/predictions.py`,
  `training/tests/birds/evaluation/test_predictions.py`
- **Details:** Produce, per photo in loader order, the true label, the top-1
  predicted label, whether it is correct, the highest softmax probability, and
  the margin between the two highest probabilities. Photo identifiers come from
  `birds.dataset.splits.split_ids()`, whose order matches an unshuffled loader
  over the same split; assert that alignment rather than assuming it (compare
  the collected label sequence against the split's labels). Use
  `birds.dataset.splits.eval_transform(name)` so each model gets its own
  preprocessing, and `torch.inference_mode()`. No database access in this task.
- **Success criteria:**
  - Hand-computed logits through a stub `nn.Identity`-style model yield the
    expected predictions, correctness, max probability and top-2 margin
  - A deliberately misaligned label sequence raises rather than storing
    mismatched rows
  - Tests need neither a GPU nor the dataset

### Task 3 — Record store

- **Objective:** Persist records with replace semantics and read them back.
- **Files:** `training/birds/evaluation/store.py`,
  `training/tests/birds/evaluation/test_store.py`
- **Details:** A write function takes an engine, the dataset name, split, model
  name and the records, deletes the existing rows for exactly that (dataset,
  split, model_name) triple, inserts the new ones, and commits — following the
  `Session(engine)` usage in `training/coco/datagen/persistence.py`. A read
  function returns the stored rows for a given split as a `pandas` DataFrame,
  ordered by photo identifier so downstream output is deterministic. Import the
  table model from `common.models` and the engine from `common.db`; do not
  redefine the table.
- **Success criteria:**
  - Writing then reading returns the same records
  - Re-writing one (split, model) replaces only those rows and leaves other
    splits' and the other model's rows untouched
  - Tests use the `postgres_engine` fixture from `training/tests/conftest.py`

### Task 4 — Recording entry point

- **Objective:** Wire computation and storage into a runnable entry point over
  both models and all three splits.
- **Files:** `training/birds/evaluation/predictions.py`,
  `training/tests/birds/evaluation/test_predictions.py`
- **Details:** A `main()` that calls `birds.modeling.build.require_cuda()`,
  loads each checkpoint the way `birds/evaluation/evaluate.py` does (including
  its missing-checkpoint error naming the train command), resolves
  `training_run_id` per model via `shared.tracking.read_run_id` on the
  checkpoint's sidecar, and for each of the two models and three splits computes
  and stores records. A missing dataset must surface the existing
  `birds.dataset.splits` error naming the download command, and must do so
  before any checkpoint or model is loaded (the flowers-review fix the
  birds-revamp spec carried over). Print a per-(model, split) row count.
- **Success criteria:**
  - With the dataset absent, the run stops with the download command named and
    writes no rows
  - With a checkpoint absent, the run stops with the train command named
  - With no GPU available, the run refuses to proceed
  - An integration test with stub models and the `postgres_engine` fixture shows
    two records per photo across the splits it covers

### Task 5 — Summary tables

- **Objective:** Derive and report the agreement, confidence-separation and
  cascade tables from the stored test-split records.
- **Files:** `training/birds/evaluation/summary.py`,
  `training/tests/birds/evaluation/test_summary.py`
- **Details:** Read the test split's records via Task 3's reader and report:
  each model's accuracy, the gap, both-right, only-large-right,
  only-small-right, both-wrong and the oracle; the separation between each
  model's correct and incorrect predictions for both confidence measures
  (`sklearn.metrics.roc_auc_score`), for both models; and, for each confidence
  measure, accuracy and share of the gap recovered when the least-confident
  0, 5, 10, 15, 20, 30, 50 and 100 percent of photos take the large model's
  prediction. Every sort breaks ties on the photo identifier so output cannot
  vary between runs. Print the tables, and log the figures to MLflow through
  `shared.tracking.run(...)` as a single standalone run covering both models —
  tracking already degrades to a warning when off or unavailable, so no extra
  handling is needed. A `main()` makes it runnable as
  `python -m birds.evaluation.summary`.
- **Success criteria:**
  - On a hand-built record set, every reported figure matches a hand computation
  - The 0 percent cascade row equals the small model's accuracy and the 100
    percent row equals the large model's
  - Records with tied confidences produce identical output across repeated calls
  - With `TRAINING_TRACKING=off`, the summary still prints its tables

### Task 6 — Migration, then the real run

- **Objective:** Populate the real table and verify the figures against an
  independent source.
- **Files:** none changed
- **Details:** **Stop and ask the user to run migration `0004`** — per
  `CLAUDE.md`'s Infrastructure rule this task must not run it, nor start or stop
  Postgres. Once the user confirms, run
  `python -m birds.evaluation.predictions` then
  `python -m birds.evaluation.summary` from `training/`, and capture the summary
  output verbatim for Task 7. Cross-check the derived test-split accuracies
  against what `python -m birds.evaluation.evaluate` reports (0.7798 small,
  0.8699 large at last run) — these must agree. Run the summary a second time
  and confirm the output is identical.
- **Success criteria:**
  - 23,576 rows exist (two models × 11,788 photos)
  - Derived test-split accuracies match `evaluate.py`'s
  - Two consecutive summary runs produce identical output
  - The captured output is available for Task 7

### Task 7 — Rewrite the findings

- **Objective:** Replace the pilot figures in `findings/birds.md` with this
  feature's output.
- **Files:** `findings/birds.md`
- **Details:** Rewrite the comparison and cascade sections from Task 6's
  captured output. State that the numbers come from the committed checkpoints
  and from committed code that reproduces them, and name the two commands that
  do. Remove the standing caveat that the cascade table came from a cleared
  scratch script and cannot be regenerated, since this feature removes it. Keep
  the file's existing explanation of why birds replaced flowers, including the
  flowers figures, which remain unreproducible and should still say so.
- **Success criteria:**
  - Every number in the comparison and cascade sections appears in Task 6's
    output
  - The file names the commands that regenerate the tables
  - No claim remains that the cascade table cannot be reproduced

### Task 8 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && ruff check . && mypy .` is clean apart from the nine
    pre-existing `ndarray` errors in `coco/`
  - `cd database && ruff check . && mypy .` is clean
  - Both packages' full test suites pass, with the training suite above its
    pre-feature count of 222
