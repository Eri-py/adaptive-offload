# Birds Revamp — Implementation Plan

## Summary

Turn the `flowers` package into a `birds` package trained on CUB-200-2011:
- rename the package and its tests
- replace the Oxford-Flowers loading with CUB loading, a stratified
  validation split and an explicit download step
- fold in the flowers review fixes
- record the pilot's flowers-vs-birds rationale
- train and evaluate both models on birds

## Approach & Key Decisions

- **Rename first, then change behaviour.** Task 1 is a pure `git mv`, with
  imports and paths updated and the class count still at 102, so each later
  diff shows only real changes. The package becomes `birds`, run as
  `python -m birds.<name>`.
- **CUB loading uses the dataset's own index files,** since torchvision has no
  built-in CUB loader. `images.txt`, `image_class_labels.txt` and
  `train_test_split.txt` give the official split. Validation is 10% of each
  species' official training photos, rounded down with a minimum of one,
  chosen with `numpy.random.default_rng(42)` exactly as in the pilot. That
  makes the pilot's validation photos the same as this spec's. The public
  loader API stays the same shape as before
  (`load_train`/`load_val`/`load_test(transform)`, `make_data_loader(...)`),
  so the training and evaluation code barely changes.
- **The download step fetches from the Caltech URL** into
  `training/data/cub200/`, then unpacks it. It skips the download when the
  data is already complete: all 11,788 images listed in `images.txt` exist.
  The dataset is already on disk from the pilot, so the real run should
  report "already present".
- **Review fixes:**
  - **S1:** `train.main` and `evaluate.main` build their data loaders before
    building any model.
  - **S2:** one `top1_accuracy` function in a new `birds/metrics.py`, used
    for both validation (training) and test (evaluation). It's in its own
    module to avoid a circular import, and it isn't named `test_*`, so pytest
    never collects it. It runs in fp32, matching the evaluation.
  - **S3:** the three missing tests.
  - **S4:** one-line comments.
  - **N1:** `monkeypatch` everywhere.
- **Data-loader workers:** a `NUM_WORKERS = 6` setting in `config.py`,
  passed to the data loaders. The pilot trained with 6 workers; birds has
  about five times as many training photos as flowers, and single-process
  loading would starve the GPU. This is a speed setting only, not tuning:
  results don't depend on it, because augmentation is seeded per run, not
  per worker.
- **The rationale write-up** is `training/birds/findings.md`. Its numbers are
  copied from `training/scratch/pilot_results.txt`, the output of the pilot
  run.

## Out of Scope

- Keeping flowers code, or supporting multiple datasets.
- Birds-specific tuning: resolution, epochs, learning rates, architectures.
- Per-photo storage, the simulator or routers on birds, and schema changes.
- Phone export, and app or server changes.

## Dependencies and Configuration

- No new packages. `numpy` and `Pillow` are already installed.
- `training/pyproject.toml`: `packages.find include` changes `flowers*` to
  `birds*`. Then reinstall with `pip install --no-deps -e training` so the
  editable install maps the renamed package, as learned in the training
  reorg.
- Data: `training/data/cub200/` (gitignored, already present).
  Checkpoints: `training/models/birds/` (gitignored, via `*.pt`).
- No database change.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/flowers/**` → `training/birds/**` | move | Attempt-2 package | Rename |
| `training/tests/flowers/**` → `training/tests/birds/**` | move | Tests mirror the source | Rename |
| `training/birds/config.py` | edit | CUB paths, 200 classes, workers setting | Birds |
| `training/birds/data.py` | edit | CUB index-file loading, stratified validation split, missing-data error | Birds |
| `training/birds/download.py` | edit | Fetch and unpack CUB, skip if already complete | Birds |
| `training/birds/metrics.py` | add | The shared `top1_accuracy` | S2 |
| `training/birds/train.py`, `training/birds/evaluate.py` | edit | Data before model, shared accuracy, one-line comments | S1, S2, S4 |
| `training/birds/findings.md` | add | Pilot rationale | Spec |
| `training/tests/birds/*.py` | edit/add | Birds data tests, S3 tests, `monkeypatch` | S3, N1 |
| `training/pyproject.toml` | edit | Package discovery | Rename |
| `.claude/coding-guidelines.md` | edit | Attempt 2 is bird species identification | Spec |

## Tasks

### Task 1 — Rename flowers to birds

- **Objective:** Move the package and tests to `birds`, with no behaviour change.
- **Files:** `training/flowers/**` → `training/birds/**`,
  `training/tests/flowers/**` → `training/tests/birds/**`,
  `training/pyproject.toml`, `.claude/coding-guidelines.md`
- **Details:**
  - `git mv` both directories, and update every `flowers` import, module
    path, command string (`python -m flowers.` → `python -m birds.`) and
    docstring.
  - In `config.py`, point `DATA_DIR` at `data/cub200` and `WEIGHTS_DIR` at
    `models/birds`. Leave everything else as it is in this task.
  - Update `pyproject.toml`'s package discovery to `coco*`, `birds*` and
    reinstall with `--no-deps -e training`.
  - In the coding guidelines, change the attempt-2 description to "bird
    species identification (CUB-200-2011)".
- **Success criteria:**
  - No `training/flowers` or `training/tests/flowers` remains, and no
    `flowers` import remains.
  - The full training suite passes (189). The flowers-dataset tests still
    pass by patching, since the dataset code changes in Task 2.
  - ruff is clean, and mypy shows only the 9 known errors.

### Task 2 — CUB-200 data, split and download

- **Objective:** Replace the Oxford-Flowers data code with CUB-200-2011
  loading, the stratified validation split and the explicit download.
- **Files:** `training/birds/config.py`, `training/birds/data.py`,
  `training/birds/download.py`, `training/tests/birds/test_data.py`
- **Details:**
  - `config.py`:
    - `NUM_CLASSES = 200`
    - `NUM_WORKERS = 6`
    - `CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"`
    - `VAL_FRACTION = 0.1`
  - `data.py`:
    - A CUB dataset class over (photo id, relative path, label 0–199)
      records. It opens images as RGB and applies the transform.
    - `split_ids()` reads the three index files and returns disjoint
      train/val/test id lists. For validation, per species in class order:
      shuffle that species' official-train ids with one
      `default_rng(SEED)` and take `max(1, n // 10)`. This is the pilot's
      procedure.
    - If the dataset root or any index file is missing, raise
      `FileNotFoundError` naming `python -m birds.download`.
    - `load_train`/`load_val`/`load_test(transform)`, `make_data_loader`
      (now passing `num_workers=NUM_WORKERS` and `pin_memory=True`), and the
      transforms stay as they were.
  - `download.py`:
    - If complete (all ids in `images.txt` exist as files), print "already
      present" with the count and exit.
    - Otherwise, stream the `.tgz` to `training/data/cub200/`, unpack it,
      delete the archive and verify completeness.
  - Tests use a tiny fake CUB tree built in `tmp_path` (a few species and
    photos) and cover:
    - the split is disjoint, the validation count per species is correct,
      and the split is deterministic across calls
    - test is exactly the official test ids
    - labels are 0-based
    - a missing root or a missing index file gives the helpful error
    - `make_data_loader` returns the named split's photos
  - Run `python -m birds.download` for real. It should report the 11,788
    images already present.
- **Success criteria:**
  - The new tests pass, ruff is clean, and mypy has no new errors.
  - The real run reports 11,788 images.
  - The split sizes printed by a one-off check are 5,400 / 594 / 5,794.

### Task 3 — Flowers review fixes and 200-way models

- **Objective:** Apply S1, S2, S4 and N1 in the code, and add the S3 tests.
- **Files:** `training/birds/metrics.py`, `training/birds/train.py`,
  `training/birds/evaluate.py`, `training/birds/models.py` (only if needed
  for `NUM_CLASSES`), `training/tests/birds/test_train.py`,
  `training/tests/birds/test_evaluate.py`,
  `training/tests/birds/test_models.py`, `training/tests/birds/test_metrics.py`
- **Details:**
  - Add `metrics.top1_accuracy(model, loader, device) -> float` (fp32,
    `inference_mode`). Training's validation and evaluation's test both use
    it, and the old duplicate functions are removed.
  - `train.main` builds the train and validation loaders (which checks the
    data) before `require_cuda()` and `build_model(...)`. `evaluate.main`
    builds its test loaders before loading any checkpoint.
  - Update the model tests for 200 classes.
  - New tests:
    - `evaluate` raises the missing-checkpoint error, naming the train command
    - evaluation uses each model's own `eval_transform` (patch and record)
    - `train.main` asks for the data before building the model (patch both,
      record the order, and simulate a missing dataset: no model is built)
    - `top1_accuracy` on a hand-computed case
  - Replace all hand-rolled patching with `monkeypatch`. Shorten every
    comment and docstring to the one-line rule.
- **Success criteria:** All tests pass, ruff is clean, and mypy has no new
  errors.

### Task 4 — Record the pilot rationale

- **Objective:** Write `training/birds/findings.md` explaining why birds
  replaced flowers.
- **Files:** `training/birds/findings.md`
- **Details:**
  - Copy the numbers exactly from `training/scratch/pilot_results.txt`.
    Include:
    - a short table per dataset: both accuracies, the gap, the
      only-large-right and only-small-right shares, the oracle, and the
      confidence AUC (max probability and top-2 margin)
    - the cascade table for both datasets
    - the pilot settings: a single run, the flowers recipe unchanged, 224 px
      images, and the same stratified validation split for birds
  - Two sentences of interpretation, without overstating: flowers' gap is
    small, and birds leaves a larger prize, with confidence still
    informative.
- **Success criteria:** Every number in the file matches `pilot_results.txt`.

### Task 5 — Train and evaluate on birds

- **Objective:** Fine-tune both models on CUB-200 and report the results.
- **Files:** none changed (checkpoints go to the gitignored `training/models/birds/`).
- **Details:**
  - From `training/`, run `python -u -m birds.train --model small`, then
    `--model large`. Run each in the background, logging to
    `training/scratch/`, and check until done.
  - Then run `python -u -m birds.evaluate`.
  - Record per model: best validation accuracy and epoch, test accuracy on
    5,794 photos, time per photo and device, and training time.
  - Don't change code to chase accuracy. Stop and report if a run fails or
    either model's test accuracy is below 60%.
- **Success criteria:**
  - Both checkpoints exist and aren't tracked by git.
  - The evaluation reports both models on all 5,794 test photos.

### Task 6 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest` passes, and `cd database && pytest` passes.
  - ruff is clean, and mypy shows only the 9 known errors.
