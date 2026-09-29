# Review — Birds Revamp

## Verdict

The implementation meets the spec and there are no blockers. The part that
matters most for research correctness — the split — is genuinely sound. I
verified it against the real dataset rather than trusting the tests: the three
splits are pairwise disjoint by id *and* by file path, their union is exactly
the 11,788 indexed photos, the test split is byte-for-byte the official 5,794
test ids, and the validation count is exactly `max(1, n // 10)` for all 200
species with no species missing. The split is deterministic across calls, and
it is identical — id for id — to the pilot's own `read_cub`, so `fit_ids`,
`val_ids` and `test_ids` all match and the pilot's numbers are comparable to
this run's. No test photo can reach training or checkpoint selection:
`train.main` only ever asks for `"train"` and `"val"` (`train.py:112-121`),
`fit` has no test parameter, and both facts are enforced by tests.

I re-ran `python -m birds.evaluate` on the saved checkpoints:

- **Accuracy:** reproduced exactly — 0.7798 small, 0.8699 large.
- **Latency:** within ~3% (5.555 vs 5.713 ms CPU, 6.823 vs 6.701 ms GPU).

I also ran the full training suite (196 passed), `ruff check .` (clean),
`mypy .` (exactly the 9 known `[type-arg]` errors), and `python -m
birds.download` (reports "already present: 11788 images", no re-download). I
did not run the `database/` suite, since it needs Postgres.

Every number in `findings/birds.md` matches `training/scratch/pilot_results.txt`
exactly — I checked all ten metric rows and all sixteen cascade cells, plus
the pilot validation bests (0.8047 @ 26, 0.8855 @ 9) against
`scratch/cub_pilot.log`. No flowers code or flowers import remains anywhere
outside the feature docs and the deliberate comparison prose.

On the two things the implementation report flagged: the incomplete commits
are a real but minor history hygiene problem (`1e7eab7` renames the files
without updating a single import, so that commit alone cannot import `birds`),
and the branch tip is correct — see N4. The findings move is out of this
plan's scope but was a direct user request, is self-consistent (the figure
link `findings/coco-router.md:569` resolves, the `two_stage.py` comment was
updated, and the convention is recorded in the guidelines), and I found no
dangling references.

Four gaps, all small: the download's completeness check doesn't look at the
index files it needs (S1), the flowers review's S4 landed everywhere except
the two `config.py` spots it cited by line number (S2), and one piece of test
coverage the spec asks for by name is missing (S3).

## Acceptance Criteria

1. **`training/birds/` and `training/tests/birds/` exist; no flowers package,
   tests or flowers-specific code remain.** MET.
   - `training/birds/` holds `config.py`, `data.py`, `download.py`,
     `metrics.py`, `models.py`, `train.py`, `evaluate.py`;
     `training/tests/birds/` mirrors it.
   - `git ls-files` shows no `training/flowers` or `training/tests/flowers`.
   - A repo-wide grep for `flowers` outside `features/` hits only
     `findings/birds.md`, which is the deliberate flowers-vs-birds comparison.
   - `training/pyproject.toml:43` discovers `coco*`, `birds*`.

2. **Missing dataset: training or evaluation stops naming the download
   command, before anything else is loaded or downloaded.** MET.
   - `data.py:72-81` raises `FileNotFoundError` naming `python -m
     birds.download`, and names the specific missing index files.
   - `train.py:110-121` builds both loaders before `require_cuda()` (`:123`)
     and `build_model(..., pretrained=True)` (`:124`), so no ImageNet weights
     download can precede the error. This is flowers S1, and it landed.
   - `test_train.py:129-146` asserts the call order is
     `["data:train", "data:val", "cuda", "model", "fit"]` and that a missing
     dataset builds no model and never reaches the GPU check.
   - `evaluate.py:76-81` builds both test loaders before any `_load_checkpoint`.
     (It does call `require_cuda()` first — see N2, a consistency nit, not a
     download.)

3. **Download puts 11,788 photos across 200 species in
   `training/data/cub200/`; re-running when complete doesn't re-download;
   nothing under `training/data/` is tracked.** MET, with a caveat.
   - I ran `python -m birds.download`: `CUB-200-2011 already present: 11788
     images at .../data/cub200/CUB_200_2011`. No network access.
   - `git check-ignore` confirms `.gitignore:15` covers `training/data/`, and
     `git ls-files training/data` is empty.
   - Caveat: "complete" is defined only over `images.txt` and the image files
     (`download.py:20-27`), not the other two index files the pipeline needs.
     See S1.

4. **Train/val/test are disjoint; validation is ~10% per species, the same
   photos every run; test is exactly the 5,794 official test photos.** MET.
   - Verified on the real dataset: sizes 5,400 / 594 / 5,794; all three
     pairwise intersections empty; no duplicate ids within a split; union
     equals all 11,788 ids; `set(test)` equals the official test ids exactly;
     `set(train) | set(val)` equals the official train ids exactly; file paths
     are also disjoint across splits (all 11,788 paths unique).
   - Per-species validation count equals `max(1, n_official_train // 10)` for
     all 200 species — zero mismatches — with every species represented
     (2 to 3 photos each).
   - Deterministic: `split_ids() == split_ids()` on the real data, and the rng
     is reconstructed per call at `data.py:94` with `SEED`, over species in a
     fixed `range(NUM_CLASSES)` order, from an id list sorted numerically
     (`data.py:91`). Nothing depends on dict iteration order or filesystem order.
   - Matches the pilot exactly: I re-implemented the pilot's `read_cub`
     (`scratch/cub_pilot.py:47-64`) and compared — `fit`, `val` and `test` id
     lists are all identical. The `int(n * VAL_FRACTION + 1e-9)` at
     `data.py:99` reproduces the pilot's `n // 10` without float drift.
   - Tests: `test_data.py:58-107` cover disjointness, per-species count,
     determinism, official test ids, 0-based labels, and both missing-data
     errors, on a fake tree with no GPU/network/dataset.

5. **No GPU: training stops with a clear message.** MET.
   - `models.py:37-44`; called at `train.py:123`.
   - `test_models.py:28-39` covers both directions.

6. **Each checkpoint chosen on validation only; both under
   `training/models/`; neither tracked.** MET.
   - `train.py:88-92` saves on strict improvement in `val_accuracy`; no test
     loader exists in the process.
   - `test_train.py:50-89` scripts accuracies `[0.5, 0.9, 0.7]` and compares
     the saved state dict tensor-by-tensor with the epoch-2 snapshot.
   - `test_train.py:92-104` pins `fit`'s signature against a test parameter.
   - Both files exist; `git check-ignore` reports `.gitignore:18` (`*.pt`)
     covers `training/models/birds/small.pt` and `large.pt`.
   - Logs confirm selection: small best 0.7879 @ 27/30, large 0.8973 @ 13/15.

7. **Evaluation reports top-1 accuracy on all 5,794 test photos and mean time
   per photo on the stated device.** MET.
   - `evaluate.py:70-106`. No `drop_last`, so all 5,794 photos count.
   - Reproduced by re-running (numbers in the Verdict). Timing excludes decode
     and transfer — only `model(image)` is between the clocks
     (`evaluate.py:44-48`), with CUDA sync on both sides.

8. **Test coverage of the listed paths, none needing a GPU, the real dataset
   or downloaded weights.** PARTIAL.
   - Missing-dataset error firing before any model: `test_train.py:138-146` ✓
   - Missing-checkpoint error: `test_evaluate.py:40-46` ✓ (flowers S3)
   - Per-model eval transform: `test_evaluate.py:49-72` ✓ (flowers S3)
   - Best-validation checkpoint selection: `test_train.py:50-89` ✓
   - Split properties: `test_data.py:58-107` ✓
   - **Validation and test sharing one function: not tested.** Nothing asserts
     `train.top1_accuracy is evaluate.top1_accuracy`. See S3.
   - All 196 tests run without a GPU, the dataset or weight downloads
     (`build_model(..., pretrained=False)` throughout). Confirmed by running them.

9. **The write-up states the pilot's flowers-vs-birds numbers and settings.**
   MET in substance, moved by request.
   - It is `findings/birds.md`, not `training/birds/findings.md` as the spec
     and plan say, because of the user's mid-branch move to a top-level
     `findings/` folder. Authorised deviation, and the guidelines now record
     the convention (`.claude/coding-guidelines.md:44-50`).
   - Content verified line by line against `scratch/pilot_results.txt`: both
     accuracies, the gap, both-right / only-large / only-small / both-wrong,
     the oracle, both confidence AUCs, and all eight cascade rows for both
     datasets. Every value matches.
   - Settings are stated (`findings/birds.md:8-23`): single run, flowers
     recipe unchanged with the hyperparameters spelled out, 224 px, the seeded
     stratified split with its 5,400 / 594 / 5,794 sizes. The label-smoothing
     caveat (`:60-61`) is a useful addition and is accurate.

10. **Guidelines describe attempt 2 as bird species identification.** MET.
    - `.claude/coding-guidelines.md:42`.

11. **Full training suite passes, ruff clean, mypy no errors beyond the known
    9.** MET.
    - Re-ran all three myself: 196 passed, ruff "All checks passed!", mypy
      "Found 9 errors in 4 files (checked 84 source files)" — the same 9
      `[type-arg]` errors in `coco/router` and `tests/coco`.
    - `training/pyproject.toml:57` adds `scratch/` to mypy's exclude. Not in
      the plan, but correct and necessary: mypy doesn't read `.gitignore`, and
      the pilot scripts are throwaway. Recorded in `learnings.md`.

## Scope

The diff matches the plan's Files Changed table for everything the plan
covers. Three things sit outside it:

- **`findings/` move** (`training/coco/router/findings.md` →
  `findings/coco-router.md`, `training/birds/findings.md` →
  `findings/birds.md`, plus `training/coco/router/two_stage.py:66` and the new
  `.claude/coding-guidelines.md:44-50` section). Out of this plan's scope, but
  an explicit mid-branch user request and flagged as such in the report. I
  checked it for damage and found none: the figure link
  (`findings/coco-router.md:569`) resolves to the existing
  `training/coco/router/latency_budget_curves.png`, no stale `findings.md`
  path references remain in code or docs, and the figure-writing script is
  untouched.
- **`.claude/agents/implementer.md:29`** — the quality-gate path list changed
  `training/flowers` to `training/birds`. Not in the plan's table, but a
  necessary consequence of the rename.
- **`training/pyproject.toml:57`** — the mypy `scratch/` exclude (see AC 11).

Nothing else is out of scope. No dataset tuning crept in: `TRAIN_SETTINGS`
(`config.py:47-50`) is byte-identical to the flowers recipe, and `NUM_WORKERS
= 6` is the speed-only setting the plan pre-authorised.

## Blockers

None.

## Suggestions

#### S1 — The download's completeness check ignores the index files the pipeline needs

- **File:** `training/birds/download.py:20-27`
- **Issue:** `_missing_images` defines "complete" as `images.txt` existing and
  every path it lists being present. `image_class_labels.txt` and
  `train_test_split.txt` are never checked, but `data.py:72-81` requires all
  three. A tree that has the images but lost an index file — an interrupted
  extraction, a partial copy — puts the user in a dead end: `download` prints
  "already present" and returns, then `train` refuses and tells them to run
  `download`. I reproduced this on a fake tree: download said "already
  present: 1 images" while `split_ids()` raised "missing:
  image_class_labels.txt, train_test_split.txt. Run `python -m
  birds.download`". The only escape is manually deleting the folder. The same
  gap is in the post-extraction verification at `:51-55`, so a truncated
  archive that still yields all images would be declared ready.
- **Fix:** Have `_missing_images` (or a small wrapper) also require the three
  files in `data._INDEX_FILES`, treating any absent one as incomplete so the
  download path runs. Importing `_INDEX_FILES` from `birds.data` keeps the two
  modules' definition of "complete" in one place, as `dataset_root()` already
  does.
- **Decision:** Accepted — addressed in "Address S1: check the index files in the download completeness check"

#### S2 — The two `config.py` spots flowers S4 cited by line number are untouched

- **File:** `training/birds/config.py:1-8`, `training/birds/config.py:44-46`
- **Issue:** The spec carries over flowers S4 ("comments follow the one-line
  rule"). It landed on the code the task rewrote — `evaluate.mean_latency_ms`,
  the checkpoint comment, `fit`'s docstring are all one line now — but
  `config.py`'s 8-line module docstring is verbatim what the flowers review
  cited as `flowers/config.py:1-8`, and the 3-line `TRAIN_SETTINGS` comment is
  verbatim `flowers/config.py:47-49`. The guideline
  (`.claude/coding-guidelines.md:165-169`) says one line, two only when truly
  needed, no multi-line blocks. Worse, that docstring still ends "`models.py`'s
  `weights_for`/`build_model` (Task 2)" and the comment says "(Task 3)" —
  references to the *flowers* plan's task numbering, shipped in source, which
  mean nothing to a later reader. `data.py:1-9` also adds a new 9-line module
  docstring on this branch.
- **Fix:** Cut `config.py:1-8` to one line (e.g. `"""Paths, seed and the
  small/large weights mapping shared by the birds pipeline."""`), drop the
  "(Task 2)"/"(Task 3)" references, and trim `:44-46` to its one load-bearing
  reason. Trim `data.py:1-9` the same way.
- **Decision:** Accepted — addressed in "Address S2: trim the config and data module docstrings"

#### S3 — No test asserts that validation and test share one accuracy function

- **File:** `training/tests/birds/test_metrics.py:1-23`
- **Issue:** The spec lists "validation and test accuracy sharing one
  function" as a path the tests must cover. The code does share
  `birds.metrics.top1_accuracy`, but nothing pins it: `test_train.py:71`
  patches `train_module.top1_accuracy` and `test_evaluate.py:66` patches
  `ev.top1_accuracy`, and both would still pass if someone reintroduced a
  second local implementation in either module. That is exactly the drift
  flowers S2 was raised to prevent, and it would silently make validation and
  test measured differently.
- **Fix:** One assertion, e.g. in `test_metrics.py`:
  `assert train.top1_accuracy is evaluate.top1_accuracy is metrics.top1_accuracy`.
- **Decision:** Accepted — addressed in "Address S3: assert validation and test share one accuracy function"

## Nitpicks

#### N1 — `top1_accuracy` has an undocumented "model already on `device`" precondition

- **File:** `training/birds/metrics.py:10-21`
- **Issue:** The merged function moves the *batches* to `device` but not the
  model, whereas the flowers `evaluate.test_accuracy` it replaced did call
  `model.to(device)`. Both current callers happen to satisfy it
  (`train.fit:54`, `_load_checkpoint:66`), so this is latent, not live — but
  the docstring doesn't say so, and a third caller would get a device-mismatch
  `RuntimeError` deep in the loop.
- **Fix:** Either add `model.to(device)` at the top, or say it in the
  docstring: `"""... `model` must already be on `device`."""`.
- **Decision:** — _(pending)_

#### N2 — `evaluate.main` checks the GPU before the dataset, unlike `train.main`

- **File:** `training/birds/evaluate.py:71-81`
- **Issue:** `require_cuda()` runs before the loaders, so on a machine with
  neither a GPU nor the dataset, evaluation reports the GPU error rather than
  the download command. `train.main` puts the loaders first for exactly this
  reason, and the comment at `evaluate.py:75` claims the same intent. Nothing
  is downloaded either way, so this is consistency, not a spec violation.
- **Fix:** Move the `test_loaders` construction above `require_cuda()`.
- **Decision:** — _(pending)_

#### N3 — The validation loop trusts `NUM_CLASSES` rather than the labels present

- **File:** `training/birds/data.py:96`
- **Issue:** `for species in range(NUM_CLASSES)` means any species whose label
  is `>= NUM_CLASSES` contributes no validation photos at all and lands
  entirely in train — silently, with no error. Correct for CUB-200 today
  (verified: all 200 species get 2–3 validation photos), and the tests only
  pass because the fixture monkeypatches `NUM_CLASSES` to 3
  (`test_data.py:47`), which is itself a sign the constant is doing work the
  data should do.
- **Fix:** Iterate `sorted(set(labels.values()))` instead, dropping the
  `NUM_CLASSES` import from the split path and the monkeypatch from the fixture.
- **Decision:** — _(pending)_

#### N4 — Two commits cannot stand on their own

- **File:** `features/birds-revamp/implementation-report.md:60-63`
- **Issue:** `1e7eab7` is a pure `git mv` with zero content changes, so at
  that commit `training/birds/*.py` still contains `from flowers.config import
  ...` while `pyproject.toml` still discovers `flowers*` — the package cannot
  be imported and the suite cannot run. `73ed47e` fixes it. The findings move
  is split the same way across `aae8aa0` and `943775f`. The branch tip is
  correct, so this only bites `git bisect` and single-commit reverts.
- **Fix:** Before opening the PR, `git rebase -i` to fixup `1e7eab7` into
  `73ed47e` and `aae8aa0` into `943775f`. The branch is unpushed, so this is
  free. (Moot if the PR is squash-merged.)
- **Decision:** — _(pending)_

#### N5 — `download` has no timeout on a multi-hundred-megabyte fetch

- **File:** `training/birds/download.py:32`
- **Issue:** `urllib.request.urlopen(CUB_URL)` with no `timeout`, so a stalled
  connection hangs the step indefinitely with no output after the one
  "downloading …" line.
- **Fix:** Pass `timeout=60`. A progress line per N chunks would also help,
  but is not required.
- **Decision:** — _(pending)_

## Tests

**Added or rewritten (22 tests in `tests/birds/`, all pass; 196 in the whole
training suite):**

- `test_data.py` (8): entirely new for CUB, on a fake three-species tree in
  `tmp_path`. Covers disjointness, per-species validation count (the layout
  `{25, 12, 5}` deliberately exercises both the `// 10` floor and the
  `max(1, …)` minimum), determinism, test-equals-official, 0-based labels,
  missing root, missing index file, and loader dispatch. This is the right set.
- `test_train.py` (5): the flowers checkpoint-selection test carried over
  (still the most valuable test here — it compares saved tensors against the
  epoch-2 snapshot), the signature guard, plus two new ordering tests. The
  order assertion `["data:train", "data:val", "cuda", "model", "fit"]` is
  nicely strict, and it incidentally proves `main` never asks for the test split.
- `test_evaluate.py` (4): missing checkpoint and per-model transform are the
  two flowers S3 gaps, now closed.
- `test_metrics.py` (1): hand-computed 0.5 on an `nn.Identity` model.
- `test_models.py` (4): updated to 200 classes; `pretrained=False` keeps them
  offline.

**Flowers N1 (`monkeypatch`) landed:** no hand-rolled `try/finally` patching
remains anywhere in `tests/birds/`.

**Gaps:**

- The shared-accuracy-function assertion the spec asks for (S3).
- `evaluate.main`'s loader-before-checkpoint order is not asserted, though
  `train.main`'s is. `test_main_uses_each_models_own_eval_transform` patches
  both and could record the order for free.
- `download.main` is untested. Reasonable — it needs the network — but the
  pure `_missing_images` function could be tested on a `tmp_path` tree, which
  would have caught S1.

**Fragility:** none found. The two `test_models.py` tests build full
ConvNeXt-Base on CPU and are slow-ish but deterministic; the latency tests
only assert positive-and-finite, which is right. The whole suite runs in 14 s.

## Recommended Decisions

- **S1** — Accept — A real dead end for the user, the fix is a few lines, and
  it closes the one soundness gap in a spec-required behaviour.
- **S2** — Accept — A direct guideline violation the spec explicitly asked to
  carry over, and the stale "(Task 2)/(Task 3)" references are shipped
  nonsense. Risk-free text edit.
- **S3** — Accept — One line of test, and it is named in the acceptance
  criteria; without it the flowers S2 fix has nothing holding it in place.
- **N1** — Accept — A one-line docstring clause or one `.to()` call; removes a
  silent trap the merge introduced.
- **N2** — Accept — Two lines moved, makes `evaluate` match `train` and its
  own comment.
- **N3** — Accept — Removes a constant that can silently skew the split, and
  simplifies the test fixture at the same time. Verify the split is unchanged
  afterwards (it will be).
- **N4** — Accept — Free to fix on an unpushed branch, and it keeps `bisect`
  usable over a rename this large. Skip if the PR is squash-merged.
- **N5** — Decline — Prototype bar, single-user step, and the dataset is
  already on disk; a hung download is obvious and Ctrl-C'able.
