# Birds Revamp — Implementation Report

**Feature:** Birds Revamp
**Directory:** `features/birds-revamp/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Rename flowers to birds | completed |
| Task 2 — CUB-200 data, split and download | completed |
| Task 3 — Flowers review fixes and 200-way models | completed |
| Task 4 — Record the pilot rationale | completed |
| Task 5 — Train and evaluate on birds | completed |
| Task 6 — Regression test run | completed |

## Files changed

The range is `feature/flower-model-finetuning..feature/birds-revamp`.

### Moved (git renames, with content changes)

- `training/flowers/**` → `training/birds/**`, and `training/tests/flowers/**` → `training/tests/birds/**`. The data, download and evaluation modules were rewritten far enough that git records them as delete-plus-add rather than renames.
- `training/coco/router/findings.md` → `findings/coco-router.md` (see Notable events).

### Added

- `training/birds/metrics.py`: the shared top-1 accuracy used by both training and evaluation.
- `training/tests/birds/test_metrics.py`
- `findings/birds.md`: why birds replaced flowers, with the pilot's numbers.
- `features/birds-revamp/spec.md`, `implementation.md`

### Edited

- `training/birds/config.py`: CUB paths, 200 classes, the validation fraction, the download URL and the worker count.
- `training/birds/data.py`: reads the dataset's own index files, builds the stratified validation split, and fails with a message naming the download command.
- `training/birds/download.py`: fetches and unpacks CUB, and reports when the data is already present.
- `training/birds/train.py`, `evaluate.py`: build the data before any model or checkpoint, use the shared accuracy function, and carry one-line comments.
- `training/birds/models.py`: prose updated for 200 bird species.
- `training/pyproject.toml`: package discovery now finds `birds`, and mypy skips the gitignored scratch folder.
- `training/coco/router/two_stage.py`: a comment's reference to the findings file.
- `.claude/coding-guidelines.md`: attempt 2 is bird species identification, plus where findings files live.
- `.claude/agents/implementer.md`: the quality-gate paths.

## Tests

### Unit

- `training/tests/birds/test_data.py` (8 tests): on a fake dataset tree, the three splits are disjoint, the validation count per species is right, the split is identical across calls, test is exactly the official test photos, labels are 0-based, a missing root or index file gives the helpful error, and the loader returns the named split.
- `training/tests/birds/test_models.py` (4 tests): both models output 200 classes, and the GPU check raises or returns correctly.
- `training/tests/birds/test_metrics.py` (1 test): top-1 accuracy on a hand-computed case.
- `training/tests/birds/test_train.py` (5 tests): a checkpoint is saved, the best validation epoch is kept rather than the last, training never takes the test split, the data is requested before the model is built, and a missing dataset means no model is built at all.
- `training/tests/birds/test_evaluate.py` (4 tests): a missing checkpoint names the training command, each model is evaluated with its own preprocessing, and timing is positive and finite including when the warm-up exceeds the dataset.

All tests pass: `training/` 196 passed, `database/` 1 passed, none skipped. Ruff is clean and mypy reports only the 9 pre-existing errors.

## Commits

`feature/flower-model-finetuning..feature/birds-revamp`: spec, plan, Tasks 1–4, a results commit, and two commits for the findings move.

## Notable events

- **Task 1 landed as two commits.** The first (`1e7eab7`) captured only the file renames, because the orchestrator passed `git add` a path that no longer existed, which makes git abort the whole staging step. The second (`73ed47e`) completed it. The same mistake split the findings move into `aae8aa0` and `943775f`. The branch is correct as a whole, but neither first commit would run on its own.
- **The findings move was requested mid-branch and is outside this plan's scope.** Both `findings.md` files moved to a new top-level `findings/` folder (`coco-router.md`, `birds.md`), the figure link and a code comment were updated, and the convention was recorded in the coding guidelines. The figure itself stays where the experiment script writes it, so that script's behaviour is unchanged.
- **The validation split matches the pilot exactly.** Task 2 checked its split against the pilot's directly: the same 5,400 / 594 / 5,794 photos, so the pilot's numbers and this spec's are comparable.
- **Real results** (RTX 5070, CUB-200-2011, 5,794 test photos):

  | Model | Best validation (epoch) | Test accuracy | Time per photo | Training time |
  |---|---|---|---|---|
  | MobileNetV3-Large | 78.79% (27 of 30) | **77.98%** | 5.713 ms, CPU (6 threads) | ~2 min 54 s |
  | ConvNeXt-Base | 89.73% (13 of 15) | **86.99%** | 6.701 ms, GPU | ~5 min 27 s |

  The pilot reached 77.27% and 86.76% on test, so the spec run matches it and is marginally better on both. The accuracy gap is about 9 points, against 2.5 on flowers.
- **The flowers dataset and checkpoints were deleted** at the user's request before this feature started, so the flowers numbers survive only in `findings/birds.md`.
