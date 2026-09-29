# Flower Model Fine-Tuning — Implementation Report

**Feature:** Flower Model Fine-Tuning
**Directory:** `features/flower-model-finetuning/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Config, download and data loading | completed |
| Task 2 — Model builders and GPU check | completed |
| Task 3 — Training loop with validation-based checkpointing | completed |
| Task 4 — Evaluation | completed |
| Task 5 — Train and evaluate the real models | completed |
| Task 6 — Regression test run | completed |

## Files changed

The range is `feature/training-reorg..feature/flower-model-finetuning`.

### Added

- `training/flowers/config.py`: paths, seed, class count, model-to-weights mapping, and per-model training settings.
- `training/flowers/download.py`: the explicit dataset download (`python -m flowers.download`).
- `training/flowers/data.py`: split loaders that never download, transforms, and the data-loader helper.
- `training/flowers/models.py`: MobileNetV3-Large and ConvNeXt-Base with 102-way heads, plus the GPU check.
- `training/flowers/train.py`: the fine-tuning loop, keeping the best validation checkpoint (`python -m flowers.train --model small|large`).
- `training/flowers/evaluate.py`: test accuracy and per-photo timing (`python -m flowers.evaluate`).
- `training/tests/flowers/test_data.py`, `test_models.py`, `test_train.py`, `test_evaluate.py`
- `features/flower-model-finetuning/spec.md`, `implementation.md`

### Edited

- `training/pyproject.toml`: declares `torch` and `torchvision`, and adds a mypy override for `torchvision`, which has no type information.

## Tests

### Unit

- `training/tests/flowers/test_data.py` (5 tests): a missing dataset gives the helpful error, and each loader and the data-loader helper request the right split.
- `training/tests/flowers/test_models.py` (4 tests): both models output 102 classes, and the GPU check raises or returns correctly.
- `training/tests/flowers/test_train.py` (3 tests): a checkpoint is saved, the kept checkpoint is the best validation epoch rather than the last, and training never takes the test split.
- `training/tests/flowers/test_evaluate.py` (3 tests): top-1 accuracy matches a hand-computed case, and timing is positive and finite, including when the warm-up is longer than the dataset.

All tests pass: `training/` 189 passed (174 before + 15 new), `database/` 1 passed, none skipped.

## Commits

`feature/training-reorg..feature/flower-model-finetuning`: spec, plan, Tasks 1–4, and a results commit. Tasks 5 and 6 changed no tracked files; their results are in `progress.md`.

## Notable events

- **Task 1:** the dataset download (about 676 MB on disk) outlasted the subagent, so the orchestrator waited for it to finish and checked the counts: 1,020 / 1,020 / 6,149.
- **Task 3:** made two small, pre-authorised additions outside its listed files: per-model training settings in `config.py`, and an optional seeded generator parameter on the data-loader helper in `data.py`.
- **Task 5 results** (real training on an RTX 5070):

  | Model | Best validation (epoch) | Test accuracy (6,149 photos) | Time per photo | Training time |
  |---|---|---|---|---|
  | MobileNetV3-Large | 94.41% (25 of 30) | 92.21% | 5.56 ms, CPU (6 threads) | ~2 min 50 s |
  | ConvNeXt-Base | 96.47% (13 of 15) | 94.67% | 6.63 ms, GPU | ~2 min 41 s |

  The checkpoints (17.5 MB and 350.8 MB) are in the gitignored `training/models/flowers/`.
