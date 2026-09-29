# Flower Model Fine-Tuning — Implementation Plan

## Summary

Add a small `training/flowers/` package that downloads Oxford 102 Flowers,
fine-tunes MobileNetV3-Large (phone-sized) and ConvNeXt-Base (server-sized)
from torchvision's ImageNet weights on the local GPU, keeps each model's best
validation checkpoint, and reports test accuracy and per-photo inference time.

## Approach & Key Decisions

- **Everything uses torchvision, and nothing new is installed.**
  - The dataset is `torchvision.datasets.Flowers102` with its official splits.
  - The models are `mobilenet_v3_large` (`IMAGENET1K_V2` weights) and
    `convnext_base` (`IMAGENET1K_V1` weights), each with its final classifier
    layer replaced by a 102-way layer.
  - `torch` and `torchvision` are already in the shared venv (pulled in by
    `ultralytics`). They become direct imports, so they're declared in
    `training/pyproject.toml`.
- **Data is acquired separately, per the coding guidelines.**
  - `python -m flowers.download` is the only thing that downloads the
    dataset, into the gitignored `training/data/flowers102/`.
  - Training and evaluation load with `download=False`. If the data is
    missing, they raise an error that names the download command.
  - Pretrained ImageNet weights are fetched by torchvision into its own
    cache. That's model setup, not dataset acquisition.
- **One training recipe for both models,** with per-model settings in
  `flowers/config.py`:
  - **Optimiser:** AdamW with a cosine learning-rate schedule and label
    smoothing 0.1.
  - **Precision:** mixed precision on the GPU.
  - **Augmentation:** random resized crop and horizontal flip for training.
    Evaluation uses each weights' standard resize and centre-crop transform.
  - **Checkpoint:** the one with the highest validation accuracy is saved to
    `training/models/flowers/<model>.pt`. `*.pt` is already gitignored.
  - **Test isolation:** the training function never receives the test split.
- **Timing definition:** the model forward pass on one preprocessed photo
  (batch of 1), after a discarded warm-up, averaged over all 6,149 test
  photos. The small model is timed on the CPU and the large one on the GPU,
  with synchronisation before reading the clock. Accuracy is top-1 on all
  6,149 test photos, measured in batches on the GPU for speed.
- **Tests never need a GPU, the real dataset or downloaded weights.** The
  model builders take a `pretrained` flag (off in tests). The training and
  evaluation functions accept any model, data loader and device, so tests
  run tiny models on synthetic tensors on the CPU. The GPU check is tested
  by patching `torch.cuda.is_available`.
- **Nothing is shared yet.** No `coco` code is reused, so nothing moves into
  `training/shared/`. Entry points are run as modules
  (`python -m flowers.<name>`), not new registered command-line tools. The
  guidelines cap those at two.

## Out of Scope

- Viability analysis: small-right-when-large-wrong rates, and confidence vs
  correctness.
- Storing per-photo results in Postgres, and any schema change.
- The simulator or routers on flower data.
- Phone export (TFLite or Core ML), and any app or server change.
- Other datasets, other architectures, and hyperparameter searches.
- A `shared/` folder.

## Dependencies and Configuration

- `training/pyproject.toml`: add `torch>=2.4` and `torchvision>=0.19` to
  `dependencies`. Both are already installed (2.14 and 0.29); they're just
  declared now. No reinstall is needed for the imports to work.
- Downloads:
  - the Oxford 102 Flowers dataset (about 350 MB) into `training/data/flowers102/`, which is already gitignored
  - ImageNet weights into torchvision's cache
- If mypy can't find type information for `torchvision`, add a
  `[[tool.mypy.overrides]]` entry with `ignore_missing_imports`, matching the
  existing `cv2`, `scipy` and `pandas` entries.
- No database or schema change.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/flowers/config.py` | add | Paths, seed, per-model training settings | One place for settings |
| `training/flowers/download.py` | add | Explicit dataset download (`python -m flowers.download`) | Data acquisition kept separate |
| `training/flowers/data.py` | add | Load splits (never downloading), transforms, data loaders | Data loading |
| `training/flowers/models.py` | add | Build the small and large models with a 102-way head; GPU check | Model construction |
| `training/flowers/train.py` | add | Train loop, validation-based checkpointing, `python -m flowers.train --model small\|large` | Fine-tuning |
| `training/flowers/evaluate.py` | add | Test accuracy and per-photo timing, `python -m flowers.evaluate` | Reporting |
| `training/tests/flowers/test_data.py` | add | Missing-data error, split and transform wiring | Coverage without the dataset |
| `training/tests/flowers/test_models.py` | add | 102-way heads, GPU check | Coverage without a GPU |
| `training/tests/flowers/test_train.py` | add | Best-validation checkpoint kept, test split never used | The spec's isolation rule |
| `training/tests/flowers/test_evaluate.py` | add | Accuracy on known predictions, timing is positive and finite | Evaluation correctness |
| `training/pyproject.toml` | edit | Declare torch and torchvision | Now direct imports |

## Tasks

### Task 1 — Config, download and data loading

- **Objective:** Fetch the dataset through an explicit step and load its
  splits without ever downloading.
- **Files:** `training/flowers/config.py`, `training/flowers/download.py`,
  `training/flowers/data.py`, `training/tests/flowers/test_data.py`,
  `training/pyproject.toml`
- **Details:**
  - `config.py` holds:
    - `DATA_DIR = <training>/data/flowers102`
    - `WEIGHTS_DIR = <training>/models/flowers`, both resolved from `__file__`
    - `SEED = 42`
    - `NUM_CLASSES = 102`
  - `download.py`'s `main()` downloads all three splits with
    `Flowers102(root, split=..., download=True)` and prints each split's count.
  - `data.py`:
    - A loader function per split with `download=False`. It catches
      torchvision's missing-data error and raises a `FileNotFoundError` that
      names `python -m flowers.download`.
    - Train and eval transforms, where eval comes from the model weights'
      own `transforms()`.
    - A data-loader helper that takes a split, a transform, a batch size and
      a shuffle flag.
  - Tests: pointing at an empty `tmp_path` raises the helpful error, and the
    loaders pass the right split names (patch `Flowers102`).
  - Run `python -m flowers.download` for real. That's network access only,
    no infrastructure. Confirm the counts are 1,020, 1,020 and 6,149, and
    that `git status` shows nothing new under `training/data/`.
- **Success criteria:**
  - The new tests pass, ruff is clean, and mypy has no new errors.
  - The dataset is downloaded with the right counts and isn't tracked by git.

### Task 2 — Model builders and GPU check

- **Objective:** Build both models with 102-way heads, and refuse to train
  without a GPU.
- **Files:** `training/flowers/models.py`, `training/tests/flowers/test_models.py`
- **Details:**
  - `build_model(name: Literal["small", "large"], pretrained: bool = True)`
    returns the torchvision model. Its final classifier layer is replaced by
    a `Linear(in_features, 102)`: for MobileNetV3, `classifier[3]`; for
    ConvNeXt, `classifier[2]`.
  - `weights_for(name)` returns the torchvision weights enum.
  - `require_cuda()` raises `RuntimeError` with a clear message when
    `torch.cuda.is_available()` is false, and returns the CUDA device
    otherwise.
  - Tests use `pretrained=False`:
    - each model's output on a random `(2, 3, 224, 224)` tensor has shape `(2, 102)`
    - `require_cuda` raises when patched to be unavailable
- **Success criteria:** The new tests pass, ruff is clean, and mypy has no
  new errors.

### Task 3 — Training loop with validation-based checkpointing

- **Objective:** Train one model and keep its best validation checkpoint,
  never touching the test split.
- **Files:** `training/flowers/train.py`, `training/tests/flowers/test_train.py`
- **Details:**
  - `fit(model, train_loader, val_loader, device, epochs, lr, weight_decay, checkpoint_path) -> FitResult`:
    - AdamW, a cosine schedule, and cross-entropy with label smoothing 0.1
    - mixed precision via `torch.autocast` when on CUDA
    - after each epoch, computes validation accuracy and saves the
      `state_dict` to `checkpoint_path` when it's the best so far
    - `FitResult` records the best validation accuracy, its epoch, and the
      per-epoch history
  - The signature takes no test loader.
  - `main()`: `--model small|large`, seeds everything from `config.SEED`,
    calls `require_cuda()`, builds the pretrained model and the train/val
    loaders, and runs `fit` with per-model settings from `config.py`:

    | Model | Epochs | Learning rate | Weight decay | Batch |
    |---|---|---|---|---|
    | small | 30 | 1e-3 | 0.05 | 64 |
    | large | 15 | 1e-4 | 0.05 | 32 |

    It prints the result.
  - Tests use a tiny CPU model and synthetic loaders: a checkpoint is saved;
    with scripted validation accuracies, the checkpoint kept is the best
    epoch's, not the last; `fit`'s signature has no test-split parameter.
- **Success criteria:** The new tests pass, ruff is clean, and mypy has no
  new errors.

### Task 4 — Evaluation

- **Objective:** Report test accuracy and per-photo timing for a trained model.
- **Files:** `training/flowers/evaluate.py`, `training/tests/flowers/test_evaluate.py`
- **Details:**
  - `test_accuracy(model, loader, device) -> float`: top-1, batched.
  - `mean_latency_ms(model, dataset, device, warmup=10) -> float`:
    - batch of 1, preprocessed tensor, forward pass only
    - discard the warm-up runs
    - `torch.cuda.synchronize()` around the timing when on CUDA
    - average over every item
  - `main()`:
    - loads both checkpoints from `WEIGHTS_DIR`, failing clearly if one is missing
    - evaluates accuracy on the full test split on the GPU
    - times the small model on the CPU (`eval()`, `torch.inference_mode()`)
      and the large one on the GPU
    - prints a two-row table: model, test accuracy, mean ms per photo, device
  - Tests: accuracy on a stub model with known predictions matches a
    hand-computed value; latency on a tiny CPU model is positive and finite.
- **Success criteria:** The new tests pass, ruff is clean, and mypy has no
  new errors.

### Task 5 — Train and evaluate the real models

- **Objective:** Fine-tune both models on the real data and report the results.
- **Files:** none changed (weights go to the gitignored
  `training/models/flowers/`).
- **Details:**
  - From `training/`, run `python -m flowers.train --model small`, then
    `--model large`. Long runs should go in the background with the output
    logged to the scratchpad and checked until done, because a single shell
    command times out after 10 minutes.
  - Then run `python -m flowers.evaluate` and capture its full output.
  - Record in the progress notes, per model: best validation accuracy and
    epoch, test accuracy, mean ms per photo and device, and training
    wall-clock time.
  - Don't change code to chase accuracy. If a run fails or accuracy is
    implausibly low (under 70% for either model), stop and report instead.
- **Success criteria:**
  - Both checkpoints exist and aren't tracked by git.
  - The evaluation output reports both models' test accuracy on all 6,149
    photos, and their timings on the stated devices.

### Task 6 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest` passes (174 existing plus the new tests).
  - ruff is clean, and mypy shows only the 9 known errors (or those 9 plus
    none new).
