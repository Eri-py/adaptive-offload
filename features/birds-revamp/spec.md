# Birds Revamp

## Overview

Attempt 2 switches from flower species to **bird species (CUB-200-2011)**. A
pilot run showed flowers leave almost nothing for a router to win: the phone
and server models differ by only 2.5 points. Birds differ by about 9.5
points, and the phone model's confidence still flags many of its own
mistakes. This spec turns the flowers code into birds code (same models, same
training recipe), fixes the issues the flowers review found, records why
birds was chosen, and trains and evaluates both models on birds.

## Requirements

- **The attempt-2 package becomes birds.** `training/flowers/` and its tests
  (`training/tests/flowers/`) become `training/birds/` and
  `training/tests/birds/`. No flowers-specific code remains. The flowers
  work stays in git history.
- **Dataset:** CUB-200-2011, 200 bird species.
  - It uses the dataset's official train/test split (5,994 / 5,794 photos).
  - A validation set is carved out of the official training photos: 10% of
    each species, chosen with a fixed seed, so it's the same every run. It's
    used only to pick the checkpoint.
  - The official test photos are used only for the final evaluation.
- **Dataset download** is a separate, explicitly run step. It fetches and
  unpacks CUB-200-2011 into the gitignored `training/data/cub200/`. If the
  data is already there and complete, it says so instead of downloading
  again. Training and evaluation never download the dataset. If it's
  missing, they stop with a message naming the download command.
- **Models and training recipe** are the same as the flowers attempt:
  MobileNetV3-Large (phone-sized) and ConvNeXt-Base (server-sized),
  fine-tuned from ImageNet weights with a 200-way output. Training runs on
  the local GPU and stops with a clear message if there isn't one. The
  checkpoint with the best validation accuracy is saved under the gitignored
  `training/models/`.
- **Evaluation** reports, for each model, top-1 accuracy on all 5,794 test
  photos and the average time per photo. The small model is timed on the
  CPU and the large model on the GPU, as in the flowers attempt.
- **Fixes carried over from the flowers review:**
  - A missing dataset is detected before anything else is loaded or
    downloaded, including the pretrained model weights (flowers S1).
  - Validation accuracy during training and test accuracy at evaluation are
    computed the same way, by one shared function (S2).
  - The paths that don't need a GPU or the dataset are covered by tests:
    - the missing-checkpoint error at evaluation
    - that each model is evaluated with its own image preprocessing
    - that training checks for the data before building the model (S3)
  - Comments follow the one-line rule in `.claude/coding-guidelines.md` (S4).
  - Tests use pytest's standard patching helper instead of hand-rolled
    save-and-restore (N1).
- **The rationale is recorded:** a short write-up in `training/birds/`
  covers the pilot's flowers-versus-birds comparison, with the numbers as
  measured:
  - accuracy of each model
  - how often only the large model is right
  - the oracle
  - how well the small model's confidence predicts its own correctness
  - the cascade table

  It also states the pilot's settings (a single run, the flowers recipe
  unchanged, 224-pixel images).
- **Docs:** the coding guidelines' description of attempt 2 changes from
  flower species identification to bird species identification.

## Out of Scope

- Keeping the flowers code, or supporting both datasets.
- Tuning for birds: higher image resolution, different epochs or learning
  rates, other architectures, or hyperparameter searches.
- Viability analysis beyond recording the pilot's numbers, such as storing
  per-photo predictions or running the simulator or routers on birds (next
  spec).
- Any database or schema change.
- Phone export, or any app or server change.
- Restoring the deleted flowers dataset or checkpoints.

## Acceptance Criteria

- Given the revamp is complete, then `training/birds/` and
  `training/tests/birds/` exist, and no flowers package, flowers tests or
  flowers-specific code remain.
- Given CUB-200-2011 is not present, when training or evaluation starts,
  then it stops with a message naming the download command, before loading
  or downloading anything else.
- Given the download step is run, then CUB-200-2011 is in
  `training/data/cub200/` with 11,788 photos across 200 species. Running it
  again when the data is complete doesn't download it again. Nothing under
  `training/data/` is tracked by git.
- Given the dataset, then the training, validation and test sets have no
  photo in common. Validation holds about 10% of each species' training
  photos, the same photos every run. Test is exactly the 5,794 official test
  photos.
- Given no GPU, when training starts, then it stops with a clear message.
- Given training has finished for both models, then each checkpoint was
  chosen on validation accuracy only, both are saved under
  `training/models/`, and neither is tracked by git.
- Given both trained models, when evaluation runs, then it reports each
  model's top-1 accuracy on all 5,794 test photos and its average time per
  photo on the stated device.
- Given the tests, then they cover:
  - the missing-dataset error, which fires before any model is built
  - the missing-checkpoint error
  - each model using its own preprocessing at evaluation
  - validation and test accuracy sharing one function
  - best-validation checkpoint selection
  - the train/validation/test split properties above

  None of them needs a GPU, the real dataset or downloaded weights.
- Given the write-up in `training/birds/`, then it states the pilot's
  flowers-versus-birds numbers and settings.
- Given this feature is complete, then the coding guidelines describe
  attempt 2 as bird species identification.
- Given this feature is complete, then the full training test suite passes,
  ruff is clean, and mypy shows no errors beyond the 9 already known.

## Open Questions

None.
