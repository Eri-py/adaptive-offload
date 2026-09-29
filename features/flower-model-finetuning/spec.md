# Flower Model Fine-Tuning

## Overview

The project is pivoting from general object detection to flower species
identification (attempt 2, in `training/flowers/`). The router needs two
models for this task: a small one that could run on a phone and a large one
for the server. This spec fine-tunes both on the Oxford 102 Flowers dataset
and reports how accurate and how fast each one is.

## Requirements

- **Dataset download:** a separate step, run explicitly, fetches the Oxford
  102 Flowers dataset (102 species) with its official train, validation and
  test splits (1,020 / 1,020 / 6,149 photos) into the gitignored
  `training/data/` folder. Training never downloads anything itself. If the
  dataset is missing, it fails with a clear message saying how to get it
  (`.claude/coding-guidelines.md`: acquiring data is an explicitly run
  concern, separate from the pipeline).
- **Two models, fine-tuned from ImageNet-pretrained weights:**
  - phone-sized: MobileNetV3-Large
  - server-sized: ConvNeXt-Base
- **Training:** each model trains on the training split. The validation split
  is used to decide when to stop, or which checkpoint to keep. The test split
  is used only once, for the final evaluation.
- **Hardware:** training runs on the local GPU. If no GPU is available, it
  stops with a clear message instead of silently running on the CPU.
- **Saved weights:** each model's final weights are saved locally under
  `training/models/` and are not committed to git.
- **Evaluation:** for each model, the step reports:
  - top-1 accuracy on the full test split
  - average inference time per photo, with the small model timed on the CPU
    (standing in for the phone) and the large model on the GPU (standing in
    for the server). This matches how the COCO simulator timed its two models.
- **Reproducibility:** randomness is seeded, so re-running gives closely
  comparable results. Exact bit-for-bit reproducibility on the GPU is not
  required.
- **Location:** all code lives in `training/flowers/`, with tests mirroring
  it in `training/tests/flowers/`.

## Out of Scope

- Any analysis of whether routing is viable, such as how often the small
  model is right when the large one is wrong, or whether the small model's
  confidence predicts correctness. That's the next spec.
- Storing per-photo predictions, confidences or timings in Postgres, and any
  schema change (next spec).
- Running the simulator or the routers on flower data.
- Exporting either model to the phone (TFLite or Core ML), or any change to
  the app or the server.
- Other datasets (such as PlantNet-300K), other model architectures, or
  hyperparameter searches beyond what's needed to train each model sensibly.
- Moving any code into a shared folder, unless this spec actually reuses it.

## Acceptance Criteria

- Given the dataset has not been downloaded, when training is started, then
  it stops with a message naming the missing data and the command that
  fetches it, and downloads nothing.
- Given the download step is run, then the Oxford 102 Flowers dataset is in
  `training/data/` with its official splits, and nothing under
  `training/data/` is tracked by git.
- Given no GPU is available, when training is started, then it stops with a
  clear message.
- Given the dataset and a GPU, when training runs for each model, then the
  checkpoint kept is chosen using the validation split only, and the test
  split is used only in the final evaluation.
- Given training has finished, then both models' weights are saved under
  `training/models/`, and `git status` shows them untracked or ignored.
- Given both trained models, when the evaluation runs, then it reports each
  model's top-1 accuracy on all 6,149 test photos, and its average time per
  photo on the stated device (CPU for the small model, GPU for the large one).
- Given this feature is complete, then the new code has tests covering
  everything that doesn't need a GPU or the real dataset. The full training
  test suite passes, ruff is clean, and mypy reports no errors beyond the 9
  already known.

## Open Questions

None.
