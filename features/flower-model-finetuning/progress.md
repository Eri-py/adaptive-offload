# Progress — flower-model-finetuning

## Task 1 — Config, download and data loading
- Status: completed
- Started: 2026-09-28 22:00:37
- Completed: 2026-09-28 22:10:40
- Notes: config/download/data added; torch+torchvision declared, torchvision mypy override. Real download: train 1020 / val 1020 / test 6149 (676 MB, gitignored). 5 new tests, 179 pass.

## Task 2 — Model builders and GPU check
- Status: completed
- Started: 2026-09-28 22:10:40
- Completed: 2026-09-28 22:19:05
- Notes: models.py (weights_for, build_model with 102-way head, require_cuda). Pretrained sanity check OK on RTX 5070. 4 new tests, 183 pass.

## Task 3 — Training loop with validation-based checkpointing
- Status: completed
- Started: 2026-09-28 22:19:05
- Completed: 2026-09-28 22:22:51
- Notes: train.py fit (AdamW, cosine, label smoothing, bf16 autocast, best-val checkpoint, no test loader) + main; TRAIN_SETTINGS in config.py; optional generator in make_data_loader (pre-authorised small extensions). 3 new tests, 186 pass.

## Task 4 — Evaluation
- Status: completed
- Started: 2026-09-28 22:22:51
- Completed: 2026-09-28 22:26:44
- Notes: evaluate.py (test_accuracy fp32, mean_latency_ms batch-1 forward with warm-up and cuda sync, checkpoint loading, two-row report). 3 new tests, 189 pass.

## Task 5 — Train and evaluate the real models
- Status: completed
- Started: 2026-09-28 22:26:44
- Completed: 2026-09-28 22:36:39
- Notes: small (MobileNetV3-Large): best val 94.41% @ epoch 25/30, test 92.21%, 5.56 ms/photo CPU (6 threads), train ~2m50s. large (ConvNeXt-Base): best val 96.47% @ epoch 13/15, test 94.67%, 6.63 ms/photo CUDA, train ~2m41s. Checkpoints untracked.

## Task 6 — Regression test run
- Status: completed
- Started: 2026-09-28 22:36:39
- Completed: 2026-09-28 22:36:52
- Notes: training: ruff clean, mypy 9 known, 189 passed; database: 1 passed. No files changed.
