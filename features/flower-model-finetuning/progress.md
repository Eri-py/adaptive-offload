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
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Train and evaluate the real models
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 6 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
