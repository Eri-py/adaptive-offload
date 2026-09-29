# Learnings — birds-revamp

## Task 1 — rename
- `mypy .` from training/ also checks the gitignored `training/scratch/` (23 extra errors, e.g. cub_pilot.py); the 9 known errors are the ones outside scratch (coco/router, tests/coco).
- Left for Task 2: `Flowers102` usage in birds/data.py, birds/download.py, tests/birds/test_data.py; "Oxford 102 Flowers" prose in data.py and models.py; NUM_CLASSES=102.

## Task 2 — CUB data
- Val split uses `int(n * VAL_FRACTION + 1e-9)` to equal the pilot's `n // 10` without float error; verified identical val/fit/test id lists vs the pilot's `read_cub` on real data (5400/594/5794).
- Tests monkeypatch `birds.data.DATA_DIR`, `NUM_CLASSES` (3-species fake tree) and `NUM_WORKERS=0` (avoid worker processes in the loader test).
- `download.py` imports `dataset_root()` from `birds.data` (pulls in torch; acceptable) so both agree on the root; completeness check reads only images.txt + image files, so the stray attributes.txt/tgz are irrelevant.

## Task 3 — review fixes
- `birds.metrics.top1_accuracy` replaces train's `_evaluate_accuracy` and evaluate's `test_accuracy`; tests patch `train_module.top1_accuracy` / `ev.top1_accuracy` (name bound at import).
- `evaluate.main` now builds both test loaders up front; its rows loop uses distinct `row_*` names because mypy infers `name` as `ModelName` from the earlier loop.

## Task 5 results
- Spec run (python -u, logs in training/scratch/birds_*.log): small best val 0.7879 @ epoch 27 (train 23:32:18-23:35:12, ~2m54s); large best val 0.8973 @ epoch 13 (23:35:25-23:40:52, ~5m27s).
- Test on all 5,794 photos: small 0.7798 (5.713 ms/photo, cpu 6 threads), large 0.8699 (6.701 ms/photo, cuda). Both above the 60% floor.
- Vs pilot (val 0.8047@26 / 0.8855@9; test 0.7727 / 0.8676): small val -1.7 pts but test +0.7; large val +1.2 but test +0.2. Differences are checkpoint-selection noise on the 594-photo val split.
- Nothing under training/models/birds/ is tracked (gitignored); `git status` shows training/models/ untracked only because of pre-existing yolov8 files there.
