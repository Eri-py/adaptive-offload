# Learnings — birds-revamp

## Task 1 — rename
- `mypy .` from training/ also checks the gitignored `training/scratch/` (23 extra errors, e.g. cub_pilot.py); the 9 known errors are the ones outside scratch (coco/router, tests/coco).
- Left for Task 2: `Flowers102` usage in birds/data.py, birds/download.py, tests/birds/test_data.py; "Oxford 102 Flowers" prose in data.py and models.py; NUM_CLASSES=102.

## Task 2 — CUB data
- Val split uses `int(n * VAL_FRACTION + 1e-9)` to equal the pilot's `n // 10` without float error; verified identical val/fit/test id lists vs the pilot's `read_cub` on real data (5400/594/5794).
- Tests monkeypatch `birds.data.DATA_DIR`, `NUM_CLASSES` (3-species fake tree) and `NUM_WORKERS=0` (avoid worker processes in the loader test).
- `download.py` imports `dataset_root()` from `birds.data` (pulls in torch; acceptable) so both agree on the root; completeness check reads only images.txt + image files, so the stray attributes.txt/tgz are irrelevant.
