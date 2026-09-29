# Learnings — birds-revamp

## Task 1 — rename
- `mypy .` from training/ also checks the gitignored `training/scratch/` (23 extra errors, e.g. cub_pilot.py); the 9 known errors are the ones outside scratch (coco/router, tests/coco).
- Left for Task 2: `Flowers102` usage in birds/data.py, birds/download.py, tests/birds/test_data.py; "Oxford 102 Flowers" prose in data.py and models.py; NUM_CLASSES=102.
