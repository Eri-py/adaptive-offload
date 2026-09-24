# Progress — frame-feature-extraction

## Task 1 — frame_features table and migration
- Status: completed
- Started: 2026-09-24 12:34:13
- Completed: 2026-09-24 12:35:42
- Notes: FrameFeatures model + migration 0003 (offline render verified, not applied); round-trip test extended. database: ruff/mypy clean, 1 passed.

## Task 2 — Shared YOLO loader
- Status: completed
- Started: 2026-09-24 12:35:43
- Completed: 2026-09-24 12:37:10
- Notes: Public load_model extracted; _build_inference_fn uses it, closure unchanged. training: ruff clean, mypy 9 known errors only, 86 passed.

## Task 3 — Feature functions
- Status: completed
- Started: 2026-09-24 12:37:10
- Completed: 2026-09-24 12:42:46
- Notes: 12 new tests (98 total pass). Real-image check: 000000000139/285/632.jpg via load_model + model(path, device=cpu) twice — confidence features identical (13/1/7 detections).

## Task 4 — Feature persistence
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Extraction step (integration)
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 6 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
