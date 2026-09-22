# Progress — real-model-inference

## Task 1 — Add the `model_inference` cache table
- Status: completed
- Started: 2026-09-22 12:38:06
- Completed: 2026-09-22 12:40:19
- Notes: 1/1 test passes, ruff/mypy clean, verified `label` enum type created exactly once across migration chain.

## Task 2 — Ground-truth parsing module
- Status: completed
- Started: 2026-09-22 12:40:33
- Completed: 2026-09-22 12:42:50
- Notes: 9/9 tests pass, ruff/mypy clean. Smoke-tested against real annotations file: 4952/5000 images have ground truth (48 have none, a known COCO val2017 quirk).

## Task 3 — Real-inference core types and IoU-based accuracy scoring
- Status: completed
- Started: 2026-09-22 12:43:02
- Completed: 2026-09-22 12:44:34
- Notes: 7 new tests, full suite 88/88 pass, ruff/mypy clean.

## Task 4 — Condition-driven latency overhead (replaces `stub_inference`)
- Status: completed
- Started: 2026-09-22 12:44:45
- Completed: 2026-09-22 12:47:58
- Notes: 28/28 scoped tests pass, ruff clean. mypy has exactly 1 expected error (run_simulation.py's still-live stub_inference import — Task 7's job). Full suite pytest currently INTERRUPTS entirely (not just skips one file) on that same import error — deferring full-suite runs to Task 7/8, using scoped paths for Tasks 5-6 verification instead.

## Task 5 — Real YOLO-backed inference builders
- Status: completed
- Started: 2026-09-22 12:48:22
- Completed: 2026-09-22 14:18:13
- Notes: Real YOLOv8n(CPU)/YOLOv8x(GPU) inference confirmed working — hand-smoke-tested by both the subagent and independently by the orchestrator (matching results on the same image). ultralytics 8.4.159 added; no mypy override needed (fully typed). Orchestrator added `*.pt` to .gitignore after the subagent flagged that downloaded weight files (yolov8n.pt 6.5MB, yolov8x.pt 137MB) landed untracked in training/. Scoped tests 28/28 pass, ruff clean, mypy has exactly the 1 expected pre-existing error (run_simulation.py's broken stub_inference import, Task 7's job).

## Task 6 — Persistence functions for the model-inference cache
- Status: completed
- Started: 2026-09-22 14:18:30
- Completed: 2026-09-22 14:20:11
- Notes: 11/11 scoped tests pass (real Postgres), ruff clean, mypy has exactly the 1 expected pre-existing error.

## Task 7 — Wire real inference into `run_simulation`'s orchestration
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 8 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
