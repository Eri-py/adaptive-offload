# Progress — Real Inference Paths

## Task 1 — Export module and its unit tests
- Status: completed
- Started: 2026-11-27 13:05:50
- Completed: 2026-11-27 13:16:00
- Notes: 10 tests added. Parity gate reloads the staged file via `litert_torch.load`
  and runs against that, not the in-memory converted model, so the artifact that
  gets published is the one that was verified. Orchestrator confirmed
  `litert_torch.load(path)` returns a callable model yielding an (1, 200) ndarray.
  Full training suite 300 passed / 0 skipped before the follow-up change.

## Task 2 — Produce and commit the real export artifacts
- Status: completed
- Started: 2026-11-27 13:16:00
- Completed: 2026-11-27 13:21:00
- Notes: Parity gate passed on 50 seeded real test photos: 0 disagreements (100%
  class agreement), worst max-softmax drift 2.59e-06 against the 1e-3 tolerance.
  small.tflite is 17,945,960 bytes; cub200-species.json is 200 unique entries,
  index 0 "Black footed Albatross", index 199 "Common Yellowthroat". Neither file
  is gitignored. No .partial file left behind.

## Task 3 — Server scaffolding, config, startup and health
- Status: not started
- Started: —
- Completed: —
- Notes: Deferred — out of scope for this invocation.

## Task 4 — Preprocessing and classification service
- Status: not started
- Started: —
- Completed: —
- Notes: Deferred — out of scope for this invocation.

## Task 5 — Classify endpoint
- Status: not started
- Started: —
- Completed: —
- Notes: Deferred — out of scope for this invocation.

## Task 6 — Request-logging table, migration and writer
- Status: not started
- Started: —
- Completed: —
- Notes: Deferred — out of scope for this invocation.

## Task 7 — End-to-end classify integration test
- Status: not started
- Started: —
- Completed: —
- Notes: Deferred — out of scope for this invocation.

## Task 8 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: Deferred — out of scope for this invocation.
