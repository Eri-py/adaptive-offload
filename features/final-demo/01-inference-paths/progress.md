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
- Status: completed
- Started: 2026-11-27 13:43:01
- Completed: 2026-11-27 13:49:00
- Notes: ruff clean, mypy strict clean (17 files), 3 tests pass. Orchestrator
  additionally verified against the REAL artifacts (not just the stub): the
  independently rebuilt ConvNeXt-Base accepts large.pt with strict=True, head is
  Linear(1024 -> 200), and load_species reads the real 200-entry file. That closes
  the main risk of the server not importing the birds package.

## Task 4 — Preprocessing and classification service
- Status: completed
- Started: 2026-11-27 13:45:41
- Completed: 2026-11-27 13:52:00
- Notes: ruff/mypy clean, 16 tests pass (11 new). Orchestrator verified both
  preprocessing branches against real photos: a 224 crop matches a normalize-only
  reference with 0.0 max abs diff, and a full-size CUB photo matches the training
  eval transform with 0.0 max abs diff. Double-transforming the 224 crop would
  have differed by 3.65 in normalized units, which is the cost the spec's
  conditional rule avoids.

## Task 5 — Classify endpoint
- Status: completed
- Started: 2026-11-27 13:48:00
- Completed: 2026-11-27 13:57:00
- Notes: ruff/mypy clean, 23 tests pass. Raw JPEG body, no multipart. Orchestrator
  verified end to end against the real model: output matches an independent
  reference exactly. A startup warmup was added after the real run showed a
  2224 ms first request; now 16.86 ms. Open question recorded in learnings: live
  server compute is ~12.5 ms vs the 6.731 ms the cost model assumes.

## Task 6 — Request-logging table, migration and writer
- Status: completed
- Started: 2026-11-27 13:51:39
- Completed: 2026-11-27 14:02:00
- Notes: database gate clean (1 test, now covering seven tables); server 27 passed.
  Migration 0005 is AUTHORED BUT NOT APPLIED — the user runs it. Orchestrator
  verified the migration against the ORM model by capturing its create_table call
  with a stand-in for alembic.op: all ten columns match on name, type and
  nullability, and the primary key matches. That check matters because the
  ephemeral test database is built with Base.metadata.create_all rather than by
  running migrations, so the suite cannot detect a model/migration mismatch.

## Task 7 — End-to-end classify integration test
- Status: completed
- Started: 2026-11-27 13:54:14
- Completed: 2026-11-27 14:08:00
- Notes: Plan correction — Task 7's Files list named only the test file, but the
  writer was left unwired by Task 6, so this task also wired log_classify_request
  into the classify route and measured total_ms there. server 29 passed, database
  1 passed. The implementer caught that conftest loads database/.env, so once the
  route called the writer the existing controller tests would have aimed inserts
  at the real database; the shared app fixture now overrides the logger with a
  no-op. Orchestrator confirmed the real database is untouched: still at alembic
  0004 with no classify_requests table.

## Task 8 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: Deferred — out of scope for this invocation.
