# Progress — Birds Cascade Under a Hard Latency Budget

## Task 1 — Cost model, scenario policies and cell evaluation
- Status: completed
- Started: 2026-11-09 21:27:58
- Completed: 2026-11-09 21:37:52
- Notes: 17 new tests, suite 262 -> 279. Three inputs the plan assumed were
  available turned out not to be; payload and the network conditions were
  resolved mid-task (payload by measurement), server compute remains open and
  is a required argument with no default.

## Task 2 — Boundary heatmap
- Status: completed
- Started: 2026-11-09 21:38:54
- Completed: 2026-11-09 21:50:35
- Notes: 5 new tests, suite 279 -> 284. Rendered and inspected a figure from
  synthetic records matching the real joint distribution; the band structure
  reads correctly. Rendering also showed per-cell threshold tuning is both too
  slow and unnecessary — see learnings; Task 3 must compute the threshold once
  per measure rather than per cell.

## Task 3 — Wire the entry point
- Status: completed
- Started: 2026-11-09 21:51:23
- Completed: 2026-11-09 21:56:39
- Notes: 5 new tests, suite 284 -> 289. Threshold tuning restructured to once
  per measure; the full run is ~8 s against synthetic records, against an
  estimated 110 minutes per budget before. `--server-compute-ms` is a required
  CLI argument, so Task 4 cannot run until the real figure is supplied.

## Task 4 — Real run and write-up
- Status: blocked
- Started: —
- Completed: —
- Notes: Blocked on the large model's measured per-photo GPU latency, the
  server-compute term. The entry point requires `--server-compute-ms` and has
  no default by design, so no real run is possible until the user supplies it
  from the tracking store (or authorises re-running
  `python -m birds.evaluation.evaluate`, a GPU pass over the test split).

## Task 5 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
