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
- Status: completed
- Started: 2026-11-10 09:28:44
- Completed: 2026-11-10 09:40:49
- Notes: Unblocked by running `birds.evaluation.evaluate`, which measured the
  large model's per-photo GPU latency at 6.731 ms. Negative result: the cascade
  is not useful in any of the 36 cells. Surfaced one acceptance-criterion miss
  from Task 3 (the always-local interval was computed but never printed) and one
  illegible figure label; both fixed in a follow-up commit with a new
  `format_report` test, suite 289 -> 290.

## Task 5 — Regression test run
- Status: completed
- Started: 2026-11-10 09:40:49
- Completed: 2026-11-10 09:40:49
- Notes: Full suite 290 passed, none skipped; ruff clean; mypy clean across
  `birds/` and `tests/birds/`. Nothing changed, so no commit of its own.
