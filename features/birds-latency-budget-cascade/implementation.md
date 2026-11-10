# Birds Cascade Under a Hard Latency Budget — Implementation Plan

## Summary

Evaluates the committed birds cascade under a hard per-request latency budget
by sweeping what the local and offload paths cost, using only the per-photo
predictions already in Postgres. Produces a boundary map over the
local-cost-by-offload-cost plane per budget, per-budget metric tables, and a
new section in `findings/birds.md`.

## Prerequisite — landed separately

This plan assumes `training/shared/routing.py` already exists, holding eight
shape-agnostic routing primitives extracted from attempt 1: the
`LOCAL`/`OFFLOAD`/`ESCALATE` constants and their action dtype,
`constant_actions`, `budget_only_actions`, `cascade_actions`,
`on_time_accuracy_per_row`, `RouterMetrics` with `metrics`, and
`BootstrapResult` with `bootstrap_mean_difference`.

That extraction is **its own refactor branch and PR**, not a task here. It
touches attempt 1's code and produces no birds result, so gating the research
work behind it was the wrong shape. It is verifiable on its own terms: the
`training/` suite passes at an unchanged test count, proving an import-only
move. Attempt 1's `outcomes` and `oracle_actions` are deliberately left where
they are — they read per-row latency columns off a COCO DataFrame, a shape
birds does not have.

## Approach & Key Decisions

- **Latency is scenario-level, so a "cell" is the unit of evaluation.** One
  cell is a (local cost, offload cost, budget) triple. Within a cell every
  photo shares both costs, so the static baselines are all-on-time or all-late
  and only the cascade varies per photo. The experiment is a loop over cells
  doing vectorised arithmetic over ~5,794 stored rows — no inference, no model
  loading, no GPU.
- **The cost model, the policies and the cell evaluation are one task, not
  three.** They are a single concern — the experiment's arithmetic — and
  splitting them would make three subagents each start cold and re-derive the
  same context. One file per concern still applies to the files.
- **The heatmap is independent of everything else** because it takes plain grid
  data, mirroring how `coco/router/budget_plot.py` is decoupled from the router
  and the database so it can be tested with synthetic input. Tasks run in
  sequence, but nothing depends on its position relative to Task 1.
- **The confidence threshold is tuned per cell on the validation records and
  applied to test**, so no test photo informs the operating point. An
  exhaustive sweep over every cut is already known to be cheap at this size.
- **The cascade's accuracy ceiling on birds is already measured, so Task 4 is
  a write-up, not an open question.** An exhaustive sweep over the stored test
  records puts the cascade's best at 0.8702 against the large model's 0.8699 —
  two photos out of 5,794, reached only by escalating 87% — and it needs ~70%
  escalation merely to match always-offload. This is a characterised negative
  result, and the spec's deliverables were written to hold either way.
- **Deliberately not done:** no learned router, no latency predictor, no
  migration or schema change, no new records, no phone or server work, no
  `[project.scripts]` entry (run as `python -m birds.router.budget_experiment`,
  as the other birds entry points are run), and no change to attempt 1's
  results or findings file.

## Out of Scope

Carried from the spec: measuring birds models on the phone or exporting to
Core ML; deploying the large model behind a real endpoint; reusing attempt 1's
YOLOv8n phone figures as birds anchors; any learned router, confidence score or
latency predictor; the soft utility and any lambda weighting; battery, energy,
bandwidth-cap, frame-rate and thermal constraints; stale-result decay;
retraining or recipe/split changes; new per-photo records, table changes,
migrations or any database create/drop; changes to attempt 1's results or
findings; more than the two committed models; measuring the exported
reduced-precision model's predictions; offload cost drawn with noise or any
model of estimation error.

Also out of scope for this plan specifically: the shared-primitives extraction,
which lands as its own PR (see Prerequisite).

## Dependencies and Configuration

None to add. `matplotlib` is already a `training/` dependency (used by
`coco/router/budget_plot.py`), `numpy`/`pandas` are already present, and no new
config entry, port or migration is needed. The experiment reads `DATABASE_URL`
from `training/.env` through the existing engine helper.

Two inputs beyond code:

- `training/shared/routing.py` must exist (see Prerequisite).
- **The large model's measured per-photo GPU latency**, for the server-compute
  term. See Open Questions.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/birds/router/__init__.py` | add | New subpackage for attempt 2's decision layer | Mirrors `coco/router/` |
| `training/birds/router/conditions.py` | add | Offload cost model, named network conditions, sweep ranges | One concern per file |
| `training/birds/router/policies.py` | add | Scenario-level outcomes, within-budget oracle, budget bands | Birds' latency shape |
| `training/birds/router/budget_experiment.py` | add | Cell evaluation, threshold tuning, and the entry point | Attempt 2's counterpart to spec 03 |
| `training/birds/router/boundary_plot.py` | add | Boundary heatmap from plain grid data | Decoupled, like `budget_plot.py` |
| `training/tests/birds/router/test_conditions.py` | add | Cost model and unit-conversion tests | |
| `training/tests/birds/router/test_policies.py` | add | Outcome, oracle and band tests | |
| `training/tests/birds/router/test_budget_experiment.py` | add | Cell evaluation, tuning, determinism, and the end-to-end run | |
| `training/tests/birds/router/test_boundary_plot.py` | add | Figure-writing test with synthetic data | |
| `findings/birds.md` | edit | New section with the results | Guidelines: results append here |

## Tasks

Four work tasks plus the regression run. Tasks 1 and 2 are independent of each
other; their order is arbitrary.

### Task 1 — Cost model, scenario policies and cell evaluation

- **Objective:** Turn a cell plus the stored per-photo predictions into every
  policy's metrics, with the cascade's threshold tuned on validation records.
- **Files:** `training/birds/router/__init__.py`,
  `training/birds/router/conditions.py`,
  `training/birds/router/policies.py`,
  `training/birds/router/budget_experiment.py`,
  `training/tests/birds/router/test_conditions.py`,
  `training/tests/birds/router/test_policies.py`,
  `training/tests/birds/router/test_budget_experiment.py`
- **Details:**
  - *Cost model* (`conditions.py`): offload cost from bandwidth, round-trip,
    payload and server compute, in milliseconds. Define the three named
    conditions (good, medium, poor) as bandwidth/round-trip pairs, the fixed
    payload size, and the server-compute constant — each commented with its
    provenance, and the server-compute constant naming the run it was read
    from. Define the local-cost sweep covering 1–150 ms, the offload sweep, and
    the marked real measurement (the small model's desktop CPU time).
  - *Policies* (`policies.py`): given a local cost, an offload cost and
    per-photo small/large correctness, produce per-photo latency and
    correctness for always-local, always-offload, budget-only and the cascade
    at a threshold, reusing the action pickers and
    `on_time_accuracy_per_row`/`metrics` from `shared.routing`. Add a
    within-budget oracle for this shape: per photo, the most accurate option
    whose cost fits, falling back to local. Add a band classifier returning
    which of the four bands a cell is in — nothing fits, local only, offload
    but no escalation, everything — plus the lockout band's width, which equals
    the local cost.
  - *Cell evaluation* (`budget_experiment.py`): read both models' test and
    validation records through the existing store reader and align them per
    photo, as the existing summary does. Per cell and per confidence measure,
    pick the threshold maximising on-time accuracy on validation, apply it to
    test, and report mean accuracy, on-time accuracy, mean latency and
    over-budget share for all five policies. Add bootstrap intervals for the
    tuned cascade's on-time accuracy against budget-only and against
    always-local, resampling test photos with the shared helper, and the
    usefulness verdict (interval against budget-only entirely above zero).
    Report per cell the fewest escalated photos at which the cascade matches
    always-offload's accuracy, or that it never matches it. Choose the reported
    budgets so at least one admits no offload under the poor condition and at
    least one admits escalation under the good one. Hold no DataFrame on the
    result record, so two runs compare with plain equality — the trap attempt
    1's result type documents.
- **Success criteria:**
  - Unit conversion is tested directly: 20 KB over 10 Mbps is 16 ms.
  - Each named condition yields a distinct, ordered offload cost (good <
    medium < poor) at the fixed payload.
  - An escalated photo is charged local plus offload cost and takes the large
    model's correctness; a kept photo is charged local cost only.
  - When local plus offload exceeds the budget, no photo is escalated; when
    local cost alone exceeds it, every policy's on-time accuracy is zero.
  - The oracle's on-time accuracy is greater than or equal to every other
    policy's, in each of the four bands.
  - Both static baselines report an over-budget share of exactly 0 or 1.
  - A test asserts the tuning input is the validation records only, so no test
    photo influences any threshold.
  - A hand-built pair of tiny record sets produces hand-computed on-time
    accuracies for all five policies.
  - Two runs over unchanged records produce equal results.

### Task 2 — Boundary heatmap

- **Objective:** Draw, per budget, which policy wins across the
  local-cost-by-offload-cost plane.
- **Files:** `training/birds/router/boundary_plot.py`,
  `training/tests/birds/router/test_boundary_plot.py`
- **Details:** Takes plain grid data — no router or database dependency, so it
  is testable with synthetic input — and writes one panel per budget, marking
  the small model's real desktop measurement and the three named network
  conditions. Use the non-interactive backend as `budget_plot.py` does, since
  this module only ever saves a file. No attempt 1 phone figure appears on it.
- **Success criteria:**
  - Writes a non-empty file for synthetic grid data, in a test.
  - Panels appear in budget order, one per budget.
  - A test asserts the marked annotations are the desktop measurement and the
    three named conditions.

### Task 3 — Wire the entry point

- **Objective:** One runnable command computes every cell, prints the tables,
  writes the figure and records the run in tracking.
- **Files:** `training/birds/router/budget_experiment.py`,
  `training/tests/birds/router/test_budget_experiment.py`
- **Details:** Add the entry point: load the environment file as the other
  birds entry points do, open the engine, run every cell, print the per-budget
  tables and the band summary, write the figure through Task 2's module, and
  record the run through the shared tracking module as its own run. A missing
  prerequisite (no stored records for a needed split) stops with a message
  naming the command that records them, matching how the existing birds entry
  points behave.
- **Success criteria:**
  - An integration test runs the entry point against a disposable Postgres
    database seeded with a small record set, with tracking off, and asserts the
    tables print and the figure is written.
  - Missing records for a split stop the run with a message naming the
    recording command, and nothing is written.

### Task 4 — Real run and write-up

- **Objective:** Run the experiment against the real records and write the
  results into `findings/birds.md`.
- **Files:** `findings/birds.md`
- **Details:** Run the entry point, then add a section carrying: the per-budget
  tables, the boundary figure by relative path, the usefulness verdicts, where
  the boundary sits and how it moves with the budget, the four bands and the
  arithmetic that escalation's band always sits above offload's, the lockout
  band's width equalling the local cost, the fewest offload calls at which the
  cascade matches always-offload (or that it never does), the related-work note
  citing DeepDecision, Big/LITTLE and DDNN, and both caveats the spec requires
  — that the predictions are a desktop full-precision stand-in for the local
  path, and that offload cost is exact so the cascade's hedging advantage is
  unmeasured, with noisy offload cost named as the next experiment. State that
  the local axis is unanchored on real hardware and name the measurement that
  would anchor it.
- **Success criteria:**
  - Every number in the new section came from this run's output.
  - The section names the commands that regenerate it.
  - The existing prediction-recording and summary entry points still produce
    their previous numbers.

### Task 5 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `training/` and `database/` suites both pass with none skipped.
  - `ruff check .` clean in both packages; `mypy` reports no new errors beyond
    the nine pre-existing `ndarray` ones in `coco/`.

## Open Questions

- **The large model's measured per-photo GPU latency**, for Task 1's
  server-compute constant. The user is supplying this from the tracking UI; the
  constant records which run it came from. Task 1's cost model cannot be
  finished without it, though its policy and cell-evaluation halves can.
