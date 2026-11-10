# Birds Cascade Under a Hard Latency Budget — Implementation Plan

## Summary

Evaluates the committed birds cascade under a hard per-request latency budget
by sweeping what the local and offload paths cost, using only the per-photo
predictions already in Postgres. Produces a boundary map over the
local-cost-by-offload-cost plane per budget, per-budget metric tables, and a
new section in `findings/birds.md`.

## Approach & Key Decisions

- **Eight shape-agnostic routing primitives move to `training/shared/routing.py`
  and attempt 1 is repointed at them.** The coding guidelines require code
  reused by both attempts to move to a shared location on first reuse, and
  `training/shared/` already exists for exactly this (it holds `tracking.py`).
  The move is import-only: no behaviour changes, and attempt 1's budget
  experiment has tests that prove it. This widens scope past the spec's "no
  change to attempt 1" line, which the user approved explicitly in preference
  to either duplicating the primitives or having attempt 2 import from
  attempt 1's package.
- **Attempt 1's `outcomes` and `oracle_actions` are deliberately *not* moved.**
  They read per-row latency columns off a COCO DataFrame, a shape birds does
  not have. Birds gets its own in `birds/router/policies.py`; generalising
  attempt 1's would mean changing its signatures, which is real risk for no
  gain.
- **Latency is scenario-level, so a "cell" is the unit of evaluation.** One
  cell is a (local cost, offload cost, budget) triple. Within a cell every
  photo shares both costs, so the static baselines are all-on-time or all-late
  and only the cascade varies per photo. The experiment is therefore a loop
  over cells doing vectorised arithmetic over ~5,794 stored rows — no
  inference, no model loading, no GPU.
- **The confidence threshold is tuned per cell on the validation records and
  applied to test**, so no test photo informs the operating point. An
  exhaustive sweep (every cut, not fixed percentiles) is already known to be
  cheap at this size.
- **A new plot module takes plain grid data, mirroring `coco/router/budget_plot.py`.**
  That module is deliberately decoupled from the router and the database so it
  can be tested with synthetic input; the birds heatmap follows the same rule
  rather than reusing the accuracy-vs-latency chart, which collapses the two
  latency axes this feature exists to separate.
- **Deliberately not done:** no learned router, no latency predictor, no
  migration or schema change, no new records, no phone or server work, and no
  change to attempt 1's results or findings file.

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

Further deferred by this plan: no `[project.scripts]` entry (run as
`python -m birds.router.budget_experiment`, matching how the other birds entry
points are run); no re-export shims left behind in attempt 1's modules, so each
moved primitive has exactly one import path.

## Dependencies and Configuration

None. `matplotlib` is already a `training/` dependency (used by
`coco/router/budget_plot.py`), `numpy`/`pandas`/`scikit-learn` are already
present, and no new config entry, port or migration is needed. The experiment
reads `DATABASE_URL` from `training/.env` through the existing engine helper.

**One input must be supplied before Task 2:** the large model's measured
per-photo GPU latency, for the server-compute term. See Open Questions.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/shared/routing.py` | add | The eight shared routing primitives | Guidelines: shared on first reuse |
| `training/coco/router/latency_policies.py` | edit | Drop the moved primitives, keep COCO-shaped ones | One source of truth per primitive |
| `training/coco/router/evaluation.py` | edit | Drop the moved bootstrap helper | Same |
| `training/coco/router/budget_experiment.py` | edit | Repoint imports at `shared.routing` | Call site of moved names |
| `training/coco/router/feature_experiment.py` | edit | Repoint bootstrap import | Call site of moved names |
| `training/coco/router/budget_plot.py` | edit | Docstring reference to the moved metrics record | Stale reference after the move |
| `training/tests/shared/test_routing.py` | add | Tests for the moved primitives | Tests follow the code |
| `training/tests/coco/router/test_latency_policies.py` | edit | Keep only the COCO-shaped tests | Rest moved to shared |
| `training/tests/coco/router/test_evaluation.py` | edit | Drop the moved bootstrap tests | Same |
| `training/birds/router/__init__.py` | add | New subpackage for attempt 2's decision layer | Mirrors `coco/router/` |
| `training/birds/router/conditions.py` | add | Offload cost model, named network conditions, sweep ranges | One concern per file |
| `training/birds/router/policies.py` | add | Scenario-level outcomes, within-budget oracle, budget bands | Birds' latency shape |
| `training/birds/router/budget_experiment.py` | add | The experiment entry point | Attempt 2's counterpart to spec 03 |
| `training/birds/router/boundary_plot.py` | add | Boundary heatmap from plain grid data | Decoupled, like `budget_plot.py` |
| `training/tests/birds/router/test_conditions.py` | add | Cost model tests | |
| `training/tests/birds/router/test_policies.py` | add | Outcome, oracle and band tests | |
| `training/tests/birds/router/test_budget_experiment.py` | add | Cell evaluation, tuning and determinism tests | |
| `training/tests/birds/router/test_boundary_plot.py` | add | Figure-writing test with synthetic data | |
| `findings/birds.md` | edit | New section with the results | Guidelines: results append here |

## Tasks

### Task 1 — Extract the shared routing primitives

- **Objective:** Move the eight shape-agnostic primitives into
  `training/shared/routing.py` and repoint attempt 1, changing no behaviour.
- **Files:** `training/shared/routing.py`,
  `training/coco/router/latency_policies.py`,
  `training/coco/router/evaluation.py`,
  `training/coco/router/budget_experiment.py`,
  `training/coco/router/feature_experiment.py`,
  `training/coco/router/budget_plot.py`,
  `training/tests/shared/test_routing.py`,
  `training/tests/coco/router/test_latency_policies.py`,
  `training/tests/coco/router/test_evaluation.py`
- **Details:** Move, unchanged in behaviour: the `LOCAL`/`OFFLOAD`/`ESCALATE`
  constants and their action dtype, `constant_actions`, `budget_only_actions`,
  `cascade_actions`, `on_time_accuracy_per_row`, `RouterMetrics` with `metrics`,
  and `BootstrapResult` with `bootstrap_mean_difference`. Leave in
  `latency_policies.py`: `outcomes`, `oracle_actions`, `raw_score`, the learned
  score helpers and the offload-latency model, all of which are COCO-shaped.
  `bootstrap_mean_difference`'s first parameter becomes `group_ids` with the
  docstring generalised off "frame"; confirm every call site passes it
  positionally before renaming. Leave no re-export shims. Move the
  corresponding tests to `tests/shared/test_routing.py` rather than
  duplicating them.
- **Success criteria:**
  - The full `training/` suite passes with the same test count as before the
    move, and `database/` still passes.
  - `ruff check .` clean; `mypy .` reports no new errors beyond the nine
    pre-existing `ndarray` ones in `coco/`.
  - No module imports a moved name from its old location.

### Task 2 — Birds offload cost model

- **Objective:** Express offload cost as server compute plus transfer plus
  round trip, with the named network conditions and the sweep ranges.
- **Files:** `training/birds/router/__init__.py`,
  `training/birds/router/conditions.py`,
  `training/tests/birds/router/test_conditions.py`
- **Details:** A cost function taking bandwidth, round-trip, payload size and
  server compute and returning milliseconds. Define the three named conditions
  (good, medium, poor) as bandwidth/round-trip pairs, the fixed payload size,
  and the server-compute constant — each with a comment naming its provenance,
  and the server-compute constant naming the run it was read from. Define the
  local-cost sweep range covering 1–150 ms and the offload sweep range, plus
  the marked real measurement (the small model's desktop CPU time). Unit
  conversion is the thing most likely to be wrong, so test it explicitly:
  20 KB over 10 Mbps is 16 ms.
- **Success criteria:**
  - A hand-computed cell's cost matches to the millisecond.
  - Each named condition yields a distinct, ordered offload cost (good <
    medium < poor) at the fixed payload.
  - Tests cover the unit conversion directly.

### Task 3 — Scenario-level policies and budget bands

- **Objective:** Turn a cell plus the stored per-photo correctness into each
  policy's per-photo latency and correctness, and classify the cell's band.
- **Files:** `training/birds/router/policies.py`,
  `training/tests/birds/router/test_policies.py`
- **Details:** Given a local cost, an offload cost and per-photo small/large
  correctness, produce per-photo latency and correctness for always-local,
  always-offload, budget-only and the cascade at a threshold, reusing the
  action pickers and `on_time_accuracy_per_row`/`metrics` from
  `shared.routing`. Add a within-budget oracle for this shape: per photo, the
  most accurate option whose cost fits, falling back to local. Add a band
  classifier returning which of the four bands a cell is in — nothing fits,
  local only, offload but no escalation, everything — and the lockout band's
  width, which equals the local cost.
- **Success criteria:**
  - An escalated photo is charged local plus offload cost and takes the large
    model's correctness; a kept photo is charged local cost only.
  - When local plus offload exceeds the budget, no photo is escalated.
  - When local cost alone exceeds the budget, every policy's on-time accuracy
    is zero.
  - The oracle's on-time accuracy is greater than or equal to every other
    policy's, in each of the four bands.
  - Both static baselines report an over-budget share of exactly 0 or 1.

### Task 4 — Cell evaluation, threshold tuning and uncertainty

- **Objective:** Evaluate every policy in a cell, tuning the cascade's
  threshold on validation records and reporting on test, with bootstrap
  intervals.
- **Files:** `training/birds/router/budget_experiment.py`,
  `training/tests/birds/router/test_budget_experiment.py`
- **Details:** Read both models' records for the test and validation splits
  through the existing store reader and align them per photo, as the existing
  summary does. For each cell and each of the two confidence measures, pick the
  threshold maximising on-time accuracy on validation, apply it to test, and
  report mean accuracy, on-time accuracy, mean latency and over-budget share
  for all five policies. Add bootstrap intervals for the tuned cascade's
  on-time accuracy against budget-only and against always-local, resampling
  test photos with the shared helper, and the usefulness verdict (interval
  against budget-only entirely above zero). Also report, per cell, the fewest
  escalated photos at which the cascade matches always-offload's accuracy, or
  that it never matches it. Pick the reported budgets so at least one admits no
  offload under the poor condition and at least one admits escalation under the
  good one. Hold no DataFrame on the result record, so two runs compare with
  plain equality — the same trap attempt 1's result type documents.
- **Success criteria:**
  - No test photo influences any threshold; a test asserts the tuning input is
    the validation records only.
  - Two runs over unchanged records produce equal results.
  - A hand-built pair of tiny record sets produces hand-computed on-time
    accuracies for all five policies.
  - The usefulness verdict follows the stated rule.

### Task 5 — Boundary heatmap

- **Objective:** Draw, per budget, which policy wins across the
  local-cost-by-offload-cost plane.
- **Files:** `training/birds/router/boundary_plot.py`,
  `training/tests/birds/router/test_boundary_plot.py`
- **Details:** Takes plain grid data — no router or database dependency — and
  writes one panel per budget, marking the small model's real desktop
  measurement and the three named network conditions. Use the non-interactive
  backend as `budget_plot.py` does, since this module only ever saves a file.
  No attempt 1 phone figure appears anywhere on it.
- **Success criteria:**
  - Writes a non-empty file for synthetic grid data, in a test.
  - Panels appear in budget order, one per budget.
  - A test asserts the marked annotations are the desktop measurement and the
    three named conditions.

### Task 6 — Wire the entry point end to end

- **Objective:** One runnable command computes every cell, prints the tables,
  writes the figure and records the run in tracking.
- **Files:** `training/birds/router/budget_experiment.py`,
  `training/tests/birds/router/test_budget_experiment.py`
- **Details:** Add the entry point: load the environment file as the other
  birds entry points do, open the engine, run every cell, print the per-budget
  tables and band summary, write the figure, and record the run through the
  shared tracking module as its own run. A missing prerequisite (no stored
  records for a needed split) stops with a message naming the command that
  records them, matching how the existing birds entry points behave.
- **Success criteria:**
  - An integration test runs the entry point against a disposable Postgres
    database seeded with a small record set, with tracking off, and asserts the
    tables print and the figure is written.
  - Missing records for a split stop the run with a message naming the
    recording command, and nothing is written.

### Task 7 — Real run and write-up

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

### Task 8 — Regression test run

- **Objective:** Run every test that exercises code added or modified in this
  plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `training/` and `database/` suites both pass with none skipped.
  - `ruff check .` clean in both packages; `mypy` reports no new errors.

## Open Questions

- **The large model's measured per-photo GPU latency**, for Task 2's
  server-compute constant. The user is supplying this from the tracking UI;
  the constant records which run it came from. Task 2 cannot be completed
  without it, and the exhaustive sweep already run means no other task depends
  on it.
