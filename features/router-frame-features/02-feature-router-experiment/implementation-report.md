# Feature Router Experiment — Implementation Report

**Feature:** Feature Router Experiment
**Directory:** `features/router-frame-features/02-feature-router-experiment/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Frame-level data loading | completed |
| Task 2 — Feature diagnostics | completed |
| Task 3 — Evaluation and bootstrap | completed |
| Task 4 — Two-stage routers | completed |
| Task 5 — Experiment script (integration) | completed |
| Task 6 — Run the experiment and write findings | completed |
| Task 7 — Regression test run | completed |

## Files changed

The files below are the ones this sub-plan changed, from `9fe98b5` (the end of
sub-plan 01's review work) to `HEAD` on `feature/router-frame-features`. For the
whole branch against `feature/real-model-inference`, see sub-plan 01's report
plus this one.

### Added

- `training/router/frame_dataset.py`
- `training/router/diagnostics.py`
- `training/router/evaluation.py`
- `training/router/two_stage.py`
- `training/router/feature_experiment.py`
- `training/tests/router/test_frame_dataset.py`
- `training/tests/router/test_diagnostics.py`
- `training/tests/router/test_evaluation.py`
- `training/tests/router/test_two_stage.py`
- `training/tests/router/test_feature_experiment.py`
- `features/router-frame-features/02-feature-router-experiment/implementation.md`

### Edited

- `training/router/findings.md`: new "Feature-router experiment" section appended; existing content unchanged.

## Tests

### Unit

- `training/tests/router/test_diagnostics.py` (5 tests): Spearman results for monotonic, inversely monotonic and independent features, ordering by |rho|, and the output columns.
- `training/tests/router/test_evaluation.py` (9 tests): hand-computed local and offload utility, decide-first and cascade scoring (an escalated row is charged local + offload latency), headroom share including the zero-denominator case, bootstrap determinism, and the usefulness verdict in both directions.
- `training/tests/router/test_two_stage.py` (6 tests): stage 1 never trains on a simulated frame, stage 2 never trains on a test photo, hand-computed cascade labels, deterministic picks, a single-class stage-2 label handled as a constant pick, and all four design/family combinations.

### Integration

- `training/tests/router/test_frame_dataset.py` (6 tests, ephemeral Postgres): the per-frame join and gap, exclusion of frames without features, dataset scoping and ordering, the simulated-frame set, and the merge of simulated rows with features.
- `training/tests/router/test_feature_experiment.py` (1 test, ephemeral Postgres): an end-to-end run on a synthetic dataset, twice. Results are identical, all four routers are present, and every number is finite.

All tests pass: `training/` 136 passed (109 before + 27 new), `database/` 1 passed, none skipped.

## Commits

`9fe98b5..HEAD` on `feature/router-frame-features`: the plan commit plus one commit per task for Tasks 1–6. Task 7 changed no files.

## Notable events

- **Task 5 found and fixed a silent row-alignment bug.** The frame-level split returns the test rows in a different order from the one the router uses internally. The experiment now derives the rows it scores the same way the router does. Nothing would have raised an error, because the scoring works by position.
- **Task 6 ran the experiment on the real database, read-only, twice,** and the output was identical. Result: all four routers score below always-offload, with 95% confidence intervals entirely below zero:

  | Router | Average utility | 95% CI |
  |---|---|---|
  | decide-first, linear | 0.6575 | [−0.0471, −0.0024] |
  | decide-first, gradient-boosted | 0.6459 | [−0.0661, −0.0085] |
  | cascade, linear | 0.6289 | [−0.0898, −0.0169] |
  | cascade, gradient-boosted | 0.6282 | [−0.0868, −0.0214] |

  Always-offload scores 0.6812 and the oracle 0.7208.
- **`router.baseline` reproduces its previous numbers:** always-offload 0.6812, utility regression 0.6811. It also prints the logistic router at 0.6574. The existing findings text describes that as "~0.68"; that looseness predates this branch.
