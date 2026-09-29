# Review — Training Reorganisation

## Verdict

The implementation meets the spec, and nothing blocks a PR. The move is clean. Every rename is a git rename (similarity 86–100%), and the content edits are limited to import roots, `__file__` depth, logger-name strings, docstring/module-path references, `pyproject.toml` and the two layout docs. I checked "no behaviour change" myself instead of relying on the implementer's captures. On this branch I re-ran both experiments read-only against the real database. `coco.router.budget_experiment` stdout, `coco.router.feature_experiment` stdout and the `--figure` PNG are all byte-identical to `reorg_before/` (`cmp` clean, stderr empty). The committed `training/coco/router/latency_budget_curves.png` still hashes to `908cbe1c…fe55`. The full suite passes (174, none skipped), ruff is clean, and mypy reports the same 9 `[type-arg]` errors (72 files checked). Both `run-simulation --help` and `score-complexity --help` are byte-identical to the baseline. I found two small stale old-path references: a code comment in `database/migrations/env.py` and one prose path in `findings.md`. Neither affects behaviour.

## Acceptance Criteria

1. **All training source under `training/coco/`, nothing at old locations — MET.** `training/datagen/`, `training/router/`, `training/tests/datagen/` and `training/tests/router/` no longer exist on disk. The gitignored `router/models/*.joblib` moved to `training/coco/router/models/` and is still ignored (`.gitignore:21` `*.joblib`). `importlib.util.find_spec("datagen")` and `find_spec("router")` both return `None`, so no stale import root silently resolves. The editable finder maps only `coco` and `flowers`.
2. **Tests mirror the new layout, 174 pass, none skipped — MET.** Tests are in `training/tests/coco/{datagen,router}/**`, plus `training/tests/coco/__init__.py` and `training/tests/flowers/__init__.py`. I re-ran it: `174 passed`, `-rs` reports no skips.
3. **`training/flowers/` exists, importable, no attempt-specific code — MET.** `training/flowers/__init__.py:1` is a one-line docstring, and `import flowers` works. `pyproject.toml` `include = ["coco*", "flowers*"]` (`training/pyproject.toml:41`).
4. **Dataset and weights paths resolve to the same files — MET.** `training/coco/datagen/config.py:38-39` changed `parents[1]` → `parents[2]`, and `LOCAL_MODEL_WEIGHTS_PATH` resolves to `training/models/yolov8n.pt` (both weights exist). I checked all 7 `load_dotenv` calls by hand and each resolves to `training/.env`:
   - `cli/*.py` go up 4 levels: cli → datagen → coco → training
   - `router/*.py` go up 3 levels
   - `tests/conftest.py:20` is correctly unchanged

   The output paths that sit next to their files are correctly left alone: the three `baseline.py:26-28` model paths and the `budget_experiment.py:476` default figure path.
5. **Experiments byte-identical before/after — MET.** I verified it independently, as described in the Verdict.
6. **CLI tools work under existing names — MET.** `training/pyproject.toml:26-27`, and both `--help` outputs are byte-identical to the baseline.
7. **Ruff clean, mypy no errors beyond the 9 known — MET.** Re-run confirmed.
8. **Docs describe new paths, no `features/` edits apart from this spec's docs — MET, with one gap (N1).**
   - `.claude/coding-guidelines.md:33-44,75-77,120,133-137` are updated and add the attempt paragraph.
   - `findings.md` run commands are updated (`training/coco/router/findings.md:85,166,317-319`).
   - `git diff --stat -- features` touches only `features/training-reorg/*`.
   - One prose path in `findings.md:322` was missed (N1).

## Scope

The diff matches the plan's Files Changed table. There is one addition: `.claude/agents/implementer.md:29`. The plan allowed it conditionally ("only if it names the old paths"), and it did, so it is in scope. No false-positive import rewrites: every rewritten line is an actual `datagen`/`router` import, logger name or module reference. Prose uses of the word "router" (e.g. `coding-guidelines.md:27` "never inline in a router") were correctly left alone.

Old-path references left outside the diff:
- `database/migrations/env.py:25`: still names `training/datagen/run_simulation.py`. This path was already wrong before the move (it was missing `cli/`). See S1.
- `HANDOFF.md:35-36,64`: shows the original scaffolding tree (`datagen/`, `router/`). This is a dated, self-described "nothing is implemented yet" snapshot, so leaving it alone is right, and it is not raised as a finding. Note that `learnings.md` says HANDOFF.md has no old-path references, which isn't quite accurate.

Small process-record inaccuracy (not a finding): the plan says there are 11 `__file__`-relative lines. There are actually 13: 9 changed plus 4 unchanged (3 model paths and 1 figure path). The report says "two output paths" were left unchanged. All 13 are correct.

## Blockers

None.

## Suggestions

#### S1 — Stale simulator path in migrations comment

- **File:** `database/migrations/env.py:25`
- **Issue:** The comment names `training/datagen/run_simulation.py`, which doesn't exist after this move (and was already missing `cli/` before it). The plan's own Task 4 success criterion ("grep finds no remaining old-path references outside `features/` and `*.egg-info`") is therefore not met.
- **Fix:** Change the comment's path to `training/coco/datagen/cli/run_simulation.py`. This only edits a comment. It doesn't move the database layer or change `database/` behaviour, so it stays within the spirit of the spec's out-of-scope line.
- **Decision:** Accepted — addressed in "Address S1: fix the stale simulator path in the migrations comment"

## Nitpicks

#### N1 — Missed old-path reference in findings.md prose

- **File:** `training/coco/router/findings.md:322`
- **Issue:** "the generic-bootstrap refactor in `router/evaluation.py`" still uses the old path. Every other module or file reference in the file was prefixed, including the similar `training/coco/router/two_stage.py` at line 213. The learnings note that the implementer's rewrite rules targeted backtick-dotted and `training/`-prefixed forms, and this bare `router/x.py` form falls outside both.
- **Fix:** Change it to `coco/router/evaluation.py` (matching the `coco/router/baseline.py` style used in `evaluation.py:64` and `feature_experiment.py:5`).
- **Decision:** — _(pending)_

## Tests

- No tests were added or removed, as expected for a pure move. All 174 moved with their modules and still pass.
- The `caplog` logger-name strings in `training/tests/coco/datagen/cli/test_run_simulation.py:515,548,583,628` were correctly updated to `coco.datagen.cli.run_simulation`. These would have been easy to miss, because a stale name makes `caplog.at_level` silently target the wrong logger, and they are covered.
- No test pins the `__file__`-relative weights path or the `.env` path. This was acceptable before the move and is still acceptable at the prototype bar. The one-off `LOCAL_MODEL_WEIGHTS_PATH.exists()` check plus the byte-identical real-DB experiment runs (which need `training/.env` to resolve) cover it in practice.
- Nothing looks fragile. The before/after baseline approach (captured in the main tree, since worktrees can't be used) was sound and reproduced cleanly on re-run.

## Recommended Decisions

- **S1** — Accept — A one-line, comment-only fix. It closes the plan's own grep criterion and leaves no stale pointer for the next reader, with zero behaviour risk.
- **N1** — Accept — A one-word doc fix that makes `findings.md` consistent with every other reference already updated in the same file.
