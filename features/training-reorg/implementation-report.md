# Training Reorganisation — Implementation Report

**Feature:** Training Reorganisation
**Directory:** `features/training-reorg/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Capture the "before" baseline | completed |
| Task 2 — Move the code and tests | completed |
| Task 3 — Verify behaviour is unchanged | completed |
| Task 4 — Update layout documentation | completed |
| Task 5 — Regression test run | completed |

## Files changed

The range is `feature/router-frame-features..feature/training-reorg`.

### Moved (git renames; content edits limited to imports, paths and docstrings)

- `training/datagen/**` → `training/coco/datagen/**`
- `training/router/**` → `training/coco/router/**`, including `findings.md` and `latency_budget_curves.png`
- `training/tests/datagen/**` → `training/tests/coco/datagen/**`
- `training/tests/router/**` → `training/tests/coco/router/**`

### Added

- `training/coco/__init__.py`, `training/flowers/__init__.py`
- `training/tests/coco/__init__.py`, `training/tests/flowers/__init__.py`
- `features/training-reorg/spec.md`, `implementation.md`

### Edited

- `training/pyproject.toml`: the entry-point targets are now `coco.datagen.cli.*`, and package discovery is `coco*`, `flowers*`. The script names are unchanged.
- `.claude/coding-guidelines.md`: the repo layout now describes the two attempts, and all paths are updated.
- `.claude/agents/implementer.md`: the quality-gate paths are updated.
- `training/coco/router/findings.md`: run commands and module paths only.

## Tests

No tests were added or removed. The existing tests moved with the code, and their imports and logger names were updated. The count is 174 before and after, with none skipped. `database/` tests: 1 passed.

## Commits

`feature/router-frame-features..feature/training-reorg`: the spec and plan commits, then Task 2 (the move) and Task 4 (docs). Tasks 1, 3 and 5 changed no tracked files.

## Notable events

- **Before/after verification.** The baseline was captured before the move, and the comparison was re-checked independently by the orchestrator. After the move:
  - Both experiments' printed output is byte-identical.
  - The latency-budget figure is byte-identical.
  - Both command-line tools' `--help` output is byte-identical.
  - Ruff results are the same, and mypy reports the same 9 errors at their moved paths. Mypy's checked-file count went from 68 to 72, which is the four new package markers.
  - Re-running the latency-budget experiment at its default output path left the committed figure unchanged.
- **Paths relative to the file's location.** Nine lines had to change:
  - weights paths: `parents[1]` became `parents[2]`
  - seven `.env` loaders: one more `.parent` each

  Each was verified to resolve to the existing `training/models/` weights and `training/.env`. Two output paths were left unchanged, because they sit next to their files and moved with them.
- **Shared venv reinstall.** It was reinstalled with `pip install --no-deps -e training` so the editable install and entry points pick up the new packages.
- **One old-path reference remains outside this spec's scope.** It's a comment in `database/migrations/env.py:25`, which names `training/datagen/run_simulation.py`. That path was already wrong before the move; the file lived at `datagen/cli/run_simulation.py`. `database/` is out of scope for this spec, so it was left as is.
