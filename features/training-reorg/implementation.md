# Training Reorganisation — Implementation Plan

## Summary

Move all existing training code (`training/datagen/`, `training/router/`) and
its tests under a new `training/coco/` package, as attempt 1, and add an
empty `training/flowers/` package for attempt 2. This is a pure move. Tests,
the experiments' printed output and figure, and the two command-line tools
must behave exactly as before.

## Approach & Key Decisions

- **Import paths name the attempt.** Modules become `coco.datagen.*` and
  `coco.router.*`, updated everywhere: source, tests, logger names, docstrings
  and `pyproject.toml`. An alternative was to keep `datagen`/`router` as import
  roots by remapping package directories, but then flower code importing
  `router.*` would look shared when it isn't. That defeats the point of the
  separation.
- **File-relative paths are fixed explicitly.** 11 lines compute paths from
  `__file__`:
  - `datagen/config.py` weights paths, with `parents[1]` → `parents[2]`
  - CLI and router `load_dotenv` calls, one more `.parent` each
  - `baseline.py` and `budget_experiment.py` output paths, which move with
    their files and stay unchanged

  `training/data/`, `training/models/` and `training/.env` stay where they
  are. `tests/conftest.py` is unchanged, because the tests root doesn't move.
- **"Before" output is captured first.** Git worktrees can't be used for
  before/after comparisons here: the editable install always imports the main
  tree (spec 03 learnings). So Task 1 records both experiments' output and the
  figure before anything moves.
- **The move uses `git mv`, so history follows the files.** The gitignored
  `router/models/` directory (saved baseline models) is moved with a plain
  `mv`. After the move, the training package is reinstalled with
  `pip install --no-deps -e training` into the shared venv. That reinstall is
  what makes the entry points and the editable import finder see the new
  packages. It isn't infrastructure: nothing is started, stopped or migrated
  (`CLAUDE.md` "Infrastructure").
- **Documentation:** `.claude/coding-guidelines.md`, `.claude/agents/implementer.md`
  (if it names these paths), and the run commands in the moved `findings.md`
  are updated. Historical documents under `features/` are not touched.

## Out of Scope

- Behaviour changes, refactors, renames beyond the move, and bug fixes.
- A `shared/` package, or splitting any file between attempts.
- Flower code.
- Moving `training/data/`, `training/models/`, `database/`, `server/` or `app/`.
- Editing historical documents under `features/`.
- Schema or database changes.

## Dependencies and Configuration

- `training/pyproject.toml`:
  - `[project.scripts]` targets become `coco.datagen.cli.run_simulation:main`
    and `coco.datagen.cli.score_complexity:main`; the script names are unchanged
  - `[tool.setuptools.packages.find] include` becomes `["coco*", "flowers*"]`
- Reinstall: `pip install --no-deps -e training` from the repo root, using the shared `.venv`.
- No new packages and no migrations.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/coco/__init__.py` | add | Package marker for attempt 1 | New package |
| `training/coco/datagen/**` | move (from `training/datagen/`) | Simulator, as attempt 1 | Separate the attempts |
| `training/coco/router/**` | move (from `training/router/`), including `findings.md` and the figure | Router code, as attempt 1 | Separate the attempts |
| `training/flowers/__init__.py` | add | Empty package for attempt 2 | Ready for the pivot |
| `training/tests/coco/__init__.py` | add | Test package marker | Tests mirror the source |
| `training/tests/coco/datagen/**`, `training/tests/coco/router/**` | move (from `training/tests/datagen/`, `training/tests/router/`) | Tests mirror the new layout | Coding guidelines |
| `training/tests/flowers/__init__.py` | add | Test package marker for attempt 2 | Mirror the source |
| `training/pyproject.toml` | edit | Entry-point targets, package discovery | New import paths |
| `.claude/coding-guidelines.md` | edit | Describe the new layout | Documentation matches code |
| `.claude/agents/implementer.md` | edit (only if it names the old paths) | Same | Documentation matches code |

## Tasks

### Task 1 — Capture the "before" baseline

- **Objective:** Record current behaviour, so the move can be verified
  against it.
- **Files:** none changed in the repo. Output goes to the session scratchpad
  (`/tmp/claude-1000/-home-eriol-projects-adaptive-offload/6382f155-1db6-4d0a-9dfe-2174de0c758a/scratchpad/reorg_before/`).
- **Details:** From `training/`, with the shared venv, run these read-only
  against the real database:
  - `python -m router.feature_experiment` → save stdout to `feature_experiment.txt`
  - `python -m router.budget_experiment --figure <scratch>/reorg_before/latency_budget_curves.png`
    → save stdout to `budget_experiment.txt`
  - `pytest -q` → save the pass count
  - `run-simulation --help` and `score-complexity --help` → save both outputs
  - `ruff check .` and `mypy .` → save the results

  Record SHA-256 hashes of every saved file in `hashes.txt`. Don't start or
  stop Postgres.
- **Success criteria:**
  - All captures exist.
  - The tests show 174 passed.
  - `git status` shows no changes to tracked files.

### Task 2 — Move the code and tests

- **Objective:** Move source and tests under `coco/`, add the empty `flowers`
  packages, and update every import, path and config so the suite passes.
- **Files:** everything in the Files Changed table except the two `.claude/` docs.
- **Details:**
  - Use `git mv training/datagen training/coco/datagen` and
    `git mv training/router training/coco/router` (tests likewise into
    `training/tests/coco/`). Plain `mv` any untracked or ignored leftovers,
    such as `training/router/models/`, so no old directory remains. Delete
    stale `__pycache__` directories.
  - Add `__init__.py` to `training/coco/`, `training/flowers/`,
    `training/tests/coco/` and `training/tests/flowers/`. Each gets a
    one-line docstring naming the attempt.
  - Rewrite imports: `datagen.` → `coco.datagen.` and `router.` →
    `coco.router.`, in `from`/`import` statements and inside functions.
    Also update logger names in tests (`caplog` `logger=`), module paths in
    docstrings (`python -m router.x` → `python -m coco.router.x`), and any
    other string that names a module.
  - Fix the `__file__`-relative paths so they still resolve to
    `training/models/`, `training/.env` and so on. Verify with a one-off
    Python check: print `coco.datagen.config.LOCAL_MODEL_WEIGHTS_PATH` and
    confirm it exists.
  - Update `training/pyproject.toml` as described under Dependencies. Then run
    `pip install --no-deps -e training` from the repo root with the shared venv.
- **Success criteria:**
  - Nothing remains at `training/datagen/`, `training/router/`,
    `training/tests/datagen/` or `training/tests/router/`.
  - `pytest -q` passes 174, with none skipped.
  - `ruff check .` is clean, and `mypy .` shows only the 9 known errors.
  - `python -c "import coco.datagen.config, coco.router.budget_experiment, flowers"` works.

### Task 3 — Verify behaviour is unchanged

- **Objective:** Prove the move changed nothing.
- **Files:** none changed.
- **Details:**
  - Re-run every Task 1 capture with the new module paths
    (`python -m coco.router.feature_experiment`, `python -m coco.router.budget_experiment --figure <scratch>/reorg_after/...`).
  - Compare with the baseline:
    - stdout for both experiments: byte-identical
    - figure: byte-identical (`cmp`)
    - `--help` output: identical, or differing only where it prints the
      program path (report exactly any differences)
    - test count: 174 both times
    - ruff and mypy: the same results
  - Also run `python -m coco.router.budget_experiment` once without
    `--figure`. It writes the committed figure at its new path, and
    `git status` must show that file unchanged.
- **Success criteria:**
  - Every comparison matches.
  - Any difference is reported and explained, not waved through.

### Task 4 — Update layout documentation

- **Objective:** Make the docs describe the new layout.
- **Files:** `.claude/coding-guidelines.md`, `.claude/agents/implementer.md`
  (only if it names the old paths), `training/coco/router/findings.md` (run
  commands only).
- **Details:**
  - Replace `training/datagen/` → `training/coco/datagen/` and
    `training/router/` → `training/coco/router/`, and `python -m router.` /
    `datagen.` module paths.
  - Add one short paragraph to the guidelines' repo layout: `training/coco/`
    is attempt 1 (COCO object detection) and `training/flowers/` is attempt 2
    (flower species identification). Code reused by both moves into a shared
    location when it's first reused.
  - Don't touch anything under `features/`.
- **Success criteria:**
  - `grep` finds no remaining old-path references outside `features/` and
    `*.egg-info`.
  - No file under `features/` is modified, apart from this feature's own
    workflow documents.

### Task 5 — Regression test run

- **Objective:** Run every test that exercises code added or modified in
  this plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest` passes (174), and `cd database && pytest` passes.
  - ruff is clean, and mypy shows only the 9 known errors.
