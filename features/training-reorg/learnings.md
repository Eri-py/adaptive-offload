# Learnings — training-reorg

## Task 1 — Capturing the "before" baseline

- Both `router.feature_experiment` and `router.budget_experiment` write
  nothing to stderr on a clean run against the real database (both
  `*.stderr.txt` captures are empty / sha256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, the
  hash of an empty file). If a future baseline capture *does* see stderr
  output, that's a signal something changed (e.g. a new warning), not
  business as usual.
- `mypy .` from `training/` currently reports exactly 9 `[type-arg]` errors
  ("Missing type arguments for generic type \"ndarray\"") across
  `router/baseline.py` (x3), `router/analyze.py` (x1),
  `tests/datagen/sampling/test_complexity.py` (x3), and
  `tests/datagen/cli/test_score_complexity.py` (x2). This is the pre-existing
  baseline the reorg's success criteria references ("no errors beyond the 9
  already known") — after the move these should reappear at the same count,
  just under the new `training/coco/...` paths.
- `ruff check .` is clean (`All checks passed!`) before the move.
- Full baseline capture (174 tests passed, ruff clean, mypy 9 known errors,
  both experiments' stdout/figure, both CLI `--help` outputs) is saved under
  the session scratchpad's `reorg_before/` with a `hashes.txt` manifest for
  byte-identical comparison after the move.
- Encountered a transient "server-side auto mode classifier gave no verdict"
  error on the Bash tool mid-task (affected even trivial commands like
  `echo`), unrelated to command content — it cleared on its own after a few
  retries interspersed with non-Bash tool calls (Read). Worth knowing this
  can happen and isn't a sign the command itself was rejected.

## Task 2 — Move the code and tests

- `git mv training/router training/coco/router` moves the whole directory on
  disk (not just tracked files), so the gitignored `router/models/` leftover
  (saved `.joblib` baselines) rode along automatically — no separate plain
  `mv` was needed for it in practice, though it's still worth checking after
  the fact since the guidance can't assume the underlying `git mv`
  implementation's exact behavior.
- The bulk import rewrite (`from datagen` → `from coco.datagen`, `from
  router` → `from coco.router`) was 100% top-level `from X import` /
  `from X.sub import` statements — no bare `import datagen`/`import router`
  and no imports inside function bodies existed anywhere in this codebase, so
  a single anchored-at-column-0 sed pass covered every case with no
  false-positive risk from names like `run_router` or `router_utility`
  (grepped for those before and after; none exist).
- Docstring/comment references needed three distinct rewrite rules, not one:
  (1) backtick-dotted names like `` `datagen.config` `` → prefix `coco.`
  right after the opening backtick; (2) path-style references like
  `training/datagen` or `router/baseline.py` → insert `coco/` after
  `training/` or before `router/`; (3) two docstrings (`run_simulation.py`,
  `score_complexity.py`) spelled out the *fully-qualified* alternative form
  `training.datagen.*` (no backtick before `datagen`, dot-joined to
  `training`) as part of an "X-not-Y" convention explanation — this needed a
  fourth, separate literal replace (`training.datagen.*` →
  `training.coco.datagen.*`) since neither of the first two rules' anchors
  matched it. Easy to miss if you only pattern-match on backtick-prefixed or
  slash-prefixed forms.
- `logging.getLogger(__name__)` in `run_simulation.py` meant the `caplog`
  `logger="datagen.cli.run_simulation"` assertions in its test just needed
  the string literal updated to `"coco.datagen.cli.run_simulation"` — no
  separate check was needed to confirm the logger name itself changes
  correctly, since `__name__` tracks the module's new dotted path
  automatically once the import root changes.
- Of the 11 `__file__`-relative lines, 7 `load_dotenv` calls each needed
  exactly one more `.parent` (the file moved one directory deeper under
  `coco/`), and `datagen/config.py`'s two weight-path lines needed
  `parents[1]` → `parents[2]` for the same one-level reason (`parents[N]`
  counts from the file's own directory: before the move `parents[1]` from
  `training/datagen/config.py` was `training`; after the move, from
  `training/coco/datagen/config.py`, `parents[2]` is `training`). Confirmed
  by running the actual path check
  (`coco.datagen.config.LOCAL_MODEL_WEIGHTS_PATH.exists()` returned `True`
  post-move and post-reinstall), not just by counting parents on paper.
- After the `coco.` import rewrite, two import lines exceeded the 100-char
  line-length limit (`ruff` `E501`) purely because of the added `coco.`
  prefix. `ruff check --fix --select I001 .` (import-sort/format only)
  auto-wrapped both into multi-line `from X import (...)` form, which also
  resolved the `E501` since the wrapped form has no single over-length line —
  confirmed via `git diff` that the only change in both files was import
  formatting, nothing else.
- `pip install --no-deps -e training` reinstall after the `pyproject.toml`
  edit picked up the new `coco.*`/`flowers*` packages and entry points
  cleanly (uninstalled the old editable install, installed the new one) —
  no stale `.egg-info` or `.egg-link` cleanup was needed.
- Post-move: 174 tests passed (0 skipped), `ruff check .` clean, `mypy .`
  reports exactly the same 9 `[type-arg]` errors as the Task 1 baseline
  (3× `coco/router/baseline.py`, 1× `coco/router/analyze.py`, 3×
  `tests/coco/datagen/sampling/test_complexity.py`, 2×
  `tests/coco/datagen/cli/test_score_complexity.py`), just at the new paths —
  confirms the move introduced no new type errors.
