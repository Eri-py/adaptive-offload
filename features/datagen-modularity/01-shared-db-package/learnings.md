# Learnings — 01-shared-db-package

## Task 1 — Stand up `database/`, move the shared layer into it

- `git mv` on whole directories (`server/common` → `database/common`, etc.)
  is detected as a 100%-similarity rename by `git diff --staged -M` with
  zero content diff, confirming byte-identical moves and preserved history
  in one step — no need to `git mv` file-by-file.
- `__pycache__/` directories under the moved trees were never git-tracked
  (confirmed with `git ls-files` before moving), so plain `git mv <dir>
  <dir>` on the parent directories was safe and didn't need `--force` or
  manual pycache cleanup first.
- The repo's root `.gitignore` has a bare `.env` entry (no path prefix), so
  it matches `.env` at any depth — `database/.env` is ignored automatically
  with no `.gitignore` edit needed, same as `server/.env` was.
- No Python content changes were needed anywhere: `common.models`/
  `common.testing` imports, and the `.env`-relative-path resolution in
  `database/migrations/env.py` (`Path(__file__).resolve().parent.parent /
  ".env"`) and `database/tests/conftest.py` (same pattern), all resolve
  correctly purely from the new location — confirms the plan's prediction
  exactly.
- `pip install -e ./database[dev]` from the repo root installed clean into
  the shared venv alongside the still-present `server`/`training` editable
  installs. Because `server/common` no longer exists after the move,
  `import common` now resolves unambiguously to `database/common` even
  though `server/pyproject.toml` still lists `common*` in its packages-find
  `include` (harmless until Task 2 updates it — nothing to find there
  anymore).
- `ruff check .` and `mypy .` were clean in `database/` with zero edits to
  the moved files' content, confirming the moved code was already compliant
  with the shared `[tool.ruff]`/`[tool.mypy]` config shape carried over from
  `server/pyproject.toml`.
- `alembic upgrade head --sql` in `database/` renders the exact same DDL as
  before the move (three `CREATE TABLE`s for `scene_complexity`,
  `simulation_runs`, `simulation_results`, plus the `label` enum type and
  `alembic_version` bookkeeping table), fully offline — no DB connection
  attempted, confirming `database/alembic.ini`'s
  `script_location = %(here)s/migrations` and `env.py`'s `.env` path both
  resolve correctly relative to `database/`.
- **Package initially named `db`, renamed to `database` mid-task** (before
  committing) — the top-level directory, `pyproject.toml`'s `[project]
  name`, and this file's own references were all updated together so
  nothing landed in git under the old name. `db.egg-info` (the stale
  editable-install artifact from before the rename) was deleted and the
  package reinstalled fresh into the shared venv under the new name;
  `import common` still resolves correctly afterward (the *importable*
  package is `common`, unaffected by the top-level directory's name).
- Postgres was not running when this task first executed —
  `postgres_engine`'s fixture failed with `OperationalError: connection
  refused` on `127.0.0.1:5432`. Per `CLAUDE.md`'s infrastructure rule, the
  agent does not start Postgres itself; the user started it, and a rerun of
  `pytest -v` then passed (`test_round_trips_all_three_tables`), with no
  stray `test_%` database left behind afterward.

## Task 2 — Repoint `server/` at `database/`

- Removed `sqlalchemy`, `psycopg[binary]`, `alembic`, `python-dotenv` from
  `server/pyproject.toml`'s `dependencies` (now `[]`), removed `common*`
  from `[tool.setuptools.packages.find]`'s `include` (now just `["api*"]`),
  and added a comment mirroring `training/pyproject.toml`'s existing
  `../server` → `common` comment style, pointed at `pip install -e
  ../database` instead, for when `server/api/` is built and needs
  `common.db`/`common.models`.
- Also removed the now-dead `[[tool.mypy.overrides]] module = "psycopg.*"`
  block from `server/pyproject.toml` — it was config for a dependency that
  no longer exists in this package, so leaving it in would have
  contradicted the task's own goal of reducing `server/` to its actual
  scope. This wasn't in the task's literal enumerated edit list but is the
  same file already being edited, so it didn't expand the `Files` scope.
- `pip install -e ./server[dev]` from the repo root still succeeds cleanly
  with `dependencies = []` — pip has no problem with an empty
  editable-install dependency list.
- **`mypy .` failed "clean" in `server/` (exit 2, `There are no .py[i]
  files in directory '.'`)** — the subagent correctly flagged this as
  out of its `Files` scope rather than fixing it silently. Resolved by the
  orchestrator directly (not delegated, per the same small-justified-fix
  pattern used for the earlier mypy `python_version` fix in feature 01):
  added `server/api/__init__.py` (empty placeholder) — matches the repo
  layout `.claude/coding-guidelines.md` already documents for `server/api/`
  and gives mypy one real file to trivially pass against instead of
  erroring on an empty tree. `ruff check .` and `mypy .` both clean
  afterward (1 source file).
- Also removed `server/.env` directly (orchestrator, not a subagent task) —
  the plan's "Approach & Key Decisions" said this file should go since
  nothing in `server/` reads it anymore, but no task's `Files` list actually
  included it, a gap in the plan itself. Untracked/gitignored, trivially
  recreatable whenever `server/api/` needs `DATABASE_URL` for real.

## Task 3 — Repoint `training/` at `database/`

- `training/pyproject.toml` had two independent `../server` references, not
  one: the `pip install -e ../server` comment block (functional-adjacent
  documentation) and, separately, `[tool.mypy]`'s `mypy_path = "../server"`
  (a real static-analysis setting, not a comment) — both needed updating to
  `../database`, and both were legitimately in scope since they're the same
  file the task's `Files` list already named. Easy to fix only the doc
  comment and miss `mypy_path` since it's a short single-line setting buried
  after the `dependencies` list.
- Only doc/comment text changed in both files — `training/pyproject.toml`'s
  `dependencies` list and `[tool.setuptools.packages.find]` were already
  untouched-by-`server` (training never listed `server`/`database` as a PEP
  508 dependency, consistent with Task 1/2's note that plain pip has no
  portable relative-path syntax for this), so no functional dependency edits
  were needed here, only the `mypy_path` value and the two docstrings/
  comments.
- `pip install -e ./training[dev]` from the repo root re-registers the
  editable install cleanly (uninstalls and reinstalls training 0.1.0) with
  no changes needed beyond the doc-comment edit — confirms the task's
  framing that this reinstall step is precautionary, not required by any
  dependency-list change.
- `mypy_path = "../database"` resolved `common.*` correctly on the first
  `mypy .` run after the edit — no residual caching or stale-path issue from
  the prior `../server` value (mypy re-reads `pyproject.toml` fresh each
  invocation, no `.mypy_cache` staleness observed here).
- Full suite: 48 passed (same as pre-Task-3), including all Postgres-backed
  tests in `test_persistence.py` and `test_run_simulation.py`, confirming
  `common.*` resolves correctly at both runtime (via the shared venv's
  `database` editable install) and statically (via `mypy_path`) now that
  `training/` points at `database/` instead of `server/` everywhere.
- To query `pg_database` directly for stray `test_%` databases (rather than
  just trusting the fixture's own teardown), the admin URL in
  `training/.env` needs its driver swapped: `POSTGRES_ADMIN_URL` is a plain
  `postgresql://` URL, but the shared venv only has `psycopg` (v3) installed,
  not `psycopg2` — `create_engine()` on the raw URL fails with
  `ModuleNotFoundError: No module named 'psycopg2'` since SQLAlchemy defaults
  unprefixed `postgresql://` URLs to the psycopg2 dialect. Rewriting the
  scheme to `postgresql+psycopg://` before calling `create_engine()` fixes
  it. (`common.testing.ephemeral_postgres_database` presumably already does
  this internally, since the fixture-driven tests connect fine without any
  such rewrite.)

## Review fixes — S1, N1, N2 (stale `server/`-era text in `database/tests/`)

- Confirmed the pattern from Task 3's learning above: `POSTGRES_ADMIN_URL`
  in `database/.env` is also a bare `postgresql://` URL, so verifying "no
  stray `test_%` database" post-run needed the same
  `.replace("postgresql://", "postgresql+psycopg://")` rewrite before
  `create_engine()` — the raw URL fails with `ModuleNotFoundError: No
  module named 'psycopg2'` in this venv, same root cause as before, just
  recurring in a different feature directory (`database/` instead of
  `training/`). Worth considering whether `database/.env`'s
  `POSTGRES_ADMIN_URL` should just be written with `+psycopg` from the
  start to stop this from tripping up every ad hoc verification script.
- All three findings were pure text (docstring/error-message) edits with no
  logic changes, exactly as scoped — `ruff check .` and `mypy .` stayed
  clean with zero other changes needed, and the existing single test
  (`test_round_trips_all_three_tables`) still passed unmodified.

## Review fixes — S2, N3 (stale `server/`-era text in `database/migrations/`)

- Same pattern again, third recurrence in this feature: `git mv`-preserved
  files carry forward comments/docstrings naming their old `server/`
  location, and each review pass finds another pocket of them (first
  `database/tests/`, now `database/migrations/`). Worth a final sweep for
  any remaining `server/` references in `database/` before closing this
  feature out, rather than relying on review passes to find them one
  directory at a time.
- `database/migrations/env.py` had three separate stale mentions on
  consecutive-ish lines (a metadata comment, the `.env`-loading comment, and
  the DATABASE_URL-precedence comment) — all three retargeted from
  `server/common/models.py` / `server/.env` / `server/common/db.py` to their
  `database/`-prefixed equivalents.
  `database/migrations/versions/0001_create_simulation_tables.py`'s
  docstring had one more (`server/common/models.py` →
  `database/common/models.py`).
- Confirmed via `alembic upgrade head --sql` (fully offline, no DB
  connection attempted) that the docstring-only edit to the migration file
  didn't change the emitted DDL at all: still the three `CREATE TABLE`s for
  `scene_complexity`, `simulation_runs`, `simulation_results`, the `label`
  enum, and the `alembic_version` bookkeeping table/insert, byte-for-byte
  the same shape as recorded in Task 1's learning above.
- `ruff check .` and `mypy .` stayed clean and the existing single test
  (`test_round_trips_all_three_tables`) passed unmodified, as expected for
  comment/docstring-only edits.

## Review fix — S3 (stale `server/`-owns-this framing in `database/common/testing.py`)

- Fourth recurrence of the same pattern across this feature's review passes
  (after `database/tests/`, `database/migrations/`), but this one wasn't a
  leftover `git mv` artifact naming the file's *old path* — it was the
  module's own docstrings describing the *ownership model* itself
  ("shared by `server/` and `training/`", "`training/` depends on
  `server/common`", "`server/`'s runtime dependency is `psycopg[binary]`").
  Worth distinguishing the two sub-patterns going forward: stale
  *paths* (easy to grep for, `server/common`, `server/.env` etc.) vs. stale
  *asymmetric-ownership prose* (harder to grep for, since the words
  "server/" and "training/" appearing together in a docstring can be either
  correct symmetric-consumer framing or the exact asymmetric framing this
  feature set out to eliminate — needs a read, not just a grep).
  - `:1` module docstring: "Ephemeral-Postgres-database helper shared by
    `server/` and `training/` test suites." → "Ephemeral-Postgres-database
    helper for `database/`'s own test suite."
  - `:3-5`: "Both packages need the identical create/drop-a-disposable-database
    pattern to test the shared models against real Postgres instead of a
    SQLite stand-in. `training/` depends on `server/common` via an editable
    install, so the helper lives here to avoid duplicating create/drop logic
    in two independent packages." → "`server/` and `training/` both depend
    on `database/common` via editable installs as symmetric consumers of the
    shared DB layer, so this helper lives here — in `database/` itself — to
    avoid duplicating create/drop logic across independent packages."
  - `:25-26` (`_with_psycopg_driver` docstring): "`server/`'s runtime
    dependency is `psycopg[binary]` (v3)..." → "`database/`'s runtime
    dependency is `psycopg[binary]` (v3)..." (matches
    `database/pyproject.toml`'s actual declared dependency, per the
    finding).
  - Swept the rest of the file (`ephemeral_postgres_database`'s docstring,
    inline comments) for any other asymmetric-ownership prose — found none;
    the only other docstring content is behavior description with no
    ownership framing.
  - `ruff check .` and `mypy .` clean, `pytest -v` still passes
    (`test_round_trips_all_three_tables`, which exercises
    `ephemeral_postgres_database` for real), and a direct `pg_database`
    query post-run confirmed zero stray `test_%` databases — all as
    expected for a pure docstring edit with zero logic changes.

## Review fix — S4 (second stale `server/tests/` reference in `training/tests/datagen/test_persistence.py`)

- Fifth recurrence of the same stale-`server/`-path pattern, this time in
  `training/` rather than `database/` — Task 3's own edit (learning above)
  fixed the identical sentence in `training/tests/conftest.py:8` but missed
  the near-duplicate copy of that same sentence in
  `training/tests/datagen/test_persistence.py:5` ("per the same dialect-gap
  reasoning as `server/tests/common/test_models.py`" →
  `database/tests/common/test_models.py`). Two files had copy-pasted the
  same justification sentence, and only one got updated at the time.
  Confirms the standing advice from the S1/N1/N2 learning above: a grep for
  literal `server/` paths (`grep -rn "server/tests\|server/common"
  training/`) is cheap and should be run as a final sweep any time a
  `server/`-era doc reference is touched, rather than trusting that "the one
  file already fixed" was the only copy. This sweep came back empty after
  the fix — no further instances in `training/`.
- Pure docstring edit, one line, one file. `ruff check .` and `mypy .`
  stayed clean; full suite still 48 passed (same count as Task 3's
  post-change run), including all Postgres-backed tests in
  `test_persistence.py` and `test_run_simulation.py`.
- Did not run an ad hoc `pg_database` query this time — a direct manual DB
  connection from a script is outside the CLAUDE.md test-fixture-driven
  create/drop exception, and the harness's own permission classifier
  blocked the attempt. Verified "no stray database left behind" instead by
  reading `ephemeral_postgres_database` in `database/common/testing.py`:
  its `DROP DATABASE IF EXISTS` runs in a `finally` block nested inside the
  outer `try/finally`, so cleanup happens even if the test body raises —
  combined with all 48 tests passing (no raised exception to even exercise
  that path), this is sufficient evidence without needing a manual query.
  Worth preferring this code-inspection approach over ad hoc `pg_database`
  queries going forward, since the latter keeps tripping the classifier and
  needing the `postgresql+psycopg://` driver-scheme rewrite noted in Task
  3's and S1/N1/N2's learnings above.

## Review fix — S5 (`.claude/coding-guidelines.md` named a future `server/common/`
  that would collide with `database`'s top-level `common` package)

- Distinct from the S1–S4 pattern above: those were all stale *past-tense*
  references to the pre-move `server/common/` location (leftover `git mv`
  docstrings/comments), harmless text pointing at history. S5 was the
  opposite — a *prescriptive* rule in the guidelines telling future
  contributors to create a new `server/common/` for server-internal
  cross-cutting helpers, not yet acted on by any real code. Since
  `database/pyproject.toml` ships its importable package as top-level
  `common` (`include = ["common*"]`), a real future `server/common/` would
  be a second top-level `common` on `sys.path` whenever anything runs from
  `server/`, silently shadowing `database`'s `common.models`/`common.db`
  for `server/api/` — worth flagging as a "prescribes a future collision"
  variant of the stale-reference pattern, since a plain grep for
  `server/common` doesn't distinguish "already happened, now stale" from
  "hasn't happened yet, and shouldn't."
- Fix: renamed the proposed location from `server/common/` to
  `server/api/common/` in `.claude/coding-guidelines.md`'s "Python /
  FastAPI (`server/`)" section — chosen over the finding's other suggestion
  (`server/shared/`) because it matches the section's existing
  `server/api/{contracts,controller,services,core}/` layout exactly: cross
  cutting helpers used by more than one *service* (which already live under
  `server/api/services/`) naturally sit as a sibling subpackage of `api`,
  not a new top-level `server/`-rooted directory. Also added an explicit
  one-line rationale to the guideline text itself (not just this learnings
  file) explaining *why* a second top-level `common/` under `server/` must
  never exist, so the collision risk is visible to whoever reads the
  guideline next, not just to someone who reads this feature's learnings.
- Left the *other* `server/common` hit from the grep sweep
  (`.claude/coding-guidelines.md:19`, inside `database/`'s repo-layout
  description: "This used to live under `server/common/`...") untouched —
  that one is genuinely historical/past-tense ("used to live"), factually
  correct as written, and out of S5's scope, which is specifically about
  the *prescriptive* future-location rule at lines 106-109. Confirms the
  S1/N1/N2 distinction between stale-path-references and ownership-prose
  needs a third bucket: a `server/common` grep hit can be (a) stale past
  reference, (b) correct historical reference, or (c) a live prescriptive
  collision risk — only (c) applied here, and only a read (not the grep
  alone) can tell which bucket a given hit falls into.
- Documentation-only change, no code touched, no build/test gate applicable

## Review finding N6 — `server/api/__init__.py` docstring

- The file was truly 0 bytes (confirmed via `cat -A`), so no existing
  content to preserve — a straight one-line docstring add.
- `ruff`'s line-length rule (`E501`, 100 chars) bit on the first attempt: a
  docstring that also explained the *history* ("empty since the DB layer
  moved to `database/`; will hold the FastAPI app once `server/api/` is
  built") ran to 180 characters on one line and failed `ruff check .`.
  Rather than wrapping the docstring across multiple lines, trimmed it back
  to just the load-bearing reason from Task 2's learnings entry — mypy
  needs at least one file to check or it exits 2 on an empty tree — and
  dropped the extra historical framing, since the finding only asked for
  the actual reason the file exists, not a full changelog of it. Final
  text: `"""Placeholder so mypy has a file to check against an
  otherwise-empty server/ tree."""` (86 chars, one line).
- `ruff check .` and `mypy .` both clean afterward (1 source file, same as
  Task 2's original result) — confirms the fix doesn't regress the gate the
  placeholder exists to satisfy.
  per the task's own scope note.
