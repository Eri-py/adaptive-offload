# Review — Shared DB Package

## Verdict

The move is correct and behavior-preserving. All twelve moved files land as
100%-similarity git renames with a literally empty content diff
(`git diff -M feature/data-gen-router-loop...feature/datagen-modularity`),
so models, engine, fixture helper, and migration are byte-identical to their
pre-move versions — the strongest possible evidence that nothing about their
behavior changed. I re-verified independently: `import common` resolves to
`/home/eriol/projects/adaptive-offload/database/common/__init__.py`,
`ruff`/`mypy` are clean in all three packages, `database/` renders the
migration offline with no DB connection (three `CREATE TABLE`s, the `label`
enum, `alembic_version`), and the suites pass at 1 (`database/`) + 0
(`server/`) + 48 (`training/`) = 49, matching the pre-move total. Every
consumer was repointed, including the non-obvious one the plan understated
(`training/pyproject.toml`'s `mypy_path`, a real static-analysis setting, not
just a comment). No blockers.

The one systematic weakness is the flip side of the zero-content-change
strategy: the moved files still carry a ring of `server/`-era path references
in their docstrings and comments — and in one case in a runtime error message
that now tells the operator to edit a file that no longer exists. None of it
breaks anything, but the whole point of this feature was to stop `server/`
looking like the owner of the DB layer, and the moved code still says it does.

## Acceptance Criteria

1. **Imports resolve from each package's own environment — MET.**
   `python -c "import common"` in the shared venv resolves to
   `database/common/__init__.py:1`. `training/pyproject.toml:51` points
   `mypy_path` at `../database`, and `mypy .` in `training/` resolves all
   eight `common.*` import sites (`training/datagen/labeling.py:11`,
   `persistence.py:13`, `run_simulation.py:29`, plus five test modules)
   cleanly. `database/alembic.ini:21` (`prepend_sys_path = .`) plus the
   editable install cover `database/migrations/env.py:9`.

2. **`server/` suite: same tests passing, same count — MET, with a caveat.**
   The one test that lived in `server/` (`server/tests/common/test_models.py`,
   1 test) moved to `database/tests/common/test_models.py` and still passes;
   `server/` now collects 0. Read literally, `server/`'s count went 1 → 0;
   read as intended (no test lost, and the test follows the code it exercises),
   it holds — and the plan explicitly reassigned it
   (`implementation.md:26-28`). Total across the repo is 49 before and after.

3. **`training/` suite: same tests passing, same count — MET.** Verified:
   `48 passed`, including the Postgres-backed `test_persistence.py` and
   `test_run_simulation.py` that exercise `common.*` through the new location.

4. **Migration renders identical DDL offline — MET.** I ran
   `alembic upgrade head --sql` from `database/`: it emits `alembic_version`,
   `scene_complexity`, `simulation_runs`, `CREATE TYPE label AS ENUM
   ('LOCAL', 'OFFLOAD')`, and `simulation_results` with its FK, with no
   connection attempted. `database/alembic.ini:8`
   (`script_location = %(here)s/migrations`) is `%(here)s`-relative, and
   `database/migrations/env.py:28` resolves `.env` from `__file__`, so both
   followed the move with no edit. Byte-identical `alembic.ini`, `env.py`, and
   `0001_create_simulation_tables.py` make DDL identity a certainty, not an
   inference.

5. **`.claude/coding-guidelines.md` updated — MET.** `database/` is a
   top-level area at `.claude/coding-guidelines.md:14-20`, the symmetric
   editable-install relationship is stated there and at `:41-57`, and the
   Database section at `:67-70` no longer claims `server/common/` owns the
   layer. See S5 for a forward-looking hazard the rewrite introduced.

6. **Old locations no longer contain the moved code — MET.** `server/` now
   holds only `pyproject.toml` and `api/__init__.py` (plus gitignored tool
   caches); `server/common/`, `server/migrations/`, `server/alembic.ini`, and
   `server/tests/` are all gone from the tree and from
   `git ls-tree feature/datagen-modularity server/`.

## Scope

The diff matches the plan's Files Changed table. Four items sit outside the
literal table, all documented in `learnings.md` and all justified:

- `server/api/__init__.py` (added) — an empty placeholder so `mypy .` in
  `server/` doesn't exit 2 on a zero-`.py` tree. Not in the table; a genuine
  plan gap, fixed the right way. See N6.
- `server/pyproject.toml` — the `[[tool.mypy.overrides]] module = "psycopg.*"`
  block and the `description` string were also updated. Beyond the table's
  literal wording, but in the same file and required for the table's stated
  purpose ("reflects `server/`'s actual scope").
- `training/pyproject.toml:51` — `mypy_path` retargeted from `../server` to
  `../database`. The plan framed this file as a comment-only edit; it was not,
  and missing it would have broken `mypy` in `training/`. Good catch.
- `server/.env` deleted (untracked, not in the diff) — named in the plan's
  Approach but never assigned to a task.

`features/datagen-modularity/02-datagen-cli-tools/spec.md` also appears in the
branch diff (from commit `066a5e7`); it is the companion spec, not work from
this plan.

No unintended scope drift. No schema change, no live-database access, no
change to `training/datagen/`'s simulation logic.

## Blockers

None.

## Suggestions

#### S1 — Runtime error message still directs the operator to `server/.env`

- **File:** `database/tests/conftest.py:35`
- **Issue:** The `RuntimeError` raised when `POSTGRES_ADMIN_URL` is unset says
  `"POSTGRES_ADMIN_URL is not set. Add it to server/.env"`. That file was
  deleted by this very plan; the fixture loads
  `Path(__file__).resolve().parent.parent / ".env"` (`:21`), i.e.
  `database/.env`. This is operator-facing text at the exact moment someone is
  confused, not a stale comment.
- **Fix:** Change `server/.env` to `database/.env` in the message.
- **Decision:** Accepted — addressed in "Address S1, N1, N2: fix stale server/ references in database/tests/"

#### S2 — `migrations/env.py` comments still name `server/.env` and `server/common/`

- **File:** `database/migrations/env.py:20`, `:23`, `:30`
- **Issue:** `:20` says the metadata "matches server/common/models.py exactly";
  `:23` says "Load DATABASE_URL from server/.env if it isn't already in the
  environment"; `:30` says "Prefer DATABASE_URL (per server/common/db.py's
  convention)". All three now name paths that don't exist. `:23` is the
  actively misleading one — it documents which `.env` this file reads, and it
  names the wrong file.
- **Fix:** Retarget all three to `database/common/models.py`, `database/.env`,
  and `database/common/db.py`.
- **Decision:** Accepted — addressed in "Address S2, N3: fix stale server/ references in database/migrations/"

#### S3 — `common/testing.py`'s docstrings still describe `server/` as the owner

- **File:** `database/common/testing.py:1`, `:5`, `:25`
- **Issue:** `:1` describes the helper as "shared by `server/` and `training/`
  test suites" (the suite that actually uses it now lives in `database/`);
  `:5` says "`training/` depends on `server/common` via an editable install,
  so the helper lives here"; `:25` says "`server/`'s runtime dependency is
  `psycopg[binary]`" — that dependency is now declared in
  `database/pyproject.toml:8` and was explicitly removed from `server/`. The
  `server/`-owns-the-DB-layer framing this feature set out to eliminate
  survives verbatim inside the moved file.
- **Fix:** Rewrite the three references so the docstrings describe `database/`
  as the home and `server/`/`training/` as symmetric consumers.
- **Decision:** Accepted — addressed in "Address S3: rewrite testing.py docstrings to describe database/ as owner"

#### S4 — A second `training/` test docstring pointing at `server/tests/` was missed

- **File:** `training/tests/datagen/test_persistence.py:5`
- **Issue:** Says "reasoning as `server/tests/common/test_models.py`". Task 3
  updated the identical stale reference in `training/tests/conftest.py:8` but
  not this one — same package, same class of edit, and the file is one `git
  grep` away from the one that was fixed.
- **Fix:** Point it at `database/tests/common/test_models.py`.
- **Decision:** Accepted — addressed in "Address S4: fix stale server/tests/ reference in test_persistence.py"

#### S5 — Guidelines reintroduce a `server/common/` that would shadow `database`'s `common`

- **File:** `.claude/coding-guidelines.md:106-109`
- **Issue:** The rewritten rule now says server-internal cross-cutting helpers
  "go in a `server/common/` module (create it if it doesn't exist yet) … Keep
  this separate from `database/`". The separation of *concerns* is right, but
  the *name* collides: `database`'s importable package is top-level `common`
  (`database/pyproject.toml:25`, `include = ["common*"]`). A future
  `server/common/` would be a second top-level `common` on `sys.path` whenever
  anything runs from `server/`, silently shadowing the DB layer for
  `server/api/` — the failure mode is an `ImportError` on `common.models` that
  looks like an install problem. The pre-move guidelines could safely use that
  name because the two were the same package; after this move they aren't.
- **Fix:** Name the future server-internal location something that can't
  collide — e.g. `server/api/common/` (a subpackage of `api`, which is what
  `packages.find` already ships) or `server/shared/`.
- **Decision:** Accepted — addressed in "Address S5: rename future server/common/ to avoid collision with database's common package"

## Nitpicks

#### N1 — `conftest.py` module docstring still says it serves `server/tests/`

- **File:** `database/tests/conftest.py:1`
- **Issue:** `"""Shared pytest fixtures for `server/tests/`."""` — the fixtures
  now serve `database/tests/`.
- **Fix:** Update to `database/tests/`.
- **Decision:** Accepted — addressed in "Address S1, N1, N2: fix stale server/ references in database/tests/"

#### N2 — Test package marker still says `server/common`

- **File:** `database/tests/common/__init__.py:1`
- **Issue:** `"""Test package for `server/common`."""`
- **Fix:** Update to `database/common`.
- **Decision:** Accepted — addressed in "Address S1, N1, N2: fix stale server/ references in database/tests/"

#### N3 — Migration docstring credits `server/common/models.py`

- **File:** `database/migrations/versions/0001_create_simulation_tables.py:7`
- **Issue:** "Creates the three tables owned by `server/common/models.py`".
- **Fix:** Update to `database/common/models.py`. Safe: the revision is
  unapplied, and a docstring edit changes neither the revision id nor the
  emitted DDL — I confirmed the offline render is unaffected by comment text.
- **Decision:** Accepted — addressed in "Address S2, N3: fix stale server/ references in database/migrations/"

#### N4 — `HANDOFF.md` still says `server/common/` will own the DB layer

- **File:** `HANDOFF.md:68`
- **Issue:** "`server/common/` will own the SQLAlchemy engine/models;
  `training/` imports from there." Contradicts the new layout.
- **Fix:** Either add a one-line "superseded — see `database/`" note, or leave
  it: the file reads as a dated record of one session's decisions, not living
  documentation, and the spec only required `.claude/coding-guidelines.md` to
  be updated.
- **Decision:** Declined — `HANDOFF.md` is a dated session record, not living docs; the spec scoped the doc update to `.claude/coding-guidelines.md`, and rewriting history-shaped files invites drift of its own.

#### N5 — `database/pyproject.toml` type-checks at 3.12 while claiming `>=3.11`

- **File:** `database/pyproject.toml:5`, `:29`, `:35`
- **Issue:** `requires-python = ">=3.11"` and ruff `target-version = "py311"`,
  but mypy `python_version = "3.12"` — the package advertises 3.11 support that
  is never type-checked. Faithfully copied from `server/pyproject.toml:5,29,35`
  (the plan asked for exactly that), and `training/` has the same shape, so
  this is a pre-existing repo-wide inconsistency, not one this plan introduced.
- **Fix:** If touched at all, align all three packages at once — not here.
- **Decision:** Declined — pre-existing and identical across all three packages; the plan explicitly required carrying server/'s config shape over unchanged, and fixing it in one package only would create real drift to fix a cosmetic one.

#### N6 — Empty `server/api/__init__.py` gives no hint why it exists

- **File:** `server/api/__init__.py:1`
- **Issue:** Zero bytes. It exists to stop `mypy .` exiting 2 on an otherwise
  empty `server/` tree; nothing in the file or the manifest says so, and the
  next person to look at an empty `server/` may delete it and rediscover the
  error.
- **Fix:** One-line module docstring, e.g. `"""FastAPI app package — empty
  until server/api/ is built."""`.
- **Decision:** Accepted — addressed in "Address N6: document why server/api/__init__.py exists"

## Tests

No tests were added, which is right: this was a structural move, and the plan
said so. What changed is location, not content —
`database/tests/common/test_models.py` is byte-identical to its `server/`
predecessor and still exercises the round-trip across all three ORM models plus
the run→result FK against a real ephemeral Postgres database.

Verified independently for this review:

- `cd database && pytest -q` → `1 passed`
- `cd training && pytest -q` → `48 passed` (includes the Postgres-backed
  `test_persistence.py` and `test_run_simulation.py`, which are what actually
  prove `common.*` resolves from `database/` at runtime)
- `ruff check .` and `mypy .` clean in `database/` (10 files), `server/`
  (1 file), and `training/` (22 files)
- `alembic upgrade head --sql` from `database/` renders fully offline

Coverage gaps, none of them regressions from this plan:

- `server/` has no tests at all. Correct for now — it has no code — but the
  0-test suite means nothing guards `server/`'s future `common` import path,
  which is the exact thing S5 warns about.
- Nothing automatically catches the stale-path class of defect in S1–S4. A
  `git grep -n 'server/common\|server/migrations\|server/\.env' -- database/`
  returning empty would be a cheap invariant if this kind of move happens
  again; not worth a test at the prototype bar.
- `mypy_path = "../database"` in `training/pyproject.toml:51` puts
  `database/tests` on mypy's search path as a top-level `tests` package
  alongside `training/tests`. I checked with `mypy . -v --no-incremental`:
  training's own `tests.*` modules are all resolved from `./tests/`, so there
  is no shadowing today. Worth knowing it exists if `database/tests` ever grows
  a module name that matters.

## Recommended Decisions

- **S1** — Accept — a runtime error message that names a deleted file will
  cost someone real debugging time; one-word fix.
- **S2** — Accept — `:23` documents which `.env` the migration reads and names
  the wrong one; fix all three references in the same pass.
- **S3** — Accept — this docstring asserts the exact ownership model the
  feature exists to overturn, inside the file that moved.
- **S4** — Accept — the same edit was already made in the sibling file; leaving
  one behind is just an oversight, and it is a one-line change.
- **S5** — Accept — cheap to fix now while `server/common/` is only a sentence
  in a guidelines file, and genuinely confusing to debug once it is a directory.
- **N1** — Accept — bundle with S1, same file, same pass.
- **N2** — Accept — one line, same sweep.
- **N3** — Accept — safe (unapplied revision, docstring only, no DDL or
  revision-id impact) and completes the sweep.
- **N4** — Decline — `HANDOFF.md` is a dated session record, not living docs;
  the spec scoped the doc update to `.claude/coding-guidelines.md`, and
  rewriting history-shaped files invites drift of its own.
- **N5** — Decline — pre-existing and identical across all three packages; the
  plan explicitly required carrying `server/`'s config shape over unchanged,
  and fixing it in one package only would create real drift to fix a cosmetic
  one.
- **N6** — Accept — one line, and it explains a file whose existence is
  otherwise a puzzle.
