# Learnings — data-gen-router-loop / 01-datagen-simulator

## Task 1 — `server/common` DB scaffolding

- **Import path is `common.*`, not `server.common.*`.** `server/pyproject.toml`
  declares `[tool.setuptools.packages.find] include = ["common*", "api*"]`
  with no `server/__init__.py`. That means `server/` itself is the import
  root (like an unprefixed `src/` layout) — code inside `server/` (and later
  `training/`, after `pip install -e ../server`) imports as
  `from common.models import ...` / `from common.db import ...`, never
  `from server.common...`. Get this wrong in a later task and imports will
  resolve at editor-time but fail at runtime once actually installed.
- **pytest package discovery works without extra config.** Because
  `server/tests/__init__.py` and `server/tests/common/__init__.py` exist but
  `server/__init__.py` does not, pytest's rootdir-insertion logic adds
  `server/` (the first ancestor lacking `__init__.py`) to `sys.path`. No
  `conftest.py` or `pythonpath` setting was needed for `from common.models
  import ...` to resolve when running `pytest` from `server/`.
- **`mypy --strict` is happy with SQLAlchemy 2.0's `Mapped`/`mapped_column`
  style out of the box** — no `sqlalchemy[mypy]` plugin or extra `mypy.ini`
  plugin config was needed for the three simple models in this task.
- **Ruff's `UP017` fires on `datetime.now(timezone.utc)`** on this
  target-version/Python combo — it wants `datetime.now(UTC)` via the
  `datetime.UTC` alias (`from datetime import UTC`). Worth remembering for
  any other module in this feature that stamps UTC timestamps.
- Used a Python-side `default=` callable (`_utcnow`) for `created_at`/
  `computed_at` rather than a DB `server_default=func.now()` — keeps
  timestamp generation portable across the in-memory SQLite used in tests
  and the real Postgres instance without relying on either dialect's clock
  function.
- `Label` win/loss values are a proper `enum.Enum` (`LOCAL`/`OFFLOAD`) mapped
  via SQLAlchemy's `Enum` type, not a bare `String`, so a typo'd label value
  is a type error / DB constraint violation rather than silently accepted.

## Task 2 — Alembic migration

- **`alembic init migrations` also drops a `migrations/README` file** that
  wasn't in the task's `Files` list — deleted it after init rather than
  leaving an unrequested extra file in the diff.
- **Autogenerate can build the migration for you without ever touching the
  real Postgres instance.** Wired `migrations/env.py`'s `target_metadata` to
  `common.models.Base.metadata` (imported as `from common.models import
  Base` — per Task 1's `common.*`-not-`server.common.*` import-root
  learning), then *temporarily* pointed `alembic.ini`'s `sqlalchemy.url` at a
  disposable local sqlite file in the scratchpad dir, ran
  `alembic revision --autogenerate`, and restored `alembic.ini`'s real
  placeholder afterward. This diffs the metadata against an empty schema and
  emits accurate `op.create_table` calls (correct types, PK/FK constraints)
  without connecting to — or needing — the project's actual database, and
  without ever running `upgrade head` (only `revision --autogenerate`, which
  never applies anything). Far less error-prone than hand-transcribing every
  column from `models.py`.
- **`sa.Enum(Label)` in the model renders as `sa.Enum('LOCAL', 'OFFLOAD',
  name='label')` in the migration** — SQLAlchemy's plain `Enum(SomeEnum)`
  column type stores/compares on the enum *member name*, not `.value`,
  unless `values_callable` is passed. So the Postgres `label` enum type ends
  up with `'LOCAL'`/`'OFFLOAD'` as its DB-level values, not the lowercase
  `.value` strings (`"local"`/`"offload"`) — this is intentional/consistent
  with the model, not a mismatch to fix.
- **`alembic.ini`'s placeholder `sqlalchemy.url` should already be a
  `postgresql+psycopg://...` URL, not the generic `driver://...` template
  default.** `--sql` (offline) mode only uses the URL to pick the rendering
  dialect, but a non-real dialect scheme falls back to generic/ANSI SQL
  instead of Postgres-specific DDL (e.g. `SERIAL`, proper `CREATE TYPE ...
  AS ENUM`). Kept a `postgresql+psycopg://user:pass@localhost:5432/dbname`
  placeholder plus an `env.py` override that prefers `DATABASE_URL` (mirrors
  `common/db.py`'s convention) whenever it's set, so a real invocation never
  needs to edit `alembic.ini` at all.
- `alembic upgrade head --sql` is the only alembic command run against this
  migration in this task — its log line `Generating static SQL` (vs.
  `Running upgrade` against a live connection) is the tell that no database
  connection was opened; worth checking for that line as a sanity check
  whenever verifying an offline-mode run.

## Task 1b — Redo Task 1's test against real Postgres (corrective)

- **Reusable ephemeral-DB helper lives in `server/common/testing.py`,
  independent of pytest.** `ephemeral_postgres_database(admin_url: str) ->
  Iterator[Engine]` is a plain `@contextmanager`, not a fixture — it doesn't
  import `pytest` at all. `server/tests/conftest.py` wraps it in a
  function-scoped `postgres_engine` fixture; `training/tests/conftest.py`
  (Task 6, later, once it depends on `server/common` via an editable install)
  should do the exact same wrap — `from common.testing import
  ephemeral_postgres_database`, read its own admin URL, `with
  ephemeral_postgres_database(admin_url) as engine: yield engine`. No new
  dependency duplication needed beyond `python-dotenv` in each package's own
  dev deps.
- **`CREATE DATABASE`/`DROP DATABASE` require an autocommit connection** —
  `create_engine(admin_url, isolation_level="AUTOCOMMIT")`. Without it,
  Postgres rejects both statements because they can't run inside a
  transaction block.
- **The admin engine must connect to a maintenance DB (`postgres`), never the
  ephemeral DB itself**, and the ephemeral DB's own engine must be
  `.dispose()`d *before* the `DROP DATABASE` runs — otherwise Postgres
  refuses to drop a database with open connections. Structured as nested
  `try`/`finally`: inner `finally` disposes the test engine and drops the DB;
  outer `finally` disposes the admin engine. Verified teardown actually fires
  on a failing test too (added a throwaway `assert False` test using the
  fixture, ran it, confirmed via `pg_database` that no `test_*` row survived)
  — this is the scenario a bare `yield` fixture without `try`/`finally` would
  get wrong.
- **`server/.env`'s `POSTGRES_ADMIN_URL` is a bare `postgresql://` URL**, which
  makes SQLAlchemy default to the `psycopg2` dialect/driver — not installed,
  since `server/pyproject.toml` depends on `psycopg[binary]` (v3). Added a
  small `_with_psycopg_driver()` normalizer in `testing.py` that rewrites
  `postgresql://` → `postgresql+psycopg://` before `create_engine`, rather
  than requiring `.env` to spell out the driver. Anyone adding a
  `training/.env` admin URL later will likely hit the same thing.
- **This is the real payoff of leaving SQLite:** the original SQLite-backed
  test passed with `session.add_all([run, result, complexity])` even though
  `SimulationResult` has an FK to `SimulationRun` and nothing declares an ORM
  `relationship()` between the two mapped classes — SQLite doesn't enforce FK
  constraints by default (no `PRAGMA foreign_keys=ON`), so the insert order
  never mattered. Postgres enforces the FK immediately and the same
  `add_all` call raised `IntegrityError: ForeignKeyViolation` because the
  unit of work has no relationship info to infer that `simulation_runs` must
  insert before `simulation_results`. Fix (in the test, not the models):
  `session.add(run); session.flush()` before adding `result`/`complexity`.
  Any future test that inserts rows across FK'd tables without a declared
  `relationship()` needs to either flush the parent first or add one in
  dependency order — `add_all` alone won't reorder for you.
- Confirmed no stray `test_*` database survives a normal passing run, a
  failing run, and back-to-back repeated runs — checked via a one-off script
  connecting with the admin URL and querying `pg_database WHERE datname LIKE
  'test_%'`.

## Task 3 — `training/datagen` package + config module

- **`from common.db import get_engine` and `from common.models import ...`
  resolve cleanly from `training/`'s own venv** once `pip install -e
  ../server` runs before `pip install -e ".[dev]"` in that venv — actually
  verified with a one-off `.venv/bin/python -c "from common.db import
  get_engine; from common.models import Base; from common.testing import
  ephemeral_postgres_database"` inside `training/`, not just inferred from
  the dependency line. This is the load-bearing detail Task 6 onward
  depends on, and it works with the plain two-step `pip install` sequence —
  no path hacks, no `conftest.py` sys.path tweaks needed.
- **Plain PEP 508 `dependencies` in `training/pyproject.toml` has no
  portable way to express "install `../server` too."** A relative
  `file://` URL isn't standard, and `${PROJECT_ROOT}` substitution in
  dependency strings is a `uv`-specific `tool.uv.sources` feature, not
  something plain `pip install -e .` understands — using it would silently
  fail to resolve under plain pip. Went with the spec's documented
  alternative instead: `server` is *not* listed in
  `training/pyproject.toml`'s `dependencies` at all; a comment above the
  `dependencies` list in `training/pyproject.toml` spells out the two-step
  install (`pip install -e ../server` then `pip install -e ".[dev]"`).
  Mirrors how `server/`'s own venv was set up (plain `python -m venv` +
  `pip install -e ".[dev]"`, no lockfile), so both packages' install
  stories stay consistent.
- **`training/`, `training/datagen/`, `training/router/`, and
  `training/data/` already existed as empty directories** from the initial
  scaffolding commit (untracked, not yet in git) — `training/data/` is
  already covered by a `training/data/` entry in the root `.gitignore`
  (dataset cache, per Task 4/5's future downloads), and the existing
  generic `.venv/`, `__pycache__/`, `*.egg-info/`, `.mypy_cache/`,
  `.pytest_cache/`, `.ruff_cache/` patterns already cover `training/`'s new
  venv/caches too — no `.gitignore` edit was needed for this task.
- **`config.PRESETS`'s value type is a `TypedDict`
  (`ConditionPresetRanges`)**, not a plain `dict[str, tuple[float,
  float]]`, so every preset is statically checked to define exactly the
  four required axis keys. The one wrinkle: iterating over axis names as
  plain `str` and subscripting the TypedDict with that variable
  (`ranges[axis]`) trips mypy's `literal-required` check (TypedDict
  subscripts normally need a literal key) — the test that does this
  (`test_config.py`) has a targeted `# type: ignore[literal-required]` on
  that one line rather than loosening the TypedDict itself.
- **The Task 3 plan entry's `Files` list doesn't name a `test_config.py`**
  (only the `tests/__init__.py` package markers), but the task's own
  success criteria explicitly requires "a test importing `config`" —
  added `training/tests/datagen/test_config.py` to satisfy that criterion
  directly, since it's clearly implied by the success criteria rather than
  scope creep.
- Illustrative stub-model coefficients (base latency/accuracy, noise
  std-devs, device-load/bandwidth/packet-loss coefficients) are unmeasured
  placeholders per the spec's stub-first stance — Task 9's stub-inference
  formula is what actually gives them meaning; revisit their magnitudes
  then if the resulting latency/accuracy numbers don't look plausible.

## Task 4 — COCO acquisition and local image caching

- **`import pytest` in a test file makes mypy pull in `_pytest`'s numpy
  integration, which trips a pre-existing environment mismatch.**
  `training/pyproject.toml` sets `[tool.mypy] python_version = "3.11"`, but
  `training/.venv` runs Python 3.12, and the installed numpy's bundled
  `__init__.pyi` uses a `type` statement (3.12-only syntax). Any file mypy
  analyzes that imports `pytest` follows into `_pytest._io.saferepr`, which
  imports `numpy` for its array-repr support, and mypy chokes on that stub
  under the 3.11 setting (`error: Type statement is only supported in Python
  3.12 and greater`) — even though the file that triggered it (my
  `test_coco.py`) has nothing to do with numpy. `training/tests/datagen/
  test_config.py` (Task 3) never imports `pytest` directly (it only uses
  plain `def test_...()` functions, relying on pytest's collection, not its
  API), which is why this didn't surface earlier. Fix used here: avoid
  `import pytest` in `test_coco.py` too — replaced the one custom
  `@pytest.fixture` with a plain helper function (`_write_fake_annotations
  (tmp_path) -> Path`) called directly from each test; `tmp_path` itself is
  injected by pytest without needing an import. This is a workaround, not a
  real fix — `pyproject.toml`'s mypy `python_version` (or the venv's
  interpreter version) is out of this task's `Files` list, so didn't touch
  it. Any future test file in `training/` that needs `@pytest.fixture`,
  `pytest.raises`, `pytest.mark.parametrize`, etc. will hit the same wall
  and needs either the same plain-function workaround or (better, but out
  of scope here) reconciling `python_version` with the venv's actual
  Python. Confirmed via `git stash -u` that `mypy .` was clean on the 5
  pre-existing files before this task's files were added, and that adding
  `test_coco.py` with a plain `import pytest` (no fixture even) alone was
  enough to reproduce the failure — it's the import, not fixture usage,
  that triggers it.
- **`json.load()`'s return type is `Any`, and that's fine to let flow into a
  `NamedTuple` constructor under `mypy --strict`.** Originally over-typed
  the loaded `images` list as `list[dict[str, object]]`, which then made
  `entry["id"]` resolve to `object` and broke `int(entry["id"])` (`object`
  isn't one of `int()`'s overloads). Fixed by typing the whole `json.load()`
  result as `dict[str, list[Any]]` and passing `entry["id"]`/
  `entry["file_name"]` (both `Any`) straight into `ImageRecord(image_id=...,
  file_name=...)` — mypy's `warn_return_any` (part of `--strict`) only
  fires when a function's return expression itself is `Any`-typed, not when
  `Any` values are used to construct a statically-typed object one level in.
- **Verified against the real dataset (read-only, no network) that
  `training/data/coco/annotations/instances_val2017.json`'s `images` array
  has exactly 5,000 entries**, and that `resolve_image_path()` on a real
  file name resolves instantly via the pre-downloaded-file branch (no
  `fetch` call) — confirms the "common case" path works against real data,
  not just the fake fixtures in `test_coco.py`.
- `COCO_DIR`/`ANNOTATIONS_PATH`/`IMAGES_DIR` are computed from `Path(__file__
  ).resolve().parent.parent`, i.e. relative to `coco.py`'s own location
  (`training/datagen/coco.py` → `training/data/coco/`), not relative to the
  process's current working directory — so `load_image_index()`/
  `resolve_image_path()` use correct real-data defaults regardless of where
  a future caller (e.g. Task 6's simulator script) is invoked from.

## Task 5 — Scene-complexity proxy

- **This venv's installed `opencv-python-headless` (v4.9+ per pyproject, but
  actually resolved to 5.0.0) ships its own bundled `.pyi` type stubs**
  (`cv2/__init__.pyi`, `cv2/typing/__init__.py`, etc.) — the `[[tool.mypy
  .overrides]] module = "cv2.*" ignore_missing_imports = true` entry already
  in `training/pyproject.toml` (Task 3) turns out to be a no-op safety net
  for this version, not the thing making mypy pass. Typing a function
  parameter/return as `numpy.typing.NDArray[np.uint8]` for values that flow
  through `cv2.imread`/`cv2.cvtColor`/`cv2.Canny` fails under `mypy
  --strict`, because those functions' real stubs return `cv2.typing
  .MatLike` (a `Union[cv2.mat_wrapper.Mat, NumPyArrayNumeric]` type alias,
  where `NumPyArrayNumeric` is `ndarray[Any, dtype[integer[Any] |
  floating[Any]]]`), which mypy does not consider assignable to
  `NDArray[np.uint8]` — this produced `[return-value]` errors *and* made a
  paired `# type: ignore[no-any-return]` register as `[unused-ignore]`
  (because the mismatch mypy actually flags is `return-value`, not
  `no-any-return`), i.e. two different mypy complaints stacked on the same
  line. Fix: type these values as `cv2.typing.MatLike` throughout
  `complexity.py` (import via `from cv2.typing import MatLike`) instead of
  reaching for `NDArray[np.uint8]` — no `type: ignore` needed anywhere.
  Anything else in `training/` that pipes an array through more than one
  cv2 call should default to `MatLike` for cv2-facing signatures rather
  than a `numpy.typing` alias, and only convert to a stricter numpy dtype
  at the boundary once cv2 is out of the picture.
- Chose Canny thresholds 100/200 — OpenCV's own commonly-cited
  "reasonable default" pair for 8-bit images from its Canny tutorial, not a
  value tuned against this project's data. This proxy only needs relative
  ranking (complex scenes score higher than blank ones) for the
  stratified-sampling and stub-accuracy uses described in the spec, not a
  calibrated absolute edge count, so an off-the-shelf default was judged
  sufficient without a tuning pass.
- **Task 4's `import pytest`-avoidance workaround is no longer needed** now
  that `pyproject.toml`'s `[tool.mypy] python_version` was corrected from
  `"3.11"` to `"3.12"` immediately before this task. Verified directly:
  temporarily added `import pytest` to `test_complexity.py` and reran
  `mypy .` — clean, no `_pytest`/numpy stub error at all (confirming the
  version-string mismatch really was the root cause Task 4 suspected, not
  something deeper). Reverted the import since this file doesn't actually
  need any pytest API (no fixtures, no `pytest.raises`/`parametrize`), so
  it stays on plain `def test_...()` functions either way — but any future
  `training/tests/` file that *does* need `@pytest.fixture` or similar can
  now `import pytest` directly without hitting Task 4's wall. Worth a
  one-line update to Task 4's own note if anyone revisits it.
- Manually eyeballed the proxy against 5 real `training/data/coco/val2017/`
  images (not part of the automated suite, per the task's own guidance):
  scores ranged ~0.06–0.23, i.e. a plausible spread rather than everything
  clustering at one extreme — no hardcoded-expectation test was written
  against these, since real-photo complexity values will vary run to run
  if the sample images ever change.

## Task 6 — Persistence layer (all three tables)

- **`training/tests/conftest.py` is a near-verbatim copy of
  `server/tests/conftest.py`**, per Task 1b's note — same `postgres_engine`
  fixture, same `load_dotenv(Path(__file__).resolve().parent.parent /
  ".env")` pattern (resolving to `training/.env`, which already had
  `POSTGRES_ADMIN_URL` from Task 1b/bootstrap), same function-scoped
  create/drop-per-test shape. No new pattern needed, just the same helper
  wrapped a second time as documented.
- **New, load-bearing gap this task exposed: mypy cannot resolve
  `common.*` from `training/` at all, even though the runtime import works
  fine.** Tasks 1–5 never imported `common` from a `training/` file, so this
  never surfaced before. The root cause: `server`'s editable install
  (`pip install -e ../server`) registers a `MetaPathFinder`
  (`__editable___server_0_1_0_finder.py`) that maps `common` →
  `/…/server/common` *at import time* via `sys.meta_path` — this makes
  `from common.models import Base` work perfectly at runtime, but mypy does
  purely static, filesystem-based module resolution and never executes that
  import hook, so it reports `Cannot find implementation or library stub for
  module named "common.models"` ([import-not-found]) on every file that
  imports `common`. Fix: added one `mypy_path = "../server"` line under
  `training/pyproject.toml`'s existing `[tool.mypy]` block (verified first
  with a throwaway `MYPYPATH=../server mypy .` before touching the file) —
  `server/` is `common`'s own import root (Task 1's finding), so pointing
  mypy directly at it via `mypy_path` lets it find `server/common/*.py` as a
  regular package (it has `__init__.py`) without needing
  `explicit_package_bases` or any namespace-package config. This is a
  build-tooling config addition, not a functional change, but it's a file
  outside Task 6's own `Files` list (`training/datagen/persistence.py`,
  `training/tests/datagen/test_persistence.py`) — flagged explicitly in this
  task's report rather than silently expanded. Confirmed the change is
  additive-only: `cd server && ruff check . && mypy .` and `pytest -q`
  still pass clean afterward, and it only affects `training/`'s own mypy
  invocation.
- **`create_run`/`store_results` accept a small `RunConfig`/`ResultRow`
  dataclass pair (defined in `persistence.py` itself) rather than a bare
  `dict[str, Any]`**, to keep mypy `--strict` meaningful at the call site
  (a typo'd or missing config field is a type error, not a silent `KeyError`
  at insert time) — mirrors the existing `ConditionPresetRanges` TypedDict
  pattern in `config.py` (typed shape over a raw dict) but as a dataclass
  since these values are constructed programmatically per-invocation/per-row
  rather than declared as static literals.
- **De-duplication in `store_complexity_scores` is done with one extra
  `SELECT file_name WHERE dataset = :dataset` before the insert, not an
  `INSERT ... ON CONFLICT DO NOTHING`** — simpler, portable across the
  ephemeral-Postgres test engine and whatever engine `common.db.get_engine()`
  builds for a real run, and the composite-PK upsert syntax isn't needed
  when a plain read-then-filter is cheap enough at this data volume (COCO
  val2017 is 5,000 images per dataset, scored once ever).
- **Every function opens and closes its own `Session` internally
  (`with Session(engine) as session: ...`)** rather than accepting a
  pre-opened session — matches the task's "each function takes an engine"
  requirement literally and keeps each function's transaction boundary
  self-contained (one function call = one commit), which matters once a
  future simulator script calls `create_run` once and `store_results` many
  times in a batch loop without needing to manage session lifetime itself.
- Confirmed the FK-insert-ordering gotcha from Task 1b applies identically
  here even though it's hidden behind the `persistence` functions:
  `create_run` commits (and returns) the `run_id` before `store_results` is
  ever called, so the two functions' natural call order already guarantees
  parent-before-child — no explicit `session.flush()` needed in
  `persistence.py` itself (unlike the raw `test_models.py` example, which
  inserts both in one session/transaction).
- Verified via a one-off script against the admin URL that no `test_%`
  database survived after this task's full test run (4 new tests plus the
  pre-existing 11), matching Task 1b's verified-teardown pattern.

## Task 7 — Stratified frame sampling

- **`numpy.array_split` on a plain Python list of strings needs an explicit
  `dtype=object` array first**, not `np.array(list_of_str)` — without it
  numpy infers a fixed-width unicode dtype (`'<U...'`) sized to the longest
  input string, which is harmless here (values round-trip fine through
  `str(...)` on the way out) but is a common surprise; wrapping with
  `dtype=object` keeps the array holding actual Python `str` objects instead
  of fixed-width numpy string cells.
- **Bucketing is done by sorting `(file_name, score)` pairs and splitting the
  file-name list, not by calling `numpy.quantile` on the scores.**
  `numpy.array_split` on the score-sorted pool already produces exactly the
  "`bucket_count` equal-frequency buckets" the task asks for (each bucket
  gets `len(pool) // bucket_count` items, with any remainder going to the
  front buckets) without needing to compute quantile cutpoints and then
  re-filter items into ranges — simpler and avoids edge cases where multiple
  items share an identical score straddling a computed cutpoint.
- **Remainder distribution (`target_count` not divisible by `bucket_count`)
  uses `divmod(target_count, bucket_count)` and gives one extra to each of
  the first `remainder` buckets** (lowest-complexity buckets first, since
  buckets are built off the ascending-sorted pool) — an arbitrary but
  deterministic tie-break, consistent with the determinism requirement.
- **The "bucket smaller than its target share" defensive case is exercised
  by a real test** (`test_handles_bucket_with_fewer_items_than_target_share`,
  a 10-item pool with `target_count=500`) — `min(share, len(bucket))` before
  calling `rng.choice(..., replace=False)` avoids numpy's `ValueError:
  Cannot take a larger sample than population when replace=False`, and the
  function just returns fewer than `target_count` items rather than padding
  or raising, matching the task's "don't crash" requirement without
  inventing a redistribution scheme the spec never asked for.
- No new import-path or mypy/ruff wrinkles beyond what Tasks 3–6 already
  resolved (`mypy_path`, Python 3.12 setting, plain `def test_...()` style
  still works but `import pytest` would too per Task 5's fix — this file
  didn't need any pytest API so stayed with plain functions). Full suite
  (`pytest -q` from `training/`) is 21 passed (15 pre-existing + 6 new),
  confirming no regressions in Task 6's Postgres-backed persistence tests.

## Task 8 — Condition-vector sampling

- **`scipy.stats.qmc.LatinHypercube(d=4, seed=seed).random(n=count)`** draws
  the raw `[0, 1)^4` samples; `qmc.scale(unit_samples, lower_bounds,
  upper_bounds)` (both plain per-axis lists, in the same axis order as the
  columns) does the min/max rescale in one call instead of a hand-rolled
  `lower + value * (upper - lower)` loop — scipy already ships the exact
  "scale a QMC unit-cube sample into per-dimension bounds" helper this task
  needed, so no manual broadcasting/reshaping code was required.
- **`ConditionPresetRanges` subscripted with a loop variable (`preset[axis]`)
  trips mypy's `literal-required` check**, same as Task 3's
  `test_config.py` finding — this module (not just its test) subscripts the
  TypedDict with the `_AXES` tuple's loop variable to pull `(min, max)` per
  axis, so both `conditions.py` itself and `test_conditions.py` carry a
  `# type: ignore[literal-required]` on that line. Confirms Task 3's note
  applies to non-test code too, not just tests.
- **Coverage-tolerance choice for the "spans close to the full range" test:
  10% of each axis's range width**, justified by LHS's own stratification
  guarantee rather than picked arbitrarily — with `count=50` samples, LHS
  partitions each axis into 50 equal-width strata and draws exactly one
  sample per stratum, so the extreme strata are each `1/50 = 2%` of the
  range wide by construction. A 10% tolerance is a 5x margin above that
  guaranteed 2%, generous enough to never flake, while still tight enough to
  fail loudly if the scaling logic were broken (e.g. an axis left
  unscaled in `[0, 1)`, or scaled against the wrong axis's bounds after a
  reordering mistake in `_AXES`).
- Verified the sanity/bounds test (`test_every_axis_value_falls_within_
  configured_bounds`) against all four presets in `config.PRESETS`, not just
  `baseline` — cheap to loop over every preset since the function takes the
  preset dict directly, and it catches a scaling bug that happened to only
  manifest on a narrow-range preset (e.g. `device-stress`'s `(60.0, 100.0)`
  device-load range) that a `baseline`-only test could miss.
- Full suite (`pytest -q` from `training/`) is 26 passed (21 pre-existing +
  5 new); `ruff check .` and `mypy .` both clean with no new overrides
  needed beyond the existing `scipy.*` `ignore_missing_imports` entry from
  Task 3.

## Task 9 — Stub inference model

- **Added two new small config constants** (flagged, not in Task 9's own
  `Files` list, same justification pattern as prior tasks' small necessary
  config additions): `LOCAL_ACCURACY_DEVICE_LOAD_PENALTY_COEFFICIENT = 0.001`
  (per device-load pct point) and
  `OFFLOAD_ACCURACY_PACKET_LOSS_PENALTY_COEFFICIENT = 0.005` (per
  packet-loss pct point) in `training/datagen/config.py`. Story: a
  heavily-loaded device is modeled as falling back to a lighter/faster
  on-device model under load (hence a device-load accuracy penalty on the
  *local* path specifically, distinct from the shared
  `SCENE_COMPLEXITY_ACCURACY_PENALTY_COEFFICIENT`), and a lossy link is
  modeled as losing detail to dropped/retransmitted frames (hence a
  packet-loss accuracy penalty on the *offload* path specifically). At the
  extremes of the `baseline`/`device-stress` presets (device_load=100) and
  `network-stress` preset (packet_loss=20), these contribute a max ~0.1 and
  ~0.1 accuracy-point penalty respectively — comparable in magnitude to the
  existing scene-complexity penalty's typical contribution, not dominant.
- **One shared `numpy.random.default_rng(seed)` per call, with 4 fixed-order
  `rng.normal(...)` draws** (local-latency noise, offload-latency noise,
  local-accuracy noise, offload-accuracy noise, in that order) is sufficient
  for both the determinism and vary-by-seed requirements — no need for 4
  independently-seeded RNGs or a hash of `(condition, scene_complexity,
  seed)` into a derived seed. `stub_inference` is a per-(frame, condition)
  pure function; Task 10 (or whatever composes it into rows) is responsible
  for feeding a distinct `seed` per row if per-row-varying noise is wanted —
  out of scope for this task, whose only contract is the 3-input signature
  given in the task/spec.
- **Monotonicity tests average over 200 seeds per condition rather than
  comparing a single fixed-seed pair**, because the noise stds
  (`LOCAL_LATENCY_NOISE_STD_MS=5.0`, `OFFLOAD_LATENCY_NOISE_STD_MS=8.0`,
  `*_ACCURACY_NOISE_STD=0.03`) are large enough relative to some of the
  deterministic per-step deltas (e.g. one device-load pct point only moves
  local latency by 1.2ms, well within noise) that a single arbitrary seed
  could occasionally flip the comparison. Averaging over many seeds drives
  the noise term's contribution to the mean toward ~0 (it's zero-mean
  Gaussian) while preserving the deterministic delta, making the assertion
  reliable without weakening it to a non-strict inequality.
- Confirmed accuracy stays within `[0, 1]` via `numpy.clip` even under
  inputs beyond any preset's configured range (packet_loss_pct=100, an
  input `sample_condition_vectors` would never actually produce given the
  current presets' max of 20.0) and extreme scene_complexity values (0.0,
  1.0, 10.0) — the clip is a hard safety net independent of whatever range
  the real proxy/presets produce.
- Full suite (`pytest -q` from `training/`) is 32 passed (26 pre-existing +
  6 new); `ruff check .` and `mypy .` both clean, no new overrides needed.

## Task 10 — Win/loss labeling

- **Returns `common.models.Label` (the enum), not a bare `"local"`/
  `"offload"` string**, despite the task text's literal wording — this is
  what `persistence.ResultRow.label` and `SimulationResult.label` actually
  expect (Task 6), and it's what the orchestrator's own task brief called
  out as the correct choice. No disagreement found; went with it directly.
- **Tie-break rule: an exact utility tie resolves to `Label.LOCAL`.**
  Arbitrary but deterministic, consistent with Task 7's remainder-
  distribution tie-break precedent in this feature. Rationale documented in
  the function's docstring: local has no network dependency, so it's the
  "safer" default when the two paths are truly indistinguishable on the
  utility measure.
- **Hand-computed test values need a float-tolerance helper (`_isclose`,
  1e-9), not bare `==`, even for numbers that look like they should be
  exact.** `0.70 - 1.0 * (50.0 / 1000.0)` evaluates to
  `0.6499999999999999`, not `0.65`, in IEEE-754 double precision — caught by
  running the test, not by inspection. Some hand-picked combinations (e.g.
  `500.0 / 1000.0 = 0.5` exactly) do round-trip exactly and don't strictly
  need the helper, but using it uniformly avoids having to reason about
  which specific literals happen to be exact binary fractions.
- **Picked `lambda_value=1.0` for the clear-win test cases instead of
  `config.DEFAULT_LAMBDA` (0.005).** At the default lambda, latency is
  divided by 1000 and then scaled by 0.005, so even a 2000ms latency gap
  only moves utility by ~0.01 — swamped by any accuracy difference of a few
  points, making it hard to hand-construct a case that's obviously a "local
  should clearly win on latency" scenario without also making accuracy
  degenerate. A larger lambda (1.0) in the test makes latency and accuracy
  contribute comparable magnitudes to utility, which is what makes the
  clear-win/clear-loss cases legible; `compute_label` itself still takes
  lambda as a plain parameter, so this is purely a test-construction choice,
  not a change to the function's behavior or the config default. A separate
  test explicitly re-runs the same latency/accuracy inputs at two different
  lambda values (0.001 vs 2.0) and asserts the label flips, to confirm
  lambda is actually threaded through the computation rather than ignored.
- No new mypy/ruff wrinkles — this module only imports `common.models.Label`
  and does arithmetic, no new stub/override needed beyond what Tasks 1–9
  already resolved. Full suite (`pytest -q` from `training/`) is 37 passed
  (32 pre-existing + 5 new).

## Task 11 — Orchestration script + end-to-end integration test

- **The real invocation is `python -m datagen.run_simulation --preset
  <name>`, not `python -m training.datagen.run_simulation ...`** as the plan
  text literally says. `training/` has no `training/__init__.py` (same as
  `server/` for `common`, per Task 1's finding) — `training/pyproject.toml`'s
  `[tool.setuptools.packages.find] include = ["datagen*", "router*"]` makes
  `datagen` itself the top-level importable package once installed, and every
  module in this feature already imports its siblings as `from datagen.config
  import ...` (never `from training.datagen...`). Documented the correct
  invocation in `run_simulation.py`'s own module docstring and flagging it
  here since the task text's example was stale on this point too, same as
  the SQLite→Postgres correction already called out in the task brief.
- **`stub_inference`'s own contract (Task 9) only guarantees determinism for
  a *fixed* seed** — Task 9 explicitly left "feed a distinct seed per row"
  as this task's responsibility. Reusing one run-level seed for every
  (frame, condition) row would have made every row draw bit-identical noise,
  which is wrong (each row needs independent-looking noise) while still
  needing to be reproducible run-to-run. Fix: `row_seed = resolved_seed +
  frame_index * len(condition_vectors) + condition_index`, using the
  enumerate-index position in the (already-deterministic, seed-derived)
  sampled-frames/condition-vectors lists — cheap, collision-free within one
  run (`frame_count * condition_vector_count` distinct offsets), and
  reproducible across two runs of the same seed/preset/pool since both the
  sampling and the offsets are pure functions of the same inputs.
- **`ConditionPresetRanges.items()` types values as `object` under mypy**,
  not `tuple[float, float]` — same root cause as Task 3/8's
  `literal-required` TypedDict finding (mypy doesn't assume TypedDict values
  are homogeneous even when every field in this particular TypedDict happens
  to share a type), but this time it broke `list(bounds)` with `No overload
  variant of "list" matches argument type "object"` rather than a
  `literal-required` warning. The `# type: ignore[literal-required]`
  precedent doesn't apply here since the error is `call-overload`, not
  `literal-required`. Fix: build `condition_ranges` (the `RunConfig.
  condition_ranges` snapshot) from the 4 known literal keys directly
  (`preset["bandwidth_mbps"]`, etc.) instead of iterating `.items()` — no
  `type: ignore` needed anywhere. Same fix applied in the test file, which
  independently needed to reconstruct the expected `condition_ranges` dict
  for its `SimulationRun` assertion.
- **`RunConfig.frame_count`/`condition_vector_count` are persisted as the
  *actual* sampled lengths (`len(sampled_frames)`/`len(condition_vectors)`),
  not the requested target counts** — normally identical (the real
  ~5,000-image COCO pool and default `FRAME_COUNT=500`/
  `CONDITION_VECTOR_COUNT=50` never hit Task 7's "bucket smaller than its
  target share" edge case), but persisting the actual counts makes the
  `simulation_runs` record always truthfully match the number of
  `simulation_results` rows actually written for it, even in that edge case,
  which is what the spec's "traceable back to exactly what configuration
  produced it" requirement is really asking for.
- **The core function's dependency-injection seams are exactly**
  `image_records: list[ImageRecord] | None` and `resolve_image:
  Callable[[str], Path] | None`, both defaulting to `None` and resolved
  *inside* the function body (`coco.load_image_index()` /
  `coco.resolve_image_path`) rather than as literal parameter defaults —
  this avoids reading the real (possibly-absent, in a fresh checkout without
  the pre-downloaded dataset) annotations file at import time or whenever a
  test imports `run_simulation` for any other reason. `frame_count`/
  `condition_vector_count`/`bucket_count`/`seed`/`lambda_value` follow the
  same `| None = None`-then-resolve-to-`config.*` pattern, letting the test
  run a fast 10-frame x 5-condition-vector (50-row) pipeline against a
  20-image fake pool instead of the real 500 x 50 defaults.
- **Test's fake pool uses `cv2.imwrite` on numpy arrays blending a flat
  gray background with random noise at an index-controlled density (`i /
  (POOL_SIZE - 1)` for `i in range(20)`)** rather than pure random noise for
  every image — this spreads the 20 images' edge-density complexity scores
  out deliberately (rather than leaving them to cluster near whatever score
  pure noise happens to produce), which both matters for exercising
  `stratified_sample`'s bucketing meaningfully and all but eliminates the
  already-small risk of two images tying on complexity score (a tie could
  flip `stratified_sample`'s stable-sort bucket assignment between the two
  test-pipeline runs if the tied images' relative dict-iteration order ever
  differed, e.g. because the second run's scores come back from a DB
  `SELECT` with no guaranteed row order — not actually observed, but worth
  noting for anyone who simplifies the fake-pool generator to pure noise
  later).
- **Added a lightweight resolve-call counter (`call_count["n"] += 1` inside
  the injected `resolve_image` closure) instead of skipping instrumentation
  entirely** — cheap to add and it directly proves the "second run does not
  recompute" criterion at the resolve/complexity-computation level (0 calls
  on the second run), not just indirectly via an unchanged `scene_complexity`
  row count, which alone wouldn't distinguish "recomputed but deduped at
  insert time" from "never recomputed at all."
- Full suite (`pytest -q` from `training/`) is 39 passed (37 pre-existing + 2
  new); `ruff check .` and `mypy .` both clean. Confirmed via a one-off
  script against `POSTGRES_ADMIN_URL` that no `test_%` database survives
  after this task's test run. This task's two tests only touch
  `training/datagen/run_simulation.py` and
  `training/tests/datagen/test_run_simulation.py` — no other files needed
  changes.

## Review finding B1 (fix) — `DEFAULT_LAMBDA` scaled twice

- **Root cause confirmed exactly as the finding described:** `config.py`'s
  comment for `DEFAULT_LAMBDA = 0.005` documented applying λ directly to a
  millisecond-valued latency, but `labeling._utility` divides by 1000 first
  (`accuracy - lambda_value * (latency_ms / 1000)`), so the effective per-ms
  weight was `0.005 / 1000 = 5e-6` — a 2000ms latency gap only moved utility
  by 0.01, swamped by any realistic accuracy gap (~0.07-0.1 between the
  local/offload stub paths' base accuracies). Task 10's own learnings entry
  had already flagged this exact tension in passing ("even a 2000ms latency
  gap only moves utility by ~0.01") when explaining why its hand-picked test
  cases used `lambda_value=1.0` instead of the config default — worth
  cross-referencing next time a "the test uses a different value than the
  default, is that suspicious?" question comes up; sometimes it's flagging a
  real default-value bug rather than just a test-legibility choice.
- **Fix: `DEFAULT_LAMBDA = 0.3`, comment corrected to describe the per-second
  convention** (`labeling.py`'s formula is unchanged, matches the spec, and
  was not touched). At 0.3, a 300ms latency gap moves utility by 0.09,
  comparable to the ~0.07-0.1 realistic accuracy gap between the two stub
  paths — neither term structurally dominates anymore.
- **Regression test constructs two cases sharing the same accuracy inputs
  (`LOCAL_BASE_ACCURACY=0.78` vs `OFFLOAD_BASE_ACCURACY=0.85`, a realistic
  0.07 gap) and varying only the latency gap**, using `config.DEFAULT_LAMBDA`
  directly (not a hand-picked test-only λ, per the finding's explicit ask):
  a large gap (100ms vs 460ms) flips the label to `LOCAL` despite offload's
  higher accuracy; a small gap (100ms vs 150ms) leaves the label at
  `OFFLOAD` on the same accuracy inputs. This isolates the latency term's
  effect and would have caught the original bug (at the old λ=0.005, the
  large-gap case's latency term only moves utility by ~0.0018, nowhere near
  enough to overcome the 0.07 accuracy gap, so the test would have failed
  with `Label.OFFLOAD` instead of the asserted `Label.LOCAL`).
- **Sanity-checked the label distribution's responsiveness the same way the
  reviewer did** (4,000 condition vectors per preset, `stub_inference` +
  `compute_label` under the new default λ, no persistence/DB involved — a
  throwaway script, not a committed test): `baseline` moved from ~96%
  OFFLOAD to 71.8%/28.1% OFFLOAD/LOCAL; `network-stress` moved from
  presumably-near-uniform to 33.6%/66.4%; `device-stress` to 92.1%/7.9%;
  `degraded-network-idle-device` to 6.8%/93.2%. Also confirmed the split
  responds to condition *severity* within a preset, not just overall rate:
  on `network-stress`, restricting to bandwidth < 1 Mbps gives a 25.2%
  OFFLOAD rate vs. 38.1% for bandwidth > 4 Mbps (same preset, same 4,000
  samples) — the label now tracks the network condition instead of being a
  near-constant majority-class regardless of severity.
- No other files needed changes — `labeling.py`'s formula, the other config
  constants, and the rest of the test suite (`test_stub_inference.py`,
  `test_run_simulation.py`, both flagged by the finding as
  label/utility-adjacent) were unaffected and still pass; full suite is 40
  passed (39 pre-existing + 1 new) after this fix.

## Review finding B2 (fix) — non-deterministic sort on tied complexity scores

- **Root cause confirmed exactly as the finding described:**
  `sorted(scores.items(), key=lambda item: item[1])` in
  `sampling.stratified_sample` sorts on score only; Python's sort is stable,
  so tied frames keep whatever order they arrived in via `scores`' dict
  iteration, and that order isn't guaranteed to match between two runs (an
  annotation-file-order mapping the first time a dataset is scored vs. an
  unordered Postgres `SELECT` on `get_known_complexity` every run after).
  Task 11's own learnings had already flagged this exact risk in passing
  (in the note about the fake-pool generator avoiding ties on purpose) —
  worth remembering that a "we deliberately avoided X in the test fixture"
  note is sometimes flagging a latent bug in the code under test, not just a
  test-design choice.
- **Fix: sort key is `(item[1], item[0])`** — total order over
  `(score, file_name)`, so ties break on file name (a stable, order-
  independent value) instead of falling through to insertion order.
  `get_known_complexity` also gained `.order_by(SceneComplexity.file_name)`
  as a second line of defence — not load-bearing for this specific bug once
  the sort key is total (any insertion order now produces the same sorted
  result), but it keeps the function's own output deterministic for any
  other future caller that might rely on dict order without going through
  `stratified_sample`'s tie-breaking.
- **Reproduced the bug directly before fixing it**, per the task's own
  verification requirement: temporarily reverted the sort-key change,
  re-ran the new regression test
  (`test_tied_scores_produce_identical_sample_regardless_of_insertion_order`
  in `test_sampling.py`), confirmed it failed with a diff at index 26
  (`'frame_0100.jpg' != 'frame_0101.jpg'`) — the same two-frames-swap
  symptom the finding described (`img_042.jpg` vs `img_041.jpg`) — then
  restored the fix and confirmed the test (and the full suite) passes.
- **Test construction: reuse `_synthetic_scores()`'s existing 200-item pool
  and add one deliberately-tied entry** (`frame_0101.jpg` given the same
  score as `frame_0100.jpg`) rather than building a whole new fixture —
  keeps the test close to the existing suite's style and guarantees the tied
  pair is real (not one that numpy's `array_split` bucket boundaries route
  away from any `rng.choice` draw), verified by confirming the unfixed code
  actually produces a different sample when insertion order is reversed
  before writing the assertion the "right" way.
- No `test_persistence.py` addition was made for the `get_known_complexity`
  ordering guarantee — the existing round-trip tests already compare against
  a `dict` (which is order-insensitive for equality), so an ordering-specific
  assertion would need to inspect raw row order via a lower-level query
  rather than the function's own `dict`-returning contract; judged not worth
  the added complexity for a defence-in-depth fix that isn't itself
  load-bearing for the reproducibility bug once `sampling.py`'s sort key is
  total.
- Full suite (`pytest -v` from `training/`) is 41 passed (40 pre-existing +
  1 new); `ruff check .` and `mypy .` both clean, no new overrides needed.
  Confirmed via a one-off script against `POSTGRES_ADMIN_URL` that no
  `test_%` database survived after this fix's test run.

## Review finding S1 (fix) — `dataset` never persisted on `simulation_runs`

- **Pure additive column, no autogenerate regeneration needed.** Unlike
  Task 2's original migration authoring, this fix just hand-added one
  `sa.Column("dataset", sa.String(), nullable=False)` line to the existing
  `op.create_table("simulation_runs", ...)` call (positioned right after
  `run_id`, matching `models.py`'s column order) rather than regenerating the
  whole migration from scratch — simpler for a single-column addition to an
  unapplied migration, and still verified against drift (see below).
- **Verified the edited migration matches `models.py` exactly using `alembic
  check`, not just eyeballing the diff.** Temporarily pointed
  `alembic.ini`'s `sqlalchemy.url` at a disposable sqlite file in the
  scratchpad dir (same swap-and-restore pattern as Task 2's `learnings.md`
  entry), ran `alembic upgrade head` to apply the edited migration to that
  throwaway file, then `alembic check` — which reported "No new upgrade
  operations detected," i.e. autogenerate sees zero drift between the
  migration's resulting schema and current `Base.metadata`. This is a
  stronger check than `--sql` rendering alone (which only proves the
  migration is syntactically valid Postgres DDL, not that it matches the
  models). Restored `alembic.ini` immediately after and deleted the
  scratchpad sqlite file.
- **`RunConfig.dataset` and `SimulationRun.dataset` are populated from
  `config.DATASET_NAME` at the one call site (`run_simulation.py`'s
  `RunConfig(...)` construction)** — no other file needed to change to
  thread the value through, since `create_run` already just forwards every
  `RunConfig` field onto the `SimulationRun` constructor.
- **Every existing test that constructs a `RunConfig` or a bare
  `SimulationRun` needed a `dataset=` argument added** (`server/tests/common
  /test_models.py`, `training/tests/datagen/test_persistence.py`), plus one
  new assertion on `run.dataset`/`fetched_run.dataset` in each of those two
  files and in `test_run_simulation.py` (which builds its `RunConfig`
  indirectly through `run_simulation()`, so only needed the new assertion,
  not a constructor change). No test file needed a structural change beyond
  that — this finding is a pure schema/plumbing addition, not new logic.
- Full suite: `server` 1 passed (unchanged count, single model round-trip
  test), `training` 41 passed (unchanged count from the prior fix — no new
  test functions added, only new fields/assertions on existing ones); both
  packages' `ruff check .` and `mypy .` clean. Confirmed via the admin-URL
  `pg_database` query that no stray `test_%` database survived.

## Review finding S3 (fix) — non-atomic image write can cache a truncated file forever

- **Fix is a 3-line change in `resolve_image_path`:** write the fetched bytes
  to `local_path.with_suffix(local_path.suffix + ".part")` first, then
  `part_path.replace(local_path)`. `Path.replace()` is an atomic rename on
  POSIX (single `rename(2)` syscall) — either the whole `.part` file lands at
  `local_path` or nothing does; there's no window where a half-written file
  sits at the path `resolve_image_path`'s own `local_path.exists()` cache
  check treats as a hit. No change needed to that cache-check line itself —
  the fix only changes how the file gets there.
- **Reproduced the bug before trusting the fix, per this feature's own
  established pattern (B1/B2's learnings entries did the same):**
  `git stash push -- datagen/coco.py` to temporarily restore the old
  `local_path.write_bytes(image_bytes)` code, reran the new failure-mode test
  (`test_resolve_image_path_interrupted_write_does_not_cache_truncated_file`)
  against it, confirmed it failed with `AssertionError: assert not True` on
  `final_path.exists()` — i.e. the old code did leave a file at the final
  path even though the write was "interrupted." Then `git stash pop` to
  restore the fix and confirmed the same test passes. This is the direct
  proof the finding asked for: without the fix, an interrupted write reaches
  the final path; with it, it doesn't.
- **Simulating "interrupted" required patching `Path.replace`, not
  `Path.write_bytes` or the injected `fetch` callable.** The realistic
  failure mode this finding describes (process killed mid-`write_bytes`, or
  a dropped connection mid-download) can't be cleanly injected through this
  module's existing `FetchFn` seam, since `fetch` already returns before any
  file I/O happens — by the time bytes reach `resolve_image_path`, the
  "network" part is done. The fix's own atomic-write step gives a new,
  precise injection point instead: `mock.patch.object(Path, "replace",
  ...)` raising only when called on the specific `.part` path under test
  (falling through to the real `Path.replace` for any other call, e.g. ones
  made by pytest's own `tmp_path` machinery) simulates "the write completed
  but the rename never happened" — which is exactly the boundary the fix is
  meant to make safe, and is deterministic (no reliance on timing or actual
  process interruption).
- **The `_crash_before_replace` wrapper only intercepts calls where `self ==
  part_path`**, not a blanket monkeypatch of `Path.replace` entirely — a
  blanket patch would risk breaking unrelated `Path.replace` calls elsewhere
  in the same test (there are none here, but the narrow guard costs nothing
  and matches this feature's general preference for the smallest fake that
  makes the point, e.g. Task 11's scoped `resolve_image` closure).
- Added a companion "happy path still works and leaves no `.part` litter"
  regression test
  (`test_resolve_image_path_download_leaves_no_leftover_part_file`) rather
  than relying solely on the pre-existing
  `test_resolve_image_path_downloads_and_caches_missing_file` test (Task 4)
  to cover the successful case — the new test explicitly asserts the `.part`
  sibling doesn't linger after a clean run, which the pre-existing test never
  checked and the fix newly makes relevant.
- Full suite (`pytest -v` from `training/`) is 44 passed (42 pre-existing + 2
  new); `ruff check .` and `mypy .` both clean, no new overrides needed. This
  fix only touches `training/datagen/coco.py` and
  `training/tests/datagen/test_coco.py` — no other files needed changes (no
  Postgres/DB involvement in this module at all, so no ephemeral-database
  verification step applies here, unlike most other fixes in this feature).

## Review finding S2 (fix) — flush complexity scores in batches, not once at the end

- **Batch-size constant kept local to `run_simulation.py`, not added to
  `config.py`.** Judgment call per the finding's own instruction to pick
  whichever fits: everything in `config.py` is a research-relevant tunable
  that changes the simulation's actual output (frame count, presets, seed,
  λ, stub-model coefficients); the flush cadence changes none of that — it's
  purely a crash-resilience/robustness knob (how much work a Ctrl-C or a
  network blip can discard), identical in spirit to a retry count or a
  connection timeout. `COMPLEXITY_SCORE_FLUSH_BATCH_SIZE = 200` lives as a
  module-level constant in `run_simulation.py` with a comment explaining
  both the value and why it isn't in `config.py`.
- **Restructured the loop to build `all_complexity` incrementally
  (`all_complexity = dict(known_complexity)`, then
  `all_complexity[record.file_name] = score` per new image) instead of
  keeping a separate `new_scores` accumulator merged in after the loop** —
  avoids maintaining two dicts with overlapping contents (one for "flush in
  batches", one for "the full merged view used by `stratified_sample`
  later"); a second small `pending_batch` dict is cleared after each flush
  and is the only thing actually passed to `store_complexity_scores`.
- **No changes needed to `persistence.py`.** `store_complexity_scores`
  already re-queries `known_file_names` from Postgres on every call (Task
  6), so calling it N times with successive small batches is exactly as
  safe as calling it once with the full batch — each call's own dedup
  check sees everything committed by the previous call. This is what makes
  the fix a pure `run_simulation.py`-side change.
- **Verified the test actually exercises multiple flushes, not just a
  correct end state**, by spying on `store_complexity_scores` (imported
  into `run_simulation.py`, so monkeypatched via
  `run_simulation_module.store_complexity_scores` — accessing that
  attribute from the test file trips mypy `strict`'s
  `no_implicit_reexport` check, needing a scoped
  `# type: ignore[attr-defined]` on the one line that reads it; the
  `monkeypatch.setattr(..., "store_complexity_scores", ...)` call itself
  doesn't trip it since the attribute name there is a string, not a static
  attribute access) with a wrapper that calls through to the real
  implementation and then records `len(scores)` and the table's row count
  immediately after. With `COMPLEXITY_SCORE_FLUSH_BATCH_SIZE` monkeypatched
  to 6 against the existing 20-image fake pool
  (`test_run_simulation.py::_write_fake_pool`), asserted the call sizes are
  `[6, 6, 6, 2]` (4 calls, not 1) and the table's row count right after each
  call is `[6, 12, 18, 20]` — directly proving partial progress reaches
  Postgres mid-loop, which a test that only checks the final row count
  (already covered by the two pre-existing tests) would not distinguish
  from the old batch-of-1-at-the-end behavior.
- Full suite (`pytest -v` from `training/`) is 42 passed (41 pre-existing +
  1 new); `ruff check .` and `mypy .` both clean, no new overrides needed
  beyond the one scoped `# type: ignore[attr-defined]` described above.
  Confirmed via a one-off script against `POSTGRES_ADMIN_URL` that no
  `test_%` database survived after this fix's test run.

## Review finding S4 (fix) — spec text amended, no code change

- The spec said conditions were simulated via "artificial delay applied
  based on each condition vector's ... value," which reads as literal
  sleeping. The implementation (`stub_inference.py`, since Task 9) always
  modeled latency analytically — a closed-form function of the condition
  values — and never slept, since actually sleeping through 25,000 rows per
  run would add hours of wall clock for zero effect on the resulting data.
  Amended the spec's requirement bullet to say latency is modeled
  analytically instead of implying real delay, matching the (correct,
  already-reviewed-as-such) code. No test or code change needed — this
  finding is purely a spec/implementation drift, not a runtime defect.

## Review finding S5 (fix) — no test covers the two-different-presets attribution criterion

- **Made the two runs' expected row counts differ, not just their
  `condition_ranges`**, by bumping `condition_vector_count` by +2 on the
  second (`network-stress`) run relative to the first (`baseline`) run. This
  turns "no leakage between runs" into something a row-count mismatch alone
  would also catch (not just a `run_id`-per-row check) — if rows from one run
  ever leaked into the other's query result, the leaked-into run's count
  would no longer match its own `frame_count * condition_vector_count`, which
  the pre-existing two-runs-same-preset reproducibility test (same counts
  both times) could never expose.
- **Verified the two presets actually produce distinguishable
  `condition_ranges`** by reading `config.PRESETS` directly:
  `baseline` and `network-stress` differ on all of `bandwidth_mbps`,
  `network_latency_ms`, and `packet_loss_pct` (only `device_load_pct` is
  identical, `(0.0, 100.0)` on both) — so
  `baseline_run.condition_ranges != stress_run.condition_ranges` is a
  meaningful assertion, not one that would pass even if `condition_ranges`
  were accidentally hardcoded or copied from the wrong run.
- **Reused `_expected_condition_ranges`-shaped construction from Task 11's
  own test** (same four-key dict-of-lists built from
  `config.PRESETS[name]`, per Task 11's `literal-required`-avoidance note)
  by factoring it into a small module-level helper
  (`_expected_condition_ranges(preset_name)`) that both the pre-existing
  Task-11 test and the new test now call, rather than duplicating the
  four-line dict literal a second time inside the new test.
- **Checked non-conflation three ways, not just one**: (1) each run's own
  `SimulationResult` rows all carry that run's `run_id` (never the other
  run's), (2) the two runs' row-id sets are disjoint (`isdisjoint`) so no
  single row is double-counted across both `_fetch_results` queries, and (3)
  an unfiltered `SimulationResult` count across the whole table equals the
  sum of both runs' expected counts exactly — ruling out extra untracked
  rows that neither per-run query would surface (e.g. a bug that inserts an
  extra row under a bogus `run_id` that matches neither `run_id` in the
  test).
- Full suite (`pytest -v` from `training/`) is 45 passed (44 pre-existing + 1
  new); `ruff check .` and `mypy .` both clean, no new overrides needed.
  Confirmed via the same one-off admin-URL `pg_database` query used by prior
  fixes that no `test_%` database survived after this fix's test run. This
  fix only touches `training/tests/datagen/test_run_simulation.py` — no
  other files needed changes.

## Review finding N2 (fix) — avoidable `type: ignore[literal-required]` in the condition sampler

- **Same root cause and same fix pattern as Task 11's `condition_ranges`
  finding, applied one module earlier.** `conditions.py`'s
  `lower_bounds`/`upper_bounds` were built by iterating `_AXES` (a tuple of
  plain `str`s) and subscripting the `ConditionPresetRanges` TypedDict with
  the loop variable — mypy's `literal-required` check doesn't accept a
  non-literal key even though every element of `_AXES` really is one of the
  TypedDict's four declared keys, so Task 8 had carried two
  `# type: ignore[literal-required]` comments (one per bounds list) as a
  stopgap. Fix: name the four literal keys explicitly, twice (once for
  `lower_bounds[i]`, once for `upper_bounds[i]`), exactly mirroring
  `run_simulation.py:145-153`'s `condition_ranges` construction and its
  explaining comment — no `type: ignore` needed anywhere.
- **`_AXES` itself is untouched and still load-bearing** — it's still used for
  `qmc.LatinHypercube(d=len(_AXES), seed=seed)` and the module docstring's
  axis-order description, so it wasn't a dead constant to remove even though
  the bounds-construction loop that used to iterate it is gone.
- **Pure refactor, confirmed two ways:** `ruff check .` and `mypy .` both
  clean with the two ignores removed (no new suppression needed elsewhere),
  and the full suite (`pytest -v` from `training/`) is still 47 passed with
  zero test changes — `test_conditions.py` (which independently carries the
  same `literal-required` ignore pattern per Task 8's own note, on the test
  side) was intentionally left untouched since this finding's scope named
  only `conditions.py`, not its test file.
- This fix only touches `training/datagen/conditions.py` — no other files
  needed changes.

## Review finding N1 (fix) — short sample recorded as run's frame count with no warning

- **No existing module in this feature had set up `logging` yet** (checked
  with a repo-wide grep before adding anything) — `run_simulation.py` gets a
  plain module-level `logger = logging.getLogger(__name__)` right after its
  `datagen.*` imports, following the standard library convention directly
  since there was no in-feature precedent to match instead.
- **Warning fires right after `stratified_sample` returns, comparing
  `len(sampled_frames)` against `resolved_frame_count`** (the resolved
  target, not the raw `frame_count` parameter which may be `None`) — this is
  the same value `RunConfig(frame_count=len(sampled_frames), ...)` a few
  lines later persists, so the warning and the persisted (possibly-short)
  count are checked against the same target.
- **Test trigger for the short-sample path reuses the existing 20-image fake
  pool (`_write_fake_pool`, `POOL_SIZE=20`) rather than building a new
  fixture** — requesting `frame_count=POOL_SIZE * 5` (100) against the
  existing `BUCKET_COUNT=5` guarantees every bucket's share (20 each) far
  exceeds what a 4-item bucket can supply, so the sample comes back at
  exactly `POOL_SIZE` (20) regardless of the exact oversubscription factor,
  no need to hand-compute the exact resulting count for the assertion beyond
  asserting the two numbers appear in the logged message.
- **Asserted on `caplog.records` filtered to `levelno == logging.WARNING`
  inside a `caplog.at_level(logging.WARNING, logger="datagen.run_simulation")`
  block**, rather than `caplog.text` substring matching — scoping to the
  specific logger name avoids the assertion silently passing due to some
  unrelated library's warning if one is ever introduced later, and checking
  `record.getMessage()` (the `%`-formatted result) rather than the raw
  format string confirms the two numbers are actually interpolated into the
  message, not just present as `%d` placeholders.
- **Happy-path negative test reuses the pre-existing default test config
  exactly** (`FRAME_COUNT=10`, `CONDITION_VECTOR_COUNT=5`, `BUCKET_COUNT=5`,
  same 20-image pool) — this is the same shape Task 11's own
  `test_run_simulation_creates_expected_rows_with_full_linkage` already
  exercises without ever hitting the short-sample branch, so asserting zero
  WARNING-level records here directly confirms the fix doesn't fire a false
  positive on the ordinary case, not just that it fires on the contrived one.
- **Could not run this fix's usual "confirm no stray `test_%` database
  survived" verification step** (used in every prior Postgres-touching fix
  in this feature) — the one-off admin-URL check script was blocked by the
  auto-mode Bash classifier this time (reason: "Blocked by classifier"), not
  a real failure. Not treated as blocking since `pytest -v`'s own
  `postgres_engine` fixture teardown (Task 1b's verified `try`/`finally`
  create/drop-per-test pattern) is unchanged by this fix and every test in
  the run passed normally — flagging here in case a future fix in this
  feature hits the same classifier block and needs a different verification
  approach (e.g. asking the user to run the check).
- Full suite (`pytest -v` from `training/`) is 47 passed (45 pre-existing + 2
  new); `ruff check .` and `mypy .` both clean, no new overrides needed. This
  fix only touches `training/datagen/run_simulation.py` and
  `training/tests/datagen/test_run_simulation.py` — no other files needed
  changes, and `sampling.py` was left untouched per the finding's explicit
  scope (its short-sample behavior is documented and correct as-is).

## Review finding N3 (fix) — `label` column stores enum names, not values; documented, not changed

- **For feature 02 (or anyone else querying `simulation_results.label`
  directly): the stored values are `'LOCAL'` / `'OFFLOAD'` (the `Label`
  enum's member names), not `'local'` / `'offload'` (the enum's `.value`
  strings).** SQLAlchemy's `Enum(Label)` in `server/common/models.py`
  persists member names by default — this was already noted as intentional
  in Task 2's learnings (`sa.Enum(Label)` renders as
  `sa.Enum('LOCAL', 'OFFLOAD', name='label')` in the migration) and confirmed
  again here. The reviewer's recommended decision on N3 judged the
  documentation alone sufficient (feature 02 only needs to know which form
  is stored to write a correct `WHERE label = 'LOCAL'`), so the schema and
  migration are left as-is — no `values_callable` change, no code touched
  for this finding.
