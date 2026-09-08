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
