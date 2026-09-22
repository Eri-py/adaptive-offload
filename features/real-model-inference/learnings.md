# Learnings — real-model-inference

## Task 1

- The ephemeral test DB fixture (`database/common/testing.py`) builds schema via
  `Base.metadata.create_all`, not by running migrations. Reusing `Enum(Label)` for a
  second column (`ModelInference.model_path`) across a different table works fine there
  with no extra config: `create_all` calls `checkfirst=True` per column's enum type, so
  the second column's type creation is a no-op once the first (`SimulationResult.label`)
  has created the Postgres `label` type. Only the *migration* script needs the explicit
  `create_type=False` guard, since Alembic's `--sql` offline rendering has no DB to check
  against and would otherwise emit a second `CREATE TYPE label` and fail on a real apply.
- `alembic upgrade head --sql` from `database/` is a reliable, fully offline way to verify
  a new migration's DDL (including confirming it does *not* reissue `CREATE TYPE`) without
  touching Postgres at all — same pattern migration `0001` used.
- Composite PKs with an enum member as one of the parts round-trip fine through
  `session.get(Model, (a, b, Label.X))`.

## Task 2

- Initially over-built `load_ground_truth`'s per-entry validation (checking each
  `categories`/`annotations` entry's shape, plus a `category_id`-not-found check),
  mirroring `image_source.load_image_index`'s per-entry checks too literally. The
  plan explicitly scopes this out ("Full five-case shape validation ... is not
  required here — file-level and key-level validation is sufficient"), and adding
  it anyway means untested code paths (extra `ValueError` branches with no test
  coverage) — trimmed back to only file-existence, JSON-validity, and top-level
  `annotations`/`categories` key-presence-and-list-type checks, letting a genuinely
  malformed entry surface as a plain `KeyError`/`ValueError` from tuple-unpacking
  instead of a custom message. Worth reading a task's explicit scope-narrowing
  language carefully before pattern-matching a sibling module's convention wholesale.
- `ruff`'s line-length limit in `training/` is 100 chars (not the more common 88/120) —
  a single-line multi-kwarg `Box(...)` assertion tripped `E501`; had to wrap it.

## Task 3

- `score_accuracy`/`_iou` are pure and only depend on `ground_truth.Box`, so no DB or
  fixture setup was needed for tests — plain module-level `Box` constants sufficed.
- Picking IoU test fixtures by exact fraction (rather than "clearly high"/"clearly
  low") makes the below-threshold and custom-threshold tests self-documenting: two
  10x10 boxes offset by (5, 5) give intersection 25 / union 175 = IoU ≈ 0.143, which
  is below the default 0.5 threshold but above a relaxed 0.1 one — one fixture pair
  covers both the "doesn't match by default" and "does match with a lower threshold"
  cases with a comment stating the exact number instead of a vague description.
- No surprises versus Task 2's setup: same venv activation (`source ../.venv/bin/activate`
  from `training/`), same ruff (100-char line length) and mypy (`strict = true`)
  config applied cleanly with no adjustments needed.
