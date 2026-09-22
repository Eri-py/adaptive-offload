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
