# Shared DB Package

## Overview

Extract the DB layer (SQLAlchemy models, engine/session setup, the
ephemeral-test-database fixture helper, and the Alembic migration) out of
`server/` into its own top-level package, so `server/` and `training/` both
depend on it as equal peers instead of `server/` owning it and `training/`
borrowing it.

## Requirements

- The SQLAlchemy models (`SimulationRun`, `SimulationResult`,
  `SceneComplexity`), the engine/session-factory setup, and the
  ephemeral-test-database fixture helper move out of `server/common/` into a
  new top-level package with its own dependency manifest, not nested inside
  `server/`.
- The Alembic migration setup (config, environment script, and the existing
  unapplied migration) moves out of `server/migrations/` into the same new
  top-level package.
- `server/` and `training/` both depend on the new package the same way —
  neither is the "owner" the other imports from; both declare it as an
  external editable dependency.
- The move changes nothing about what the models, engine, fixture, or
  migration actually do — same tables, same columns, same fixture behavior,
  same migration contents.
- Every existing automated test that exercised this code continues to pass
  after the move.
- The project's documented repo-layout conventions
  (`.claude/coding-guidelines.md`) are updated to describe the new package's
  location and its relationship to `server/` and `training/`.

## Out of Scope

- Any schema change — no new/changed/removed tables or columns.
- Any change to `server/api/` (doesn't exist yet) or to `training/datagen/`'s
  actual simulation logic.
- Applying the migration or touching the real Postgres instance — unchanged,
  still the user's action alone.
- The config/presets split and the new CLI tools (covered by the sibling
  02 spec).

## Acceptance Criteria

- Given the new package's location, when `server/` or `training/` code
  imports the shared models, engine, or test fixture, then the import
  resolves correctly from each package's own environment.
- Given the full `server/` test suite, when it's run after the move, then
  every test that passed before still passes, with the same count.
- Given the full `training/` test suite, when it's run after the move, then
  every test that passed before still passes, with the same count.
- Given the Alembic migration after the move, when it's rendered offline
  (`alembic upgrade head --sql`), then it produces the same DDL as before
  the move, without connecting to any database.
- Given `.claude/coding-guidelines.md` after the move, when its repo-layout
  section is read, then it accurately describes the new package's location
  and that `server/` and `training/` depend on it symmetrically.
- Given the old `server/common/` and `server/migrations/` locations, when
  the codebase is inspected after the move, then neither directory still
  contains the moved code.

## Open Questions

None — resolved during spec review (new package sits at the repo top
level, `server/`/`training/` depend on it symmetrically via editable
install, pure structural move with no behavior change).
