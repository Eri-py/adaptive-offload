# Training Reorganisation

## Overview

The project is pivoting from general object detection (COCO) to flower species
identification. Before the flower work starts, the existing training code
should sit in its own folder as "attempt 1", with an empty folder ready for
"attempt 2", so the two attempts stay clearly separated. This is a pure move:
nothing about what the code does or produces may change.

## Requirements

- All existing training code (the data-generation simulator and all router
  code, including the findings write-up and its figure) moves into a
  `training/coco/` folder representing attempt 1.
- The existing tests move into a matching location, still mirroring the source
  layout as the coding guidelines require.
- A `training/flowers/` folder exists for attempt 2, empty apart from what is
  needed for it to be a valid, importable package.
- No shared folder is created yet. Pieces the flower work reuses will be
  moved into a shared location later, in the spec that first reuses them.
- The COCO dataset (`training/data/`) and the model weights
  (`training/models/`) stay where they are, and the moved code still finds
  them.
- Behaviour is unchanged:
  - every test still passes
  - both router experiments (the feature-router experiment and the
    latency-budget experiment) print byte-identical output and produce a
    byte-identical figure
  - the two registered command-line tools (`run-simulation` and
    `score-complexity`) still work under their existing names
- Documentation that describes the current layout is updated to the new
  paths: `.claude/coding-guidelines.md` and any run instructions in the
  moved `findings.md`. Historical feature specs, plans and reports under
  `features/` are left as they are, because they record what was true at
  the time.

## Out of Scope

- Any change to what the code does, including refactors, renames beyond the
  folder move, and bug fixes.
- Creating a shared folder or splitting any file between attempts.
- Any flower-classification code.
- Moving the COCO dataset, model weights, the database layer (`database/`),
  the server (`server/`) or the app (`app/`).
- Rewriting historical documents under `features/`.
- Any database or schema change.

## Acceptance Criteria

- Given the reorganisation is complete, then all existing training source
  code is under `training/coco/`, and no training source code remains at the
  old top-level locations.
- Given the reorganisation is complete, then the tests live in a location
  mirroring the new source layout, and the full training test suite passes
  with the same number of tests as before (174), none skipped.
- Given the reorganisation is complete, then `training/flowers/` exists, can
  be imported, and contains no attempt-specific code.
- Given the COCO dataset and model weights at their current paths, when the
  moved code resolves those paths, then it finds the same files as before.
- Given the real database (read-only), when the feature-router experiment and
  the latency-budget experiment are run before and after the move, then their
  printed output is byte-identical and the latency-budget figure is
  byte-identical.
- Given the training package is reinstalled into the shared virtual
  environment, when `run-simulation --help` and `score-complexity --help` are
  run, then both work under their existing names.
- Given the reorganisation is complete, then ruff is clean and mypy reports
  no errors beyond the 9 already known.
- Given the reorganisation is complete, then `.claude/coding-guidelines.md`
  and the moved `findings.md` describe the new paths, and no file under
  `features/` has been edited apart from this spec's own workflow documents.

## Open Questions

None.
