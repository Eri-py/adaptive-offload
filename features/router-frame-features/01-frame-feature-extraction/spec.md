# Frame Feature Extraction

## Overview

The router can't beat "always offload" because its only frame-level input,
scene complexity (edge density), carries no signal about which path wins
(`training/router/findings.md`). Before any more phone or server work, we
want to test whether better frame-level features exist. This spec computes
two groups of candidate features for every frame and stores them in Postgres,
so spec 02 can analyse them. It is the first of two specs (see Related specs).

## Requirements

- A runnable step computes candidate features for every frame of the
  configured dataset (`coco_val2017`, all 5,000 frames) and stores one row per
  frame in Postgres.
- **Image features** (computed from the image alone, before any model runs),
  each chosen to be cheap enough to compute on a phone:
  - sharpness (how blurred the image is)
  - brightness
  - contrast
  - colourfulness
  - entropy (how much varied detail the image has)
- Edge density is already stored as the scene-complexity score and is not
  recomputed.
- **Local-model confidence features** (from running YOLOv8n on the frame):
  - number of detections
  - highest, mean and lowest detection confidence
  - mean and smallest detected box size, relative to the image area
- The confidence features come from the same YOLOv8n weights and detection
  settings that produced the stored local-path accuracy, so they describe the
  same predictions that accuracy was scored on.
- A frame with no detections gets defined values: a detection count of zero
  and zero for the confidence and box-size features. It is never skipped or
  left with missing values.
- Re-running the step only computes frames that don't have features stored
  yet, like the existing complexity scoring step.
- When the step finishes, it reports the average time per frame to compute
  each feature group, as a rough measure of how cheap the features are.
- The step fails with a clear message if an image or the model weights are
  missing. It never downloads anything.
- The new table is created by a migration that the user runs. No existing
  table changes.

## Out of Scope

- Analysing the features, training routers, or updating `findings.md`
  (spec 02).
- Recomputing or changing edge density, the stored accuracies, or the stored
  latencies.
- Features from the large (offload) model's output.
- Measuring feature cost on the phone. The desktop timing is only a rough
  proxy.
- New simulation runs.
- Any app or server change. The server/phone work in
  `features/server-served-benchmark/` is parked until this experiment
  reports.

## Acceptance Criteria

- Given the migration has been applied and the step has run, when the new
  table is queried, then it holds exactly one row for each of the 5,000
  `coco_val2017` frames, and every feature listed above has a value.
- Given a frame on which YOLOv8n makes no detections, when its row is
  inspected, then the detection count and the confidence and box-size
  features are zero rather than missing.
- Given a frame, when its confidence features are compared with a fresh run
  of YOLOv8n using the simulator's weights and settings, then they match.
- Given the step has already run on some frames, when it runs again, then it
  computes only the frames without stored features and leaves existing rows
  unchanged.
- Given the step finishes, then its output includes the average per-frame
  compute time for the image features and for the confidence features.
- Given an image file or the YOLOv8n weights are missing, when the step runs,
  then it stops with a message naming what is missing and downloads nothing.
- Given this feature is complete, then the only schema change is the new
  table, added through a migration.

## Related specs

- `features/router-frame-features/02-feature-router-experiment/`: measures
  how well these features predict the local-vs-offload accuracy gap, trains a
  decide-first router and a cascade router on them, and records the results
  in `training/router/findings.md`.

## Open Questions

None.
