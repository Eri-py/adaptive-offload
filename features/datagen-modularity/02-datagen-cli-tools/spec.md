# Data-Gen Config Split and CLI Tools

## Overview

Split `training/datagen/config.py`'s tunables from its condition-scenario
presets, and add standalone CLI entry points for the simulator's reusable
pieces (scene-complexity scoring, stratified-sample preview,
condition-vector preview, and re-labeling an existing run under a different
λ) so each is independently usable without running the full
`run_simulation.py` pipeline. Also separate COCO's one-time
dataset-acquisition (download) step from the ongoing pipeline's read path.

## Requirements

- The condition-scenario presets are defined in their own module, separate
  from the general simulation tunables (frame count, seed, λ, dataset name,
  stub-model coefficients).
- A standalone entry point scores every image in a given folder for scene
  complexity and reports the results — the folder is not required to be the
  COCO cache; any folder of images works.
- A standalone entry point previews a stratified frame sample (which frames
  a given target count/bucket count/seed would select) without requiring a
  full simulator run.
- A standalone entry point previews the condition vectors a given
  preset/count/seed would sample without requiring a full simulator run.
- A standalone entry point recomputes win/loss labels for an existing
  simulator run under a supplied λ, using that run's already-persisted
  latency/accuracy values, without re-running scene-complexity scoring,
  frame sampling, condition sampling, or stub inference.
- COCO's dataset-file download step becomes a standalone, explicitly-run
  entry point (populate/refresh the local COCO image cache) rather than
  something the main simulator pipeline can trigger automatically mid-run.
  The main pipeline and the complexity-scoring entry point only ever read
  local files, and fail clearly (not silently fetch) if a needed file is
  missing.
- Enumerating which images belong to the COCO val2017 dataset (reading the
  annotation file) continues to work as it does today — this requirement
  doesn't change that.
- Running the full simulator pipeline end to end still produces the same
  output (same rows, same fields, same reproducibility guarantees) as it
  does today — none of the above changes its behavior for a normal full
  run.

## Out of Scope

- Any change to the shared DB package's location (covered by the sibling
  01 spec).
- Any change to the actual win/loss utility formula, the stub-inference
  model, or the scene-complexity algorithm — the new entry points reuse
  existing logic, they don't change it.
- Persisting re-labeled results anywhere — the re-labeling entry point
  reports/outputs recomputed labels; it does not overwrite the original
  run's stored labels or create a new run record.
- Adding entry points for `persistence.py`'s database-write functions —
  those remain internal to the pipeline, not exposed as standalone tools.
- Supporting datasets other than COCO val2017 — the generic
  complexity/sampling/condition tools work on arbitrary input, but no new
  dataset-acquisition adapter (e.g. for Open Images V7) is built here.

## Acceptance Criteria

- Given `training/datagen/config.py` and a new presets module after the
  split, when either is inspected, then tunables and presets are no longer
  defined in the same file.
- Given a folder of images that isn't the COCO cache, when the
  complexity-scoring entry point is run against it, then it reports a
  complexity value for every image in that folder without requiring any
  other part of the pipeline to run.
- Given a target frame count, bucket count, and seed, when the
  sample-preview entry point is run, then it reports the selected frames
  without writing anything to Postgres or running any other pipeline stage.
- Given a preset name, count, and seed, when the condition-preview entry
  point is run, then it reports the sampled condition vectors without
  writing anything to Postgres or running any other pipeline stage.
- Given an existing simulator run and a new λ value, when the re-labeling
  entry point is run, then it reports the label each row would have under
  that λ, computed from that run's already-persisted latency/accuracy
  values, without re-scoring images, re-sampling frames, re-sampling
  conditions, or re-running stub inference.
- Given the local COCO image cache is missing a file, when the main
  simulator pipeline or the complexity-scoring entry point runs, then it
  fails with a clear error rather than silently downloading the missing
  file.
- Given the local COCO image cache is missing files, when the new
  cache-population entry point is run, then it downloads exactly the
  missing files and leaves already-cached files untouched.
- Given the full simulator pipeline is run end to end after these changes,
  when its output is compared to a pre-change run with the same
  config/seed, then the rows produced are identical.

## Open Questions

None — resolved during spec review (four new standalone entry points:
complexity scoring, sample preview, condition preview, re-labeling; COCO
acquisition split into a separate cache-population entry point;
config/presets split into two modules).
