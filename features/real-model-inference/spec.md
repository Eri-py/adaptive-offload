# Real Model Inference for the Data-Gen Simulator

## Overview

Replace `stub_inference`'s fully-synthetic latency/accuracy model with real
object detection: a real local model and a real offload model each run for
real on every sampled frame, producing real measured latency and real
accuracy (scored against COCO ground truth), instead of numbers drawn from
hand-picked formulas.

## Requirements

- The local path runs YOLOv8n (nano) on CPU; the offload path runs YOLOv8x
  (extra-large) on GPU. Both are pretrained (COCO weights), used purely for
  inference — no training/fine-tuning.
- For each sampled frame, each model's real inference latency and real
  detection accuracy are measured once per (dataset, model), not recomputed
  per condition vector — a frame's real detection result does not depend on
  simulated network/device conditions.
- Real per-(dataset, frame, model) inference results (latency + accuracy)
  persist in Postgres and are reused across runs, the same way
  `scene_complexity` scores are today: a rerun of a dataset only computes
  real inference for frames not already covered.
- Detection accuracy is measured per frame as a simple IoU match rate
  against that frame's real ground-truth boxes (from the COCO-format
  annotations file's `annotations` array, matched to the frame via
  `image_id`): for each ground-truth box, check whether any same-class
  predicted box matches it at IoU ≥ 0.5; accuracy is the fraction of
  ground-truth boxes matched. A frame with zero ground-truth boxes needs a
  defined accuracy value (see Open Questions).
- `network_bandwidth_mbps`/`network_latency_ms`/`network_packet_loss_pct`
  (offload) and `device_load_pct` (local) continue to add synthetic latency
  overhead on top of each path's real measured base latency, using the same
  condition-driven formula shape `stub_inference` uses today (config-driven
  coefficients) — only the base latency changes from a constant to a real
  measurement.
- Conditions no longer affect accuracy at all. Every synthetic
  accuracy-modeling coefficient in `config.py` (scene-complexity penalty,
  device-load penalty, packet-loss penalty, accuracy noise, and the
  `LOCAL_BASE_ACCURACY`/`OFFLOAD_BASE_ACCURACY` constants) is removed —
  accuracy is only ever the real measured value for that (dataset, frame,
  model).
- `run-simulation`'s existing behavior for a given (frame, condition) row —
  persisting one `simulation_results` row with latency, accuracy, and a
  computed win/loss label — is unchanged in shape; only where
  latency/accuracy come from changes.
- `run-simulation`'s progress logging (already present for complexity
  scoring) is matched by equivalent progress logging for the real-inference
  pass, since this pass is expected to take real, possibly multi-minute
  wall-clock time on a full pool.

## Out of Scope

- Building `training/router/` (the actual decision-layer/router model) — a
  separate, future feature. This feature only changes what feeds
  `simulation_results`.
- Any change to frame sampling, condition-vector sampling, presets, or
  win/loss labeling — those stay exactly as they are today.
- Full COCO mAP via `pycocotools` — this feature uses the simpler
  hand-rolled per-frame IoU match rate described above instead.
- Fine-tuning or training either YOLO model.
- Live/on-device deployment of either model to the actual RN app — this
  remains the offline data-gen simulator, run in Python on a dev machine.
- Re-running or migrating any of the existing `simulation_results` rows
  already produced by the old fully-synthetic `stub_inference` — those stay
  in Postgres as historical data from before this feature; only new runs
  use real inference.
- Any change to how `scene_complexity` itself is computed or cached — the
  new real-inference cache is a parallel, independent table, not a change
  to the existing one.

## Acceptance Criteria

- Given a `run-simulation` invocation against a dataset with no cached real
  inference results yet, then every sampled frame gets real YOLOv8n (CPU)
  and real YOLOv8x (GPU) inference run on it, and the results are persisted
  for reuse.
- Given a second `run-simulation` invocation against the same dataset
  (frames already covered by cached real inference), then no real model
  inference is re-run for those frames — the cached latency/accuracy values
  are reused directly.
- Given a sampled frame and its two real inference results, then every
  `simulation_results` row for that frame (across all sampled condition
  vectors) uses that same real base latency/accuracy, with only the
  condition-driven latency overhead varying row to row — accuracy is
  identical across every condition for a given frame.
- Given a frame's real detection output and its real COCO ground truth,
  then accuracy is computed as the fraction of ground-truth boxes matched
  by a same-class predicted box at IoU ≥ 0.5.
- Given the resulting `simulation_results` rows, then win/loss labeling
  (`compute_label`) runs unchanged on the real latency/accuracy values,
  exactly as it does today on synthetic ones.

## Open Questions

- Exact behavior for a frame with zero ground-truth boxes in the
  annotations (accuracy is undefined by the match-rate formula as written —
  e.g. treat as `1.0` since nothing was missed, exclude the frame, or some
  other convention) is deferred to the implementation plan.
- Where the new caching table's schema and the ground-truth-parsing/
  IoU-matching logic live (module organization) is deferred to the
  implementation plan, per the spec's own "no implementation details" rule.
