# Feature Router Experiment

## Overview

Using the frame features from spec 01, this experiment tests whether a router
can beat "always offload" by more than noise. It uses the existing desktop
data only, and it decides whether the router idea is worth building the
phone and server infrastructure for. It compares two designs: deciding
before any inference, and a cascade that runs locally first and offloads only
when the small model looks unsure.

## Requirements

- **Feature diagnostics:** for every candidate feature (spec 01's features
  plus the existing scene complexity), the experiment reports how strongly it
  relates to the per-frame accuracy gap between the local and offload models.
  It uses all 5,000 frames, whose accuracies are already stored.
- **Held-out photos:** the photos used for testing are never used in any
  training step. That includes models trained on the 5,000-frame pool. The
  test photos are the existing frame-level split's held-out photos from the
  500 simulated frames.
- **Decide-first router:** a router that picks local or offload before any
  inference runs. It uses the new image features, scene complexity, and the
  existing network and device features.
- **Cascade router:** the local model always runs first. The router then
  decides whether to accept the local result or also send the frame to the
  server. It can use the local-model confidence features, the image features,
  and the network and device features.
  - An accepted request is scored with the local path's accuracy and latency.
  - An escalated request is scored with the offload path's accuracy and a
    latency equal to the local latency plus the offload latency.
- **Evaluation**, scored on the held-out photos' simulated rows with the same
  utility formula and weighting the stored labels were generated with:
  - average utility for always-local, always-offload, the existing oracle
    ceiling, and each router
  - each router's share of the headroom between always-offload and the oracle
  - a 95% bootstrap confidence interval, resampling held-out photos, for each
    router's utility minus always-offload's
- A router counts as **useful** only if that confidence interval lies
  entirely above zero.
- The experiment uses the same small model families as the existing findings
  (a linear model and gradient-boosted trees), so its results stay comparable
  with them.
- All randomness (the split, the bootstrap, model training) is seeded, so
  re-running gives the same numbers.
- The experiment is a runnable script, not a notebook.
- `training/router/findings.md` gains a new section with:
  - the diagnostics
  - both routers' results against the baselines and the ceiling
  - whether each router met the usefulness bar
  - spec 01's feature compute times
  - a recommendation on whether to continue with the phone and server work

## Out of Scope

- Computing or storing features (spec 01).
- Phone-measured latencies: all latencies are the stored desktop values.
- Changing the utility formula, sweeping its weighting, or re-labelling
  simulation rows.
- New simulation runs or more simulated frames.
- Model families beyond the linear and gradient-boosted ones, and
  hyperparameter searches.
- Deploying either router to the phone or the server.
- Changing the existing router scripts' behaviour or results.

## Acceptance Criteria

- Given spec 01's features are stored, when the experiment runs, then it
  prints each candidate feature's relationship with the accuracy gap across
  all 5,000 frames.
- Given the frame-level split, when the experiment trains any model, then no
  held-out test photo appears in that model's training data.
- Given the held-out rows, when the decide-first router is evaluated, then
  the output shows its average utility next to always-local, always-offload
  and the oracle, plus its share of the headroom.
- Given the held-out rows, when the cascade router is evaluated, then each
  escalated row is scored with offload accuracy and local-plus-offload
  latency, each accepted row with local accuracy and latency, and the same
  comparisons are reported.
- Given each router, when the experiment finishes, then it reports a 95%
  bootstrap confidence interval (resampling held-out photos) for utility
  minus always-offload's, and states whether the router meets the usefulness
  bar.
- Given the same database contents, when the experiment runs twice, then
  every reported number is identical.
- Given the experiment has run, then `training/router/findings.md` has a new
  section covering the diagnostics, both routers' results, the usefulness
  verdicts, the feature compute times, and a recommendation on the phone and
  server work.
- Given this feature is complete, then the existing router scripts still run
  and produce their previous results.

## Related specs

- `features/router-frame-features/01-frame-feature-extraction/`: computes and
  stores the features this experiment uses.

## Open Questions

None.
