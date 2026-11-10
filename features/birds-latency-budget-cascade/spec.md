# Birds Cascade Under a Hard Latency Budget

## Overview

The birds cascade is currently reported as accuracy against the *share* of
photos escalated to the large model, with no notion of what escalating costs.
That tells us the cascade recovers the small-to-large accuracy gap, but not
whether it is affordable: a cascade pays for the small model first and the
large model second, so under a deadline it can miss the deadline entirely.
This feature evaluates the same cascade under a hard per-request latency
budget, sweeping what the local and offload paths cost, and reports where the
cascade beats running only the small model or only the large one — and how that
boundary moves as the budget tightens.

The hard-budget framing comes from DeepDecision (INFOCOM 2018), which treats
latency as a constraint rather than folding it into a blended score; the
escalate-on-low-confidence framing comes from Big/LITTLE (CODES+ISSS 2015) and
DDNN (ICDCS 2017). Attempt 1 applied the same framing to COCO, and this is the
birds counterpart of that experiment.

## Requirements

- **The experiment reads the stored per-photo bird predictions and runs no
  inference.** Every accuracy figure comes from the recorded predictions for
  the two committed checkpoints. No model is retrained, re-evaluated or loaded.

- **Latency is supplied as swept parameters, not read per photo.** Unlike
  attempt 1, no per-photo latency exists for birds, so each evaluated cell
  names what the two paths cost:
  - **Local cost** is swept continuously over a range wide enough to cover any
    plausible on-device cost for the small model. The one real birds
    measurement — the small model's desktop CPU time — is marked on every
    output, and the write-up states plainly that no on-device birds measurement
    exists yet, so the local axis is a sweep rather than a calibrated figure.
  - **Offload cost** is composed of a server-compute term, a transfer term
    (payload divided by available bandwidth) and a round-trip term. The
    server-compute term comes from the large model's existing measured
    per-photo GPU time, and the write-up states that a desktop GPU is standing
    in for a server one. The payload is a single fixed size for every request,
    measured by encoding one photo at the models' input resolution, because a
    deployed scan sends a freshly captured frame at a fixed input size rather
    than an archive file.
  - The offload axis is likewise swept continuously, and the named network
    conditions the project's design calls for (a good, a medium and a poor
    condition, each a bandwidth and round-trip pair) are marked on it so a
    result can be read as "offload wins on a good network but not a poor one".

- **The stored predictions are a desktop stand-in for the local path, and the
  write-up says so.** Both models' predictions were computed on a desktop GPU
  in full precision, but the local path in a deployed app would run an exported,
  likely reduced-precision model on the phone. Conversion shifts not only
  accuracy but the confidence values themselves, which is exactly what the
  cascade's threshold keys on. So every cascade figure here describes the
  desktop small model, not a measured on-device one, and the write-up states
  that measuring the exported model's predictions — not just its speed — is
  required before any threshold is trusted on a phone.

- **Offload cost is treated as known exactly, and the write-up names what that
  hides.** Each cell fixes one offload cost, so no policy is ever surprised by
  a slow network. A deployed router instead works from an *estimate*, and the
  cascade has a property this design cannot score: having run the small model
  first, it already holds an answer, so it can return the local result when the
  network disappoints rather than missing the deadline with nothing. The
  write-up records this as the cascade's unmeasured advantage and names
  offload cost drawn with noise as the next experiment.

- **A consequence of scenario-level latency is stated explicitly in the
  write-up:** because every photo in a cell shares the same local and offload
  cost, the two static baselines are all-on-time or all-late within a cell, and
  the cascade is the only policy whose decision varies photo to photo. This is
  a structural difference from attempt 1, where latency varied per request.

- **Five policies are evaluated in every cell**, each deciding per request
  under a hard budget:
  - **Always-local** — run the small model only.
  - **Always-offload** — run the large model only.
  - **Budget-only** — offload when the offload cost fits the budget, otherwise
    run locally. This is the DeepDecision-style policy carried over from
    attempt 1.
  - **Cascade** — always run the small model first; keep its result when its
    confidence meets a threshold; otherwise escalate to the large model, but
    only when local plus offload cost still fits the budget, and keep the local
    result when it does not. An escalated request is charged local plus offload
    cost and takes the large model's correctness.
  - **Within-budget oracle** — knows each photo's true correctness under both
    models and picks the most accurate option that fits the budget, falling
    back to local when nothing fits. It bounds what any router could achieve.

- **The cascade's routing rule is a deterministic confidence threshold, and no
  learned router is built.** Attempt 1 tested a learned score against a plain
  threshold head to head and they produced identical accuracies at most
  budgets, so a learned variant has to earn its place before it is worth
  building again.

- **Both recorded confidence measures are compared** as the cascade's
  threshold input — the small model's highest class probability, and its margin
  between the top two classes — since the stored records show the margin
  recovering more of the accuracy gap at every escalation level.

- **The threshold is tuned on the validation records and reported on the test
  records.** The operating threshold for a cell is the one with the highest
  on-time accuracy on the validation photos, and that threshold is then applied
  to the test photos. No test photo informs the threshold. The write-up notes
  both that the validation split is small enough to make the tuned threshold
  noisy, and that the split was already used once for checkpoint selection.

- **Every policy is scored on the same four metrics** in every cell: mean
  accuracy, **on-time accuracy** (a result arriving after the budget counts as
  zero), mean latency, and the share of requests that missed the budget.

- **Uncertainty is reported for the cascade's advantage.** For each reported
  cell, the tuned cascade's on-time accuracy minus budget-only's, and minus
  always-local's, each with a 95% bootstrap interval resampling test photos.
  The cascade counts as **useful in a cell** only when its interval against
  budget-only lies entirely above zero.

- **The reported budgets span tight to loose**, chosen so that at least one
  budget admits no offload on a poor network and at least one admits
  escalation on a good one.

- **The experiment is a runnable script whose output is deterministic:** two
  runs over unchanged records produce identical numbers, and all randomness is
  seeded.

- **The run is recorded in experiment tracking** as its own run, as every
  training and evaluation run in this project is.

- **The write-up lands in the project's birds findings file** and contains:
  - a boundary figure per budget over the local-cost-by-offload-cost plane,
    showing which policy wins where, with the real desktop measurement and the
    named network conditions marked
  - per-budget metric tables for every policy at the marked latencies
  - the usefulness verdict per reported cell
  - a statement of where the boundary sits and how it moves as the budget
    tightens — the result this experiment exists to produce
  - the four budget bands implied by the two costs — nothing fits, local only,
    offload but no escalation, and everything — with the arithmetic fact that
    the escalation band always sits above the offload band, since local plus
    offload exceeds offload alone. A hard budget therefore never makes the
    cascade the only option that fits, so the cascade's case has to be made on
    accuracy and on offload calls avoided rather than on fitting a deadline
    nothing else fits.
  - the width of the band where offload fits but escalation does not, which
    equals the local cost — so cheaper local inference widens the range of
    budgets in which the cascade can act at all
  - the fewest offload calls at which the cascade matches always-offload's
    accuracy, which is the cascade's payoff when accuracy ties
  - a short related-work note citing the three papers above
  - an explicit statement that the local axis is unanchored on real hardware,
    naming the on-device measurement that would anchor it

## Out of Scope

- Measuring the birds models on the phone, exporting either model to Core ML,
  and any change to the mobile benchmark. The local axis stays a sweep; this
  feature is what makes a future measurement readable off a curve.
- Deploying the large model behind a real endpoint, or measuring real network
  latency. The offload cost stays analytical.
- Re-using attempt 1's YOLOv8n phone figures as birds anchors. They measure a
  different model on a different task.
- Any learned router, learned confidence score, or trained latency predictor.
- Measuring the exported, reduced-precision model's predictions or confidence.
  The stored desktop predictions stand in for the local path.
- Offload cost drawn with noise, network jitter, or any model of estimation
  error. Each cell's costs are exact.
- The soft utility score and any lambda weighting. Latency is a hard
  constraint here, not a penalty term.
- Battery, energy, bandwidth caps, frame rate, and thermal constraints.
- Stale-result accuracy decay. A request is a single scan.
- Retraining either model, changing the training recipe, or changing the split
  definition or its seed.
- Recording new per-photo predictions, adding or altering a table, authoring or
  running a migration, and any create or drop of a database.
- Any change to attempt 1's code, its results, or its findings file.
- Choosing among more than the two committed models, or among input
  resolutions.
- Anything phone-side or server-side: no endpoint serves this result.

## Acceptance Criteria

- Given the stored predictions and a cell naming a local cost, an offload cost
  and a budget, when the experiment runs, then it reports mean accuracy,
  on-time accuracy, mean latency and over-budget share for always-local,
  always-offload, budget-only, the cascade under each confidence measure, and
  the within-budget oracle.
- Given the experiment has run, then no model was loaded and no inference was
  performed; every accuracy traces to the stored records.
- Given a cell, when the offload cost is computed, then it is the sum of a
  server-compute term, the fixed payload divided by that cell's bandwidth, and
  that cell's round-trip time.
- Given a cell whose offload cost exceeds the budget, when budget-only decides,
  then every request runs locally.
- Given a cascade request whose confidence meets the threshold, then its local
  result is kept and it is charged local cost only.
- Given a cascade request whose confidence is below the threshold, then it is
  escalated and charged local plus offload cost with the large model's
  correctness when that total fits the budget, and otherwise keeps the local
  result.
- Given any cell, when the cascade's operating threshold is chosen, then it was
  selected on validation photos alone, and no test photo influenced it.
- Given a result whose charged latency exceeds the budget, when on-time
  accuracy is computed, then that result counts as zero.
- Given any cell, then the within-budget oracle's on-time accuracy is at least
  that of every other policy in that cell.
- Given a cell where the local cost alone exceeds the budget, then every policy
  reports zero on-time accuracy, because nothing can arrive in time.
- Given any cell, then both static baselines report an over-budget share of
  either zero or one, and this degenerate behaviour is stated in the write-up
  as a structural difference from attempt 1.
- Given each reported cell, then the output shows 95% bootstrap intervals for
  the tuned cascade's on-time accuracy against budget-only and against
  always-local, and a usefulness verdict following the stated rule.
- Given the reported budgets, then at least one budget admits no offload under
  the poor network condition, and at least one admits escalation under the good
  one.
- Given unchanged records, when the experiment runs twice, then every reported
  number is identical.
- Given the experiment has completed, when experiment tracking is inspected,
  then this run is recorded there.
- Given the experiment has completed, when the birds findings file is read,
  then it carries a boundary figure per budget, the per-budget tables, the
  usefulness verdicts, a statement of where the boundary sits and how it moves
  with the budget, the related-work note, and the statement that the local axis
  is unanchored on real hardware together with the measurement that would
  anchor it.
- Given the boundary figure, then the small model's real desktop measurement
  and the three named network conditions are marked on it, and no phone-side
  figure from attempt 1 appears as a birds anchor.
- Given the birds findings file, then it states that the cascade figures come
  from desktop full-precision predictions standing in for the local path, and
  that the exported model's predictions and confidence must be measured before
  a threshold is trusted on a phone.
- Given the birds findings file, then it records that offload cost is exact in
  every cell, names the cascade's unmeasured hedging advantage, and names
  offload cost drawn with noise as the next experiment.
- Given the birds findings file, then it sets out the four budget bands, states
  that the escalation band always sits above the offload band, and gives the
  lockout band's width as equal to the local cost.
- Given the birds findings file, then it reports the fewest offload calls at
  which the cascade matches always-offload's accuracy, or states that it never
  matches it.
- Given this feature is complete, then the existing prediction-recording and
  summary entry points still run and produce their previous numbers, and no
  migration has been authored or run.

## Open Questions

None.
