# Latency-Budget Router

## Overview

Spec 02 showed that under the soft utility (`accuracy − 0.3 × latency in
seconds`), "always offload" can't be beaten: latency is nearly free, so no
router gains anything. Published systems frame this decision differently.
DeepDecision (INFOCOM'18) treats latency as a hard constraint. DDNN
(ICDCS'17) and Big/LITTLE (CODES+ISSS'15) keep the local result when the
small model is confident and escalate otherwise, and they report a trade-off
curve instead of a single weighted score. This experiment reframes the
router that way:

- each request has a hard latency budget
- the router maximises detection accuracy within it
- results are reported as curves over the budget and the confidence
  threshold, using the existing simulation data

## Requirements

- **Two routing policies,** each deciding per request given a latency budget:
  - **Budget-only:** offload if the predicted offload latency fits within the
    budget; otherwise run locally.
  - **Budget + cascade:** always run locally first. Keep the local result if
    its confidence meets a threshold. Otherwise escalate to the offload path,
    but only if the predicted offload latency fits within what's left of the
    budget after the local run. If it doesn't fit, keep the local result.
    - An escalated request is charged local plus offload latency and gets
      offload accuracy.
    - A kept local result gets local latency and local accuracy.
- **Two confidence scores for the cascade, compared:**
  - **Raw:** YOLOv8n's mean detection confidence for the frame, as stored by
    spec 01.
  - **Learned:** a predicted probability that the local result is at least
    as accurate as the offload result, learned from all six confidence
    features. It is trained only on frames that are not in the simulation,
    as in spec 02.
- **Offload latency is predicted from network and device conditions** by a
  model trained on training-photo rows. The router never sees a request's
  true latency when it decides.
- **Local latency and all accuracies** are the stored simulated values used
  by specs 01 and 02. No phone-calibrated latencies.
- **Evaluation** uses the held-out photos from the existing frame-level split
  (seed 42), for latency budgets of 100, 150, 200, 300, 400 and 500 ms. For
  each budget and policy, it reports:
  - mean detection accuracy
  - **on-time accuracy:** mean accuracy where any result that arrives after
    the budget counts as 0, because it missed the deadline
  - mean latency
  - the share of requests whose actual latency exceeded the budget
- **Comparisons** at every budget:
  - always-local
  - always-offload
  - a **within-budget oracle** that knows each request's true latencies and
    accuracies and picks the most accurate option that fits the budget
    (local, offload, or local-then-offload), falling back to local when
    nothing fits
- **Threshold handling for the cascade:**
  - The full sweep of confidence thresholds is reported as a curve of
    accuracy against mean latency for each budget.
  - A single operating threshold per budget is chosen on the training photos
    only (the one with the highest on-time accuracy on the training photos)
    and then reported on the held-out photos.
- **Uncertainty:** for each budget, the tuned cascade's on-time accuracy
  minus the budget-only policy's, and minus always-local's, each with a 95%
  bootstrap confidence interval that resamples held-out photos. The cascade
  counts as **useful at a budget** only if its interval against budget-only
  lies entirely above zero.
- **Reproducibility:** all randomness is seeded, so running it twice gives
  identical numbers. The experiment is a runnable script, and it only reads
  from the database.
- **Write-up:** `training/router/findings.md` gains a new section with:
  - the per-budget results tables
  - an accuracy-versus-latency figure (one curve per policy and confidence
    score, plus the baselines and the oracle), saved next to `findings.md`
  - the usefulness verdict at each budget
  - a short related-work note citing the papers above
  - an updated recommendation on the parked phone/server work, which
    replaces spec 02's recommendation

## Out of Scope

- Phone-measured or phone-scaled latencies, including a what-if with
  iPhone CPU latency.
- The soft-utility routers from spec 02, and any change to them or to their
  results.
- Frame rate, battery, bandwidth or energy constraints. Latency is the only
  constraint.
- Stale-result accuracy decay (DeepDecision's video setting). Requests are
  single scans.
- Choosing among more than two models, or between resolutions and bitrates.
- Retraining the detectors, or computing new frame features.
- New simulation runs, or any schema change.
- Deploying the router to the phone or the server.
- Hyperparameter search beyond the threshold sweep.

## Acceptance Criteria

- Given the stored data, when the experiment runs, then for every budget
  (100, 150, 200, 300, 400, 500 ms) it reports mean accuracy, on-time
  accuracy, mean latency and over-budget share for:
  - budget-only
  - the cascade with the raw score
  - the cascade with the learned score
  - always-local
  - always-offload
  - the within-budget oracle
- Given a request whose predicted offload latency exceeds the budget, when
  the budget-only policy decides, then it runs locally.
- Given a cascade request whose local confidence meets the threshold, then
  its local result is kept.
- Given a cascade request whose confidence is below the threshold:
  - when the predicted offload latency fits the remaining budget, then it is
    escalated and charged local plus offload latency with offload accuracy
  - otherwise the local result is kept
- Given any model or threshold used in routing, then no held-out photo's
  rows were used to train or tune it, and the learned confidence score was
  never trained on a simulated frame.
- Given a result that arrives after the budget, when on-time accuracy is
  computed, then that result counts as 0.
- Given the within-budget oracle, when it is computed, then its on-time
  accuracy at every budget is at least that of every policy, and it never
  exceeds the budget where any option fits.
- Given each budget, then the output shows the full threshold sweep for
  both confidence scores, and the tuned operating point chosen on training
  photos alone.
- Given each budget, then the output shows 95% bootstrap intervals for the
  tuned cascade's on-time accuracy against budget-only and against
  always-local, and a usefulness verdict following the rule above.
- Given the same database contents, when the experiment runs twice, then
  every reported number is identical.
- Given the experiment has run, then `findings.md` has the new section, the
  figure exists next to it, and the recommendation on the phone/server work
  reflects this experiment's results.
- Given this feature is complete, then the spec 02 experiment and the
  earlier router scripts still run and produce their previous results.

## Related work (confirmed)

- Kang et al., *Neurosurgeon*, ASPLOS 2017.
- Ran, Chen, Zhu, Liu & Chen, *DeepDecision*, INFOCOM 2018.
- Han et al., *MCDNN*, MobiSys 2016.
- Park et al., *Big/LITTLE deep neural network for ultra low power
  inference*, CODES+ISSS 2015.
- Teerapittayanon, McDanel & Kung, *Distributed Deep Neural Networks over
  the Cloud, the Edge and End Devices*, ICDCS 2017.
- Wang et al., *IDK Cascades*, UAI 2018.
- Chen, Zaharia & Zou, *FrugalGPT*, 2023.
- Taylor et al., *Adaptive Deep Learning Model Selection on Embedded
  Systems*, LCTES 2018.

## Open Questions

None.
