# Router findings — direct classifier / non-linear classifier / utility regression

Three router designs were trained and evaluated on real `simulation_results` data
(100,000 rows / 500 unique photos across the 4 presets, frame-level train/test
split so no photo's rows cross the split): a scaled logistic regression on
label, a `HistGradientBoostingClassifier` on label, and a linear regression per
utility target (local/offload latency + accuracy) combined into a predicted
utility per path. All three trained and evaluated on the same 5 input features:
`network_bandwidth_mbps`, `network_latency_ms`, `network_packet_loss_pct`,
`device_load_pct`, `scene_complexity`.

## Result: none of them beat "always offload"

| Approach | Avg utility (held-out) |
|---|---|
| Always offload (static baseline) | 0.6812 |
| Logistic classifier | 0.6574 (comparable, no clear win) |
| HistGB non-linear classifier | did not outperform logistic |
| Linear utility regression | 0.6811 — a statistical tie with always-offload |

The utility-regression router picked LOCAL on only 228 of 20,000 held-out rows
(1.1%), all inside the `degraded-network-idle-device` preset, and even there its
average utility (0.662952) was marginally *below* always-offload's (0.663559) —
135 wins vs. 93 losses that roughly cancel out.

## The task itself has real headroom — the models just aren't capturing it

Computed an oracle ceiling: per row, whichever of local/offload has the higher
*real* (measured, not predicted) utility. Result on the same held-out set:

- Always-offload avg utility: 0.6812
- **Oracle avg utility: 0.7208** — headroom of 0.0396 (~5.8% relative)
- Router avg utility: 0.6811 — captured **-0.4%** of that headroom (noise)
- Oracle picks LOCAL on 46.5% of rows — not a rare edge case

Critically, the oracle prefers LOCAL a lot even in `baseline` (2213/5000) and
`device-stress` (1986/5000) — presets where the router never once picked LOCAL,
and where fast network + idle device would naively suggest offload should
dominate. So the correct decision is not mostly a function of macro network/
device conditions, which is the only kind of signal these 5 features give a
per-condition-vector router.

## Root cause: `scene_complexity` carries no signal about the margin

Pearson correlation between each feature and the real per-row margin
(`local_utility - offload_utility`), computed on the full dataset (not just the
held-out split — this is a feature diagnostic, not a trained model, so there's
no leakage concern):

| Feature | Correlation with margin |
|---|---|
| `network_latency_ms` | 0.172 |
| `network_packet_loss_pct` | 0.108 |
| `network_bandwidth_mbps` | -0.105 |
| `device_load_pct` | -0.091 |
| `scene_complexity` | **0.002** |

The network/device features carry weak-but-real signal (consistent with the
router occasionally firing under bad-network/idle-device conditions). But
`scene_complexity` — the only *frame-specific* feature available — is
statistically uncorrelated with which path wins for a given frame.

Since real per-frame accuracy is a fixed property of (model, frame) that
doesn't depend on network/device conditions (by this project's own design —
see `CLAUDE.md`'s reference to the research notes), and network/device
conditions are shared across every frame sampled under one condition vector,
`scene_complexity` was the router's only lever for frame-specific decisions.
It's dead weight. This is why oracle disagrees with always-offload across
*every* preset (not just the stressed ones) while the router only ever fires
in one: the router literally cannot see the thing that actually predicts a
frame's outcome.

## Conclusion

This is a feature-starvation problem, not a model-capacity or wrong-objective
problem — all three model designs (linear, non-linear, and a different
training objective) hit the same wall because they share the same 5 inputs.
No amount of retraining on these features would be expected to close the gap
to the oracle ceiling. The real headroom (~5.8% relative utility) is genuine
and worth pursuing, but the next step is better frame-level features, not
another model architecture on the same ones.

## Feature-router experiment (spec 01 image/confidence features)

`python -m router.feature_experiment` tests whether the frame-level features
computed in spec 01 — five image statistics plus scene complexity, and six
local-model confidence/detection statistics — close the headroom the section
above identified. Two runs on the real database produced byte-identical
output.

This experiment differs from the three baseline routers above in shape, not
just inputs, so the numbers aren't directly comparable row-for-row:

- **Two-stage, to use all 5,000 frames without leaking the 500 simulated
  photos.** Stage 1 is trained only on the ~4,500 frames that never appear in
  `simulation_results`, and predicts a per-frame accuracy gap and a
  local-good-enough probability from frame features alone. Stage 2 is a
  classifier trained on the existing frame-level split's 400 train photos,
  combining stage 1's two outputs with the network/device features; it is
  evaluated on the split's 100 held-out test photos.
- **Eight routers**, one per design × model family × stage-2 objective:
  `decide_first_linear`, `decide_first_gbt`, `cascade_linear`, `cascade_gbt`
  (stage 2 is an unweighted 0/1 classifier on the stored/cascade label — the
  original four), plus the same four design/family combinations again with
  `_costaware` appended (stage 2 instead regresses the per-row utility
  margin, `local_utility − alternative_utility`, and picks LOCAL/ACCEPT where
  the predicted margin is positive — discussed after the results table
  below). Decide-first picks LOCAL/OFFLOAD before any inference; the cascade always
  runs local first and decides whether to also escalate to offload, charged
  local+offload latency when it does.
- Evaluation is on the held-out photos' simulated rows only (100 photos, not
  the earlier baselines' 500), with a bootstrap CI over those held-out photos.

### Feature diagnostics (Spearman rho vs. offload − local gap, all 5,000 frames)

| feature | spearman_rho | p_value |
|---|---|---|
| mean_confidence | -0.223945 | 7.233081e-58 |
| min_confidence | -0.195415 | 3.182400e-44 |
| min_box_area | -0.146368 | 2.395080e-25 |
| detection_count | 0.138978 | 5.466211e-23 |
| max_confidence | -0.125996 | 3.797722e-19 |
| mean_box_area | -0.114944 | 3.557516e-16 |
| entropy | 0.067329 | 1.887492e-06 |
| contrast | 0.057111 | 5.330019e-05 |
| sharpness | 0.042509 | 2.642907e-03 |
| scene_complexity | 0.038575 | 6.372150e-03 |
| brightness | -0.034129 | 1.580647e-02 |
| colorfulness | -0.008962 | 5.263734e-01 |

The six confidence features (need a local inference pass to compute) carry
the strongest signal — `mean_confidence` and `min_confidence` lead at
|rho| ≈ 0.20–0.22, both p ≪ 0.001. The five image statistics that don't
require running any model are much weaker: `scene_complexity` is significant
(rho = 0.038575, p = 0.0064) but tiny, and `colorfulness` is not
distinguishable from zero (p = 0.53). This matches the standalone quick check
that motivated this experiment (image-only AUC ≈ 0.56, confidence-feature AUC
≈ 0.71): the frame-level signal exists mostly in what the local model itself
reports about a frame, not in the frame's raw image statistics.

### Router results (100 held-out test photos)

| router | avg util | local | offload | oracle | headroom | 95% CI | useful | casc ceiling |
|---|---|---|---|---|---|---|---|---|
| decide_first_linear | 0.6575 | 0.6022 | 0.6812 | 0.7208 | -59.95% | [-0.0471, -0.0024] | False | - |
| decide_first_gbt | 0.6555 | 0.6022 | 0.6812 | 0.7208 | -64.94% | [-0.0563, -0.0003] | False | - |
| cascade_linear | 0.6289 | 0.6022 | 0.6812 | 0.7208 | -132.12% | [-0.0898, -0.0169] | False | 0.7091 |
| cascade_gbt | 0.6330 | 0.6022 | 0.6812 | 0.7208 | -121.93% | [-0.0832, -0.0169] | False | 0.7091 |
| decide_first_linear_costaware | 0.6810 | 0.6022 | 0.6812 | 0.7208 | -0.41% | [-0.0020, 0.0014] | False | - |
| decide_first_gbt_costaware | 0.6672 | 0.6022 | 0.6812 | 0.7208 | -35.30% | [-0.0334, 0.0017] | False | - |
| cascade_linear_costaware | 0.6611 | 0.6022 | 0.6812 | 0.7208 | -50.89% | [-0.0302, -0.0104] | False | 0.7091 |
| cascade_gbt_costaware | 0.6549 | 0.6022 | 0.6812 | 0.7208 | -66.42% | [-0.0469, -0.0046] | False | 0.7091 |

`casc ceiling` (cascade rows only, added for review finding S2): the cascade
design's own ceiling, `max(local_utility, escalated_utility)` per row
(`router.evaluation.cascade_ceiling_utility`), averaged over the held-out
rows — identical across all four cascade routers since it depends only on
the held-out rows, not on any router's picks.

Held-out split: 400 train photos / 100 test photos (`router.dataset
.frame_level_split`, seed 42). Two runs of `python -m router.feature_experiment`
on the real database produced byte-identical output for all eight rows,
including the `casc ceiling` column and the always-escalate figure below.

**The classifier objective's negative result is mostly a cost-blind
threshold, not feature starvation or overfitting — except for GBT, which
does overfit.** Stage 2's classifier is fit on 0/1 labels and picks
LOCAL/ACCEPT wherever its predicted probability exceeds 0.5. But the two
kinds of mistake it can make cost very different amounts: on the train
photos, a correct LOCAL pick gains 0.088 utility on average, while a wrong
LOCAL pick loses 0.220 — so the break-even predicted probability is about
0.71, not 0.5. A classifier thresholded at 0.5 picks LOCAL far more often
than that asymmetry justifies. `decide_first_linear`'s 0.6575 is not a
*stronger*, more negative result than the three baseline routers above — it
reproduces `router.baseline`'s logistic classifier almost exactly (0.6574,
corrected above), the same classifier-loss artifact under a different
feature set.

**Cost-aware stage 2 (margin regression) changes the linear verdict.**
Retraining stage 2 as a regressor on the per-row utility margin
(`local_utility − alternative_utility`), picking LOCAL/ACCEPT wherever the
predicted margin is positive, raises `decide_first_linear` to
`decide_first_linear_costaware`'s 0.6810 with 95% CI [-0.0020, 0.0014] — a
statistical tie with always-offload, not a loss. `cascade_linear_costaware`
also improves, to 0.6611, but its CI ([-0.0302, -0.0104]) stays entirely
below zero: cost-awareness helps the cascade design without clearing the
bar. Both GBT cost-aware variants improve over their classifier counterparts
too (`decide_first_gbt_costaware` 0.6672, `cascade_gbt_costaware` 0.6549)
but remain significantly negative — `decide_first_gbt_costaware`'s CI
([-0.0334, 0.0017]) now brushes zero at its upper end, but still spans it
rather than clearing it, so it is not useful either.

**The overfitting claim only holds for the GBT variants, and its cause was
`predicted_gap` acting as a photo ID — fixed by constraining stage 2's leaf
size.** Linear stage 2's train/test accuracy is close (0.570/0.560 for
decide-first, 0.644/0.604 for cascade) — not overfitting, so the linear
family needed no change. GBT stage 2 originally showed a large train/test
gap — 0.982/0.555 (decide-first) and 0.992/0.526 (cascade) — because
`predicted_gap` (stage 1's own regression output, one of stage 2's five
inputs) takes ~400 distinct values across the 400 train photos, essentially
one per photo: an unconstrained `HistGradientBoostingClassifier`/
`HistGradientBoostingRegressor` can split on that near-unique value per
training row instead of learning a generalizable relationship, memorizing
which photo each row belongs to rather than the actual gap.

**Fix (S1):** every GBT stage-2 model (the classifier and the cost-aware
regressor) now fits with `min_samples_leaf` set to `STAGE2_MIN_LEAF_PHOTOS`
(20) photos' worth of rows, computed from the training data itself (`len(df_train)
/ df_train["frame_id"].nunique()` rows per photo, × 20) rather than a
hard-coded row count — 4,000 rows on this dataset's 200 rows/photo. A leaf
that must hold 20 photos' rows can never isolate one photo's `predicted_gap`
value, so the near-unique-per-photo split that caused the memorization is no
longer available. The linear family is untouched (`training/router/two_stage.py`).
After the fix, GBT stage 2's train/test accuracy is 0.8157/0.5847
(decide-first) and 0.8394/0.5594 (cascade) — the train/test gap roughly
halves (0.427 → 0.231 decide-first, 0.465 → 0.280 cascade) instead of
closing entirely, because a 4,000-row leaf can still span ~20 neighboring
photos and pick up some real signal from `predicted_gap`'s ordering (which
does correlate with the true label, since stage 1 is trained to predict the
gap) rather than memorizing any single photo. The router utility numbers in
the table above are the post-fix numbers; no GBT variant's verdict changes
(all four remain `useful=False`), but every GBT row's CI narrows and shifts
toward zero.

The `oracle` column above is the same unconstrained per-row max of local and
offload utility used throughout this file (0.7208), which the spec directs
both designs to be compared against — including the cascade rows, so their
`headroom` percentages are measured against 0.7208, not against the lower
number below. The cascade design's own ceiling — the best any
ACCEPT/ESCALATE assignment could do, `max(local_utility, escalated_utility)`
per row — is **0.7091** (`casc ceiling` column above). That is below the
0.7208 oracle but *above* always-offload's 0.6812: a cascade router has real
headroom of about **0.028** over always-offload, which the oracle-relative
`headroom` percentages in the table can't show. For reference, always-escalate
(every row ESCALATEs) averages **0.6591** — below always-offload, not above
it. The reason the cascade ceiling sits below the oracle is simply that
ESCALATE is charged both the local pass's and the offload pass's latency,
while the oracle's OFFLOAD alternative is only ever charged the offload
pass's: escalating adds the local pass's latency (0.6812 − 0.6591 = 0.0221
utility, ≈74 ms at `DEFAULT_LAMBDA = 0.3`) to every row, which a cascade
router can never avoid paying once it decides to escalate.

### Feature compute cost (from spec 01)

Image features: 5.3 ms/frame. Confidence features: 20.5 ms/frame, almost all
of it the YOLOv8n inference pass itself, not the statistics computed from its
output. Both measured on desktop CPU. Neither is free, and the confidence
features — the ones actually carrying signal — cost roughly 4x the image
features specifically because they require running the local model, which a
decide-first router is trying to avoid running at all.

### Recommendation

Do not resume the parked phone/server work
(`features/server-served-benchmark/`) yet. Of the eight routers this
experiment built — spanning both a decide-first and a cascade design, a
linear and a gradient-boosted model family, and both a classifier and a
cost-aware stage-2 objective — `decide_first_linear_costaware` and
`decide_first_gbt_costaware` reach a statistical tie with always-offload
(their CIs straddle zero); the other six, including every cascade variant,
remain worse with 95% confidence. A tie is not a router worth deploying
either. The oracle headroom above always-offload is real
(~0.0396 utility, matching the earlier section), but nothing built here
shows a way to capture it reliably. Building phone-side feature extraction
and a server-side serving path only pays off once there is a router that
beats always-offload, and none of these eight do.

If the router idea is revisited before touching phone/server infrastructure,
cost-aware stage 2 (margin regression, not more train photos) was the first
candidate next step, and this experiment already tried it — see the
`_costaware` rows above. It closed the gap for `decide_first_linear` (a tie)
and, once the GBT memorization fix (S1) landed, also brought
`decide_first_gbt_costaware`'s CI to straddle zero (a tie, not a win) — but
not for the cascade design under either family. One next step remains
untried: possibly dropping the image statistics entirely in favor of the
confidence features alone, since the diagnostics above show almost all of
the frame-level signal comes from running the local model, not from its raw
image statistics — which also undercuts the decide-first design's premise
of avoiding that inference pass.

If a future rerun does clear the usefulness bar — particularly for the
cascade, whose local-first design is the one that would actually save
offload traffic — the concrete next measurement for
`features/server-served-benchmark/` is real on-phone latency for the
local-first path. Every number in this experiment (like the earlier
baselines) uses the stored desktop latencies; the cascade's premise is
skipping the network round trip on accepted frames, and that premise's value
depends on the local model's actual on-phone inference time, which nothing
in this repo has measured yet.

**Superseded by spec 03.** `features/router-frame-features/03-latency-budget-router`
reframes the router's objective away from the soft weighted-utility score
this whole file uses (under which "always offload" is nearly unbeatable
because latency is cheap) toward a hard per-request latency budget with
accuracy maximized inside it. That reframing supersedes the recommendation
above, not just its next-steps list — the "does any router beat
always-offload on this utility" question this file answers stops being the
relevant question once the objective changes.
