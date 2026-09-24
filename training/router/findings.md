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
| Logistic classifier | ~0.68 (comparable, no clear win) |
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
- **Four routers**, one per design × model family: `decide_first_linear`,
  `decide_first_gbt`, `cascade_linear`, `cascade_gbt`. Decide-first picks
  LOCAL/OFFLOAD before any inference; the cascade always runs local first and
  decides whether to also escalate to offload, charged local+offload latency
  when it does.
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

| router | avg util | local | offload | oracle | headroom | 95% CI | useful |
|---|---|---|---|---|---|---|---|
| decide_first_linear | 0.6575 | 0.6022 | 0.6812 | 0.7208 | -59.95% | [-0.0471, -0.0024] | False |
| decide_first_gbt | 0.6459 | 0.6022 | 0.6812 | 0.7208 | -89.15% | [-0.0661, -0.0085] | False |
| cascade_linear | 0.6289 | 0.6022 | 0.6812 | 0.7208 | -132.12% | [-0.0898, -0.0169] | False |
| cascade_gbt | 0.6282 | 0.6022 | 0.6812 | 0.7208 | -133.86% | [-0.0868, -0.0214] | False |

Held-out split: 400 train photos / 100 test photos (`router.dataset
.frame_level_split`, seed 42).

**None of the four routers meets the usefulness bar.** All four 95%
bootstrap CIs for (router utility − always-offload utility) lie entirely
*below* zero — this is a stronger, more negative result than the three
baseline routers above, which were statistical ties with always-offload.
Every one of these four routers is worse than always-offload with 95%
confidence, on held-out photos the frame features gave it real (if weak)
signal about. `decide_first_linear` is the least bad of the four; both
`cascade` variants and `decide_first_gbt` do markedly worse, consistent with
stage 2 overfitting a 400-photo training set rather than finding real
structure.

The `oracle` column above is the same unconstrained per-row max of local and
offload utility used throughout this file (0.7208), which the spec directs
both designs to be compared against. The cascade design's own ceiling — the
best any ACCEPT/ESCALATE assignment could do once escalated rows are charged
local **and** offload latency — is strictly below that shown oracle, since an
accepted row can never earn the (uncharged) offload latency the oracle
implicitly assumes when it prefers offload. The cascade rows' -132%/-134%
headroom-share figures are measured against a ceiling neither cascade router
could reach even with perfect decisions; they should not be read as "still
this far from a reachable 0%."

### Feature compute cost (from spec 01)

Image features: 5.3 ms/frame. Confidence features: 20.5 ms/frame, almost all
of it the YOLOv8n inference pass itself, not the statistics computed from its
output. Both measured on desktop CPU. Neither is free, and the confidence
features — the ones actually carrying signal — cost roughly 4x the image
features specifically because they require running the local model, which a
decide-first router is trying to avoid running at all.

### Recommendation

Do not resume the parked phone/server work
(`features/server-served-benchmark/`) yet. All four routers this experiment
built — spanning both a decide-first and a cascade design, a linear and a
gradient-boosted model family, and access to every stored frame feature —
are worse than always-offload with 95% confidence, not merely tied with it.
The oracle headroom above always-offload is real (~0.0396 utility, matching
the earlier section), but nothing built here shows a way to capture any of
it reliably from 400 training photos and 100 held-out test photos. Building
phone-side feature extraction and a server-side serving path only pays off
once there is a router worth deploying, and none of these four qualify.

If the router idea is revisited before touching phone/server infrastructure,
the more promising next steps are on the data/modeling side, not the
infrastructure side: more than 400 train photos for stage 2 (a small set for
a classifier combining a prior probability with four network/device
features), and possibly dropping the image statistics entirely in favor of
the confidence features alone, since the diagnostics above show almost all
of the frame-level signal comes from running the local model, not from its
raw image statistics — which also undercuts the decide-first design's
premise of avoiding that inference pass.

If a future rerun does clear the usefulness bar — particularly for the
cascade, whose local-first design is the one that would actually save
offload traffic — the concrete next measurement for
`features/server-served-benchmark/` is real on-phone latency for the
local-first path. Every number in this experiment (like the earlier
baselines) uses the stored desktop latencies; the cascade's premise is
skipping the network round trip on accepted frames, and that premise's value
depends on the local model's actual on-phone inference time, which nothing
in this repo has measured yet.
