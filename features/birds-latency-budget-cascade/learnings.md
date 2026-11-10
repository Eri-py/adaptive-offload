# Learnings — Birds Cascade Under a Hard Latency Budget

Entries added during execution, newest last.

## Task 1 — cost model, policies, cell evaluation

- Payload size and the three named conditions' bandwidth/RTT pairs were not in the task
  inputs. `conditions.py` leaves `payload_kb` and `server_compute_ms` as required arguments
  (no default) and marks the condition values as assumed design choices to confirm.
- The fewest-escalations-to-match search can only return None when escalation does not fit the
  budget: escalating every photo always reproduces always-offload's accuracy.
- `tests/` mypy: `**dict` kwargs into keyword-only typed functions fail type-check; use a small
  local helper function in the test instead.

## Task 1 — three missing inputs, only one of them really missing

The plan listed one open input (the large model's server-compute latency) but
the cost model actually needed three. The implementer refused to invent any of
them, which was right.

- **Payload was measurable all along.** 300 CUB test photos through the models'
  own transforms (`Resize(232)` + `CenterCrop(224)`, verified identical for
  both) at JPEG q=85: mean 11.39 KB, median 10.97, range 3.97-26.71. It is now
  `PAYLOAD_KB`, with that provenance in a comment. Worth noting it is roughly
  half the 20 KB a worked example had assumed, which moves every band boundary
  — at 2 Mbps the transfer term is 45.6 ms, not ~80 ms. Measure before
  assuming, even for a term that looks like a detail.
- **Named network conditions did not exist in the repo.** Attempt 1's
  `coco/datagen/presets.py` samples *ranges* (baseline 0.5-100 Mbps /
  10-400 ms) rather than naming good/medium/poor points, so there was nothing
  to inherit. The three named points are the project's own choice; the comment
  records that they sit inside attempt 1's ranges so the attempts stay
  comparable.
- **Server compute stays a required argument with no default.** A fabricated
  latency constant is precisely the failure that made the desktop stand-in
  misleading in the first place, so the code refuses to proceed without the
  real figure rather than carrying a plausible-looking one.

## Task 1 — two behavioural notes worth remembering

- **"Fewest escalations to match always-offload" returns None only when
  escalation is unaffordable** (local + offload over budget). Otherwise
  escalating every photo reproduces always-offload exactly, so a match always
  exists. The interesting case is therefore the band, not the accuracy.
- **The band classifier handles offload cheaper than local.** A cell can sit in
  "offload but no escalation" while local alone does not fit, so the
  "local over budget zeroes every policy" property assumes offload cost is at
  least local cost. The test uses such a cell deliberately.

## Task 2 — boundary heatmap

- The fill is "cascade vs best deployable policy" (loses / ties / beats, plus a grey "no policy
  on time"), not an argmax: the oracle wins every cell and budget-only equals
  max(always-local, always-offload), so an argmax is flat. Tie tolerance is 0.001 on-time
  accuracy (~6 of 5,794 photos); the real best case beats always-offload by 2 photos.
- Band edges (local = budget, offload = budget, local + offload = budget) are drawn as lines;
  the gap between the last two is the local cost, which is the figure's headline.
- Named conditions' offload positions are passed in as data (they need the unsupplied
  server-compute value); only the desktop 5.877 ms default is imported from `conditions.py`.
- Tests read marks from `fig.axes[0].texts` via `build_boundary_figure`, which returns the
  figure so tests need no pytest fixtures; `tempfile` covers the save test.
- mypy: use `matplotlib.lines.Line2D`, not `plt.Line2D` (not exported).

## Task 2 — the tuned threshold is band-dependent, not cell-dependent

Rendering a boundary grid exposed that `evaluate_cell` costs ~0.2 s, because
`tune_threshold` does a full pass per candidate threshold *inside every cell*.
A full-resolution grid (150 local x 200 offload) would be ~110 minutes per
budget. But the per-cell work is not merely slow, it is unnecessary — and the
reason is exact, not an approximation.

Write the cascade's on-time accuracy as a function of the threshold tau in a
cell with local cost L, offload cost O and budget B:

- **L > B** — kept-local results are late and escalated ones (L+O) are later
  still, so on-time accuracy is 0 for every tau.
- **L <= B, L+O > B** — escalation never fits, so nothing escalates and the
  cascade *is* always-local, for every tau.
- **L <= B, L+O <= B** — kept results arrive at L and escalated at L+O, both
  within budget, so everything is on time and on-time accuracy equals plain
  accuracy: `mean(score < tau ? large_correct : small_correct)`. That depends
  on tau alone — not on L, O or B.

So the tuned threshold is identical across every cell of the third band and
irrelevant in the other two. **Compute it once per confidence measure, not once
per cell.** The grid then costs `candidates x val_photos` plus a band
classification per cell: milliseconds rather than hours.

Two consequences:

- **This is a research statement, not just a speed fix.** The cascade's
  standing relative to the other policies depends only on which band a cell is
  in, never on the latency values within a band. That is what makes the
  boundary map interpretable, and it is why drawing the band edges as lines
  over the fill is the right design.
- **Deferring the question was the wrong call.** Task 1's implementer flagged
  the per-candidate pass and offered to vectorise; the instruction was to wait
  until Task 2 showed whether it mattered. It does. The fix is better than
  vectorising, though — the per-cell work disappears instead of getting faster.
