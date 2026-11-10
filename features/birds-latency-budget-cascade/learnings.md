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
