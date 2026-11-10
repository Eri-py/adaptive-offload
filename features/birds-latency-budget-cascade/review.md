# Review — Birds Cascade Under a Hard Latency Budget

## Verdict

The negative result is correctly established and, with one exception, honestly
reported. The band-independence argument — the spine of the write-up — is
**exact as stated for the threshold**: in the escalation band every result is on
time, so on-time accuracy collapses to `mean(score < tau ? large : small)`,
which depends on tau alone; outside it no photo escalates (or everything is
late), so tau cannot move any number. The code matches the argument
(`escalation_fits`, `budget_experiment.py:162-167`), the closed-form grid
matches the per-cell path, and I re-derived the headline numbers against the
real records read-only: small 0.7798 / large 0.8699, thresholds 0.8728 and
0.8537, escalation 83.24% / 82.29%, cascade on-time 0.8700, vs budget-only
[-0.0007, +0.0010], vs always-local [+0.0803, +0.0999], fewest escalations
4,184 / 3,978 — every one reproduces the write-up exactly. The cost model, the
band assignments and the 22/2/12/0 cell counts all follow arithmetically from
11.39 KB, 6.731 ms and the three named conditions. The late fix in `1a9af06` is
complete: the report prints both intervals, a test pins that, and the write-up's
reading of the always-local interval (the cascade recovers essentially the whole
0.0901 small-to-large gap, so its failure is redundancy, not accuracy) is sound
and correctly bracketed.

The one blocker is a write-up defect, not a code defect: the four-band table is
presented as a partition and as exact, and it is neither. A cell with
`O ≤ B < L` (offload cheap, local expensive) matches both the "nothing fits" row
and the "offload but no escalation" row, and the consequence column for the
first row is wrong there. The derived claim that "inside a band, nothing depends
on L, O or B" fails in the same corner: I measured the cascade's interval against
budget-only at [-0.0998, -0.0803] for (L=50, O=28.6, B=50) and [-0.8781, -0.8612]
for (L=60, O=28.6, B=50) — same band, different numbers. **No reported cell is
affected** (every marked local cost is ≤ the smallest budget), the code's
classifier resolves the corner correctly, the grid test covers it, and the
figure's fill is right — so this costs two sentence-level edits, not a re-run.
Everything else — the spec's required caveats, the related-work note, the
unanchored-axis statement, determinism, the CLAUDE.md infrastructure rules — is
in order. `290 passed`, `ruff check` clean, `mypy` clean on `birds/` and
`tests/birds/`, all re-run here.

## Acceptance Criteria

1. **All four metrics for all five policies (cascade under each measure).**
   MET — `budget_experiment.py:240-289`, `_policy_table:340-347`,
   `format_report:350-394`; `test_hand_computed_on_time_accuracy_for_all_five_policies`
   (`test_budget_experiment.py:59`); tables in `findings/birds.md:447+` carry six
   rows per cell.
2. **No model loaded, no inference; every accuracy traces to stored records.**
   MET — `load_records` reads through `read_predictions`
   (`budget_experiment.py:100-108`); no `torch` import anywhere under
   `training/birds/router/`; the only data path is the two aligned DataFrames.
3. **Offload cost = server compute + payload/bandwidth + round trip.**
   MET — `conditions.py:64-73`; `test_offload_cost_sums_server_transfer_and_round_trip`
   (`test_conditions.py:76`). Re-derived: 1.82 / 9.11 / 45.56 ms transfer →
   28.6 / 75.8 / 202.3 ms, matching `findings/birds.md:238-242`.
4. **Offload over budget → budget-only runs every request locally.**
   MET — `shared/routing.py:38-46`; pinned indirectly by
   `test_grid_matches_evaluate_cell_in_every_band` (`test_budget_experiment.py:196`),
   where `grid_accuracy`'s `np.where(offload_fits, large, always_local)` is checked
   against `evaluate_cell` in the local-only band.
5. **Confident cascade request keeps local, charged local only.**
   MET — `shared/routing.py:49-67` + `policies.py:37-41`;
   `test_escalated_photo_pays_both_costs_and_takes_large_correctness`
   (`test_policies.py:17`).
6. **Low-confidence request escalates at L+O with large correctness when it fits,
   else keeps local.** MET — `shared/routing.py:63-66`, `policies.py:37-41`;
   `test_no_escalation_when_local_plus_offload_exceeds_budget`
   (`test_budget_experiment.py:76`).
7. **Threshold selected on validation alone.** MET — `tune_accuracy_threshold`
   takes only the validation records (`budget_experiment.py:143-159`);
   `test_threshold_is_tuned_on_validation_records_only`
   (`test_budget_experiment.py:116`) and
   `test_tuned_threshold_is_the_same_in_every_escalation_band_cell:170` both
   vary the *test* records and assert the threshold is unmoved.
8. **Late results count as zero.** MET — `on_time_accuracy_per_row`
   (`shared/routing.py:70-79`), used by every metric path
   (`budget_experiment.py:127`, `234-237`).
9. **Oracle ≥ every other policy in every cell.** MET — `policies.py:44-72`;
   `test_oracle_is_at_least_every_policy_in_all_four_bands`
   (`test_budget_experiment.py:94`), which asserts `seen == set(Band)`. The real
   tables agree (oracle 0.8999 everywhere escalation fits).
10. **Local cost alone over budget → every policy zero.**
    PARTIAL — holds whenever offload also exceeds the budget, which is every
    reported cell, and `test_everything_is_zero_when_local_cost_alone_exceeds_budget`
    (`test_budget_experiment.py:86`) covers exactly that case. It does **not**
    hold when `O ≤ B < L`: I measured always-offload at 0.8699 on-time for
    (L=60, O=28.553, B=50). The code behaves correctly (the criterion itself is
    the imprecise statement), but the write-up repeats the imprecise form — see B1.
11. **Both static baselines over-budget 0 or 1, stated as a structural difference.**
    MET — `test_static_baselines_are_all_on_time_or_all_late`
    (`test_budget_experiment.py:107`); every table row shows 0.00% or 100.00%;
    stated at `findings/birds.md:225-231`.
12. **Bootstrap intervals vs budget-only *and* vs always-local, plus the verdict.**
    MET (after `1a9af06`) — computed at `budget_experiment.py:267-275`, both
    printed at `:371-382`, pinned by `test_report_shows_both_bootstrap_intervals`
    (`test_budget_experiment.py:280`); reported per band for all 36 cells at
    `findings/birds.md:645-670`. I reproduced both intervals exactly.
13. **Budgets: one admits no offload under poor, one admits escalation under good.**
    MET — `validate_budgets` (`budget_experiment.py:292-304`),
    `test_validate_budgets_requires_a_lockout_and_an_escalation_budget:159`;
    stated at `findings/birds.md:349-350`.
14. **Determinism.** MET — bootstrap seeded (`shared/routing.py:148`);
    `test_two_runs_over_unchanged_records_are_equal:126` and the integration test
    comparing two full printed reports (`test_budget_experiment.py:253-261`);
    write-up records a byte-identical diff of two runs (`findings/birds.md:217-219`).
15. **Run recorded in experiment tracking.** MET by code path —
    `tracking.run("budget-experiment", …)` with per-cell metrics
    (`budget_experiment.py:516-525`). I did not inspect the MLflow store (that
    would mean touching infrastructure); the claim rests on the code and the
    write-up's statement at `findings/birds.md:207-211`.
16. **Findings file carries figure, tables, verdicts, boundary statement,
    related-work note, unanchored-axis statement + anchoring measurement.**
    MET — figure `:277`, tables `:447+`, verdicts `:645+`, boundary movement
    `:302-326`, related work `:171-176`, unanchored axis and the Core ML/TFLite
    measurement that would anchor it `:256-274`. One imprecise sentence in the
    figure description — S4.
17. **Figure marks the desktop measurement and the three conditions; no attempt-1
    phone figure.** MET — `boundary_plot.py:121-152`; `conditions.py:33-36`
    documents the prohibition; `test_marks_are_desktop_measurement_and_three_named_conditions_only`
    (`test_boundary_plot.py:155`). I rendered the committed PNG: marks are
    "desktop CPU 5.877 ms" and good/medium/poor only; 17.34 and 106.01 appear
    nowhere in the diff. The test's attempt-1 assertion itself is weak — N1.
18. **Desktop full-precision stand-in caveat, including that the exported
    model's predictions must be measured.** MET — `findings/birds.md:704-713`;
    it names predictions *and* confidences, not just speed.
19. **Offload cost exact, hedging advantage named, noisy offload cost named as
    the next experiment.** MET — `findings/birds.md:714-723`, in those terms.
20. **Four bands, escalation band above offload band, lockout width = L.**
    PARTIAL — the escalation-above-offload argument (`:341-347`) and the lockout
    width (`:349-353`) are exact and correct, and `lockout_width_ms`
    (`policies.py:87-90`) matches. The band table itself is not a partition and
    its first row's consequence is wrong in one corner — B1.
21. **Fewest offload calls to match always-offload, or that it never matches.**
    MET — `fewest_escalations_to_match` (`budget_experiment.py:191-202`);
    `test_fewest_escalations_to_match_always_offload:131`; reported as 4,184
    (72.2%) and 3,978 (68.7%) at `findings/birds.md:685-700`, both reproduced.
    The quantity is read off the test split — N2.
22. **Existing entry points still produce previous numbers; no migration.**
    MET — `findings/birds.md:221-228` records the re-check (`evaluate` reproduced
    0.7798 / 0.8699; `summary` identical apart from the log timestamp); the diff
    touches no file under `database/`, authors no migration, and the full
    `training/` suite passes at 290.

## Scope

The diff matches the plan's Files Changed table exactly, plus
`training/birds/router/latency_budget_boundary.png` (the rendered figure, listed
in the implementation report and precedented by
`training/coco/router/latency_budget_curves.png`) and the workflow artifacts.
No attempt-1 file, no `database/` file, no migration, no `[project.scripts]`
entry, no schema change. `grid_accuracy` and `format_report` live in
`budget_experiment.py` rather than files of their own, which is within the
plan's stated "cell evaluation, threshold tuning, and the entry point" scoping.

CLAUDE.md infrastructure rules are respected: the only DB create/drop is the
pre-existing `postgres_engine` fixture (`training/tests/conftest.py`, unchanged
in this diff), tracking is forced off for the suite by an autouse fixture, and
nothing starts or stops a long-running process. Re-running
`birds.evaluation.predictions` during Task 4's verification does write to the
shared database (delete-and-reinsert per model and split, per `learnings.md`),
but that is a short-lived entry point, not a migration or a create/drop, so it
is inside the rules; it is worth being deliberate about repeating.

## Blockers

#### B1 — The four-band table is presented as a partition and as exact; it is neither

- **File:** `findings/birds.md:330`, `findings/birds.md:335`,
  `findings/birds.md:353-359`, `findings/birds.md:655`
- **Issue:** "exactly one of four bands holds" is false: a cell with `O ≤ B < L`
  satisfies both the "Nothing fits | L > B" row and the "Offload but no
  escalation | O ≤ B < L + O" row, and in that cell the first row's consequence
  ("every policy has 0 on-time accuracy") is wrong — always-offload is on time.
  The derived claim at `:353-359` ("Inside a band, nothing depends on L, O or B
  … the cascade's standing … never on the latency values inside the band") and
  the claim at `:655` ("The interval is identical across all cells of a band")
  fail in the same corner: measured on the real records, the cascade's interval
  against budget-only is [-0.0998, -0.0803] at (L=50, O=28.553, B=50) and
  [-0.8781, -0.8612] at (L=60, O=28.553, B=50) — both "offload but no
  escalation". The code is correct (`classify_band`, `policies.py:75-84`, tests
  the bands in precedence order, and `test_grid_matches_evaluate_cell_in_every_band`
  exercises a `O ≤ B < L` cell), and no reported cell is affected, since every
  marked local cost is at most the smallest budget — which the write-up already
  says at `:338`. Only the stated generality is wrong, and it is stated in the
  section whose whole point is that the result is exact.
- **Fix:** Change the "Nothing fits" condition to `L > B and O > B`. Qualify the
  offload-no-escalation row (and the `:353-359` paragraph and the `:655`
  sentence) with `L ≤ B` — e.g. "inside a band, with L ≤ B, nothing depends on
  L, O or B; when local alone exceeds the budget the cascade is always-local and
  late, so its on-time accuracy is 0". Add one clause noting that all 36
  reported cells have L ≤ B, so no number in the tables changes.
- **Decision:** Accepted — addressed in "Correct the four-band table and qualify band independence"

## Suggestions

#### S1 — `_metrics` re-derives correctness instead of using `outcomes`' second return value

- **File:** `training/birds/router/budget_experiment.py:234-237`
- **Issue:** `_metrics` calls `_policy_rows` (which calls `outcomes` and
  discards its `correct` array) and then recomputes
  `np.where(actions == LOCAL, small, large)` — a verbatim copy of
  `policies.py:40`. If the action-to-correctness mapping ever changes (a third
  model, an ESCALATE that keeps the local answer on a server error), the two
  copies drift silently and `mean_accuracy` stops matching the latencies it is
  reported beside.
- **Fix:** Have `_policy_rows` return `(latency, correct, on_time)` and let
  `_metrics` use the `correct` it already computed.
- **Decision:** Accepted — addressed in "Apply the remaining review findings"

#### S2 — The grid/evaluate agreement test does not assert it covered all four bands

- **File:** `training/tests/birds/router/test_budget_experiment.py:196`
- **Issue:** The test is named `..._in_every_band` and is the guard that keeps
  the closed-form grid from drifting from `evaluate_cell`, but unlike
  `test_oracle_is_at_least_every_policy_in_all_four_bands:94` it never asserts
  which bands its budget/local/offload lists actually produce. Today they cover
  all four (I checked); a future edit to those three lists could quietly drop a
  band — most likely the `O ≤ B < L` corner that B1 is about — and the test
  would still pass under its own name.
- **Fix:** Collect `classify_band(...)` per cell into a set and assert
  `seen == set(Band)`, as the oracle test does.
- **Decision:** Accepted — addressed in "Correct the four-band table and qualify band independence"

#### S3 — "It never matches always-offload" reads oddly against the local-only tables

- **File:** `findings/birds.md:699-702`
- **Issue:** In the 12 local-only cells the tables show always-offload at
  0.0000 on-time and the cascade at 0.7798 — the cascade beats it — yet the
  prose says it "never matches always-offload there". Both are true, because
  `fewest_escalations_to_match` compares *plain* accuracy
  (`budget_experiment.py:191-202`) while the tables report on-time accuracy, but
  the sentence does not say which it means and sits a few hundred lines from the
  tables that appear to contradict it.
- **Fix:** Say "never reaches always-offload's *plain* accuracy there, because
  it cannot escalate at all — though always-offload's on-time accuracy in those
  cells is 0, so the comparison that matters is the one in the tables."
- **Decision:** Accepted — addressed in "Apply the remaining review findings"

#### S4 — The grey region of the figure is described by only half its condition

- **File:** `findings/birds.md:294-297`
- **Issue:** "Grey fills the region where local cost exceeds the budget (dotted
  vertical line): at 50 ms it starts at 50 ms local cost" omits the offload
  condition. Grey requires `L > B` **and** `O > B`; right of the dotted line but
  below the dashed line the fill is blue, which the preceding bullet already
  claims for the wedge. I recomputed the classification over the real sweep: at
  the 50 ms budget, grey occupies local 51-150 **with** offload 55-1000 only,
  and the rendered PNG agrees.
- **Fix:** "Grey fills the region where local cost *and* offload cost both
  exceed the budget — right of the dotted line and above the dashed one."
- **Decision:** Accepted — addressed in "Apply the remaining review findings"

## Nitpicks

#### N1 — The attempt-1-anchors assertion in the figure test cannot fail

- **File:** `training/tests/birds/router/test_boundary_plot.py:163`
- **Issue:** `assert not any("17.34" in t or "106" in t for t in texts)` runs
  against marks the test itself supplies (`_MARKS` = 40/80/150 ms), so it is
  true by construction and would stay true even if `conditions.py` grew a COCO
  phone constant. The real guarantee lives in `boundary_plot.py:41`, which
  imports only `SMALL_DESKTOP_CPU_MS`.
- **Fix:** Assert against the module's own defaults instead — e.g. that
  `BoundaryMarks()` and `conditions.py`'s constants contain neither 17.34 nor
  106.01 — or drop the line as decorative.
- **Decision:** Accepted — addressed in "Apply the remaining review findings"

#### N2 — The "fewest offload calls" figure is read off the test split

- **File:** `findings/birds.md:685-700`
- **Issue:** 4,184 / 3,978 are the minimum prefixes computed on the test records
  (`evaluate_cell` passes `test` at `budget_experiment.py:275`). That is a best
  case you could only pick with hindsight, unlike the threshold, which the
  write-up is careful to say was tuned on validation. The section does say "on
  the test split", but does not flag what that implies.
- **Fix:** Add half a sentence: "this is a hindsight figure read off the test
  split, not an operating point selectable in advance."
- **Decision:** Accepted — addressed in "Apply the remaining review findings"

#### N3 — "As the band argument predicts" covers only half of what it is attached to

- **File:** `findings/birds.md:655-656`
- **Issue:** The band argument predicts that the interval is identical across
  cells of a band; it says nothing about the two confidence measures agreeing.
  They do agree to four decimals (I reproduced both at [-0.0007, +0.0010]), but
  that is an empirical coincidence of two differently-escalated photo sets, not
  a consequence of the argument.
- **Fix:** Split the sentence: identical across cells *by the band argument*;
  identical across measures *as it happens, at four decimals*.
- **Decision:** Accepted — addressed in "Apply the remaining review findings"

#### N4 — `fewest_escalations_to_match` restates `escalation_fits` inline

- **File:** `training/birds/router/budget_experiment.py:199`
- **Issue:** `if cell.local_ms + cell.offload_ms > cell.budget_ms` is the
  negation of `escalation_fits(cell)` (`:162-167`) minus the `L ≤ B` clause —
  equivalent for non-negative costs, but it is the same predicate written twice,
  and the second copy is the one without the documented assumption attached.
- **Fix:** `if not escalation_fits(cell):`.
- **Decision:** Accepted — addressed in "Apply the remaining review findings"

#### N5 — Four-line provenance comments against the guidelines' one-to-two-line rule

- **File:** `training/birds/router/conditions.py:22-25`,
  `training/birds/router/conditions.py:53-56`
- **Issue:** `.claude/coding-guidelines.md` asks for one-line comments, "two
  lines only when truly needed". These are four each. The content is genuine
  provenance (the payload measurement; why the named conditions sit inside
  attempt 1's sampled ranges) and is exactly the sort of thing that should not
  be lost, so this is a placement nit, not a delete-it nit.
- **Fix:** Move the measurement provenance into the module docstring, which
  already carries the server-compute provenance, and leave a one-line comment at
  each constant.
- **Decision:** Declined — the provenance is the point of these comments and the reviewer agrees it must not be lost. Moving it to the module docstring separates each fact from the constant it explains, which is where a reader meets it. Keeping four lines beside the constant is the better trade here, per the reviewer's own recommendation.

## Tests

28 new tests across four files; full `training/` suite 290 passed, none skipped,
in 16.7 s. `ruff check .` clean, `ruff format --check` clean on the touched
paths, `mypy birds tests/birds` clean (39 files). I re-ran all of these.

What the tests cover well: hand-computed metrics for all five policies; the four
bands with an explicit `seen == set(Band)` assertion on the oracle bound; the
tie-break and offload-cheaper-than-local corner in `test_policies.py`; the
closed-form grid against `evaluate_cell`, which is the guard that matters most
given the write-up leans on the grid for every figure; threshold tuning proven
insensitive to the test records from two directions; determinism both at
`evaluate_cell` level and end-to-end against a real ephemeral Postgres
database; and, since `1a9af06`, the report's text contents.

Gaps, none of them blocking at the prototype bar:

- The grid/evaluate test does not assert band coverage (S2).
- The attempt-1-anchors assertion is vacuous (N1).
- `test_report_shows_both_bootstrap_intervals` checks that the interval labels
  appear, not that the printed numbers match the `CascadeResult` fields. Given
  that the bug it was written for was a dropped value, a label check is the
  right 80%; a formatted-number assertion would close it.
- Nothing tests `tracked_metrics` key construction or that `run_on_engine`
  records exactly one run — the integration test runs with tracking off, so the
  tracking path is exercised only as a no-op.
- No test pins the `O ≤ B < L` behaviour of `evaluate_cell` directly (it is
  covered incidentally inside the grid test). A named test asserting
  "always-offload is on time while the cascade is not" would have made B1
  impossible to write.

## Recommended Decisions

- **B1** — Accept — The write-up's own exactness claim is the thing at stake;
  the fix is two conditions and one qualifying clause, changes no number, and
  the corner is already understood (it is written down in `learnings.md`).
- **S1** — Accept — Two lines, removes a literal duplicate of the correctness
  mapping that would drift silently.
- **S2** — Accept — Four lines, and it is the test standing between the figure
  and the per-cell path; its name already promises the assertion.
- **S3** — Accept — One sentence, and it removes an apparent contradiction
  between the prose and the tables in the deliverable.
- **S4** — Accept — One sentence; the current wording describes a region the
  figure does not have.
- **N1** — Accept — Cheap, and the assertion as written gives false assurance
  about a spec requirement.
- **N2** — Accept — Half a sentence, and it keeps the test/validation discipline
  the rest of the section is scrupulous about.
- **N3** — Accept — One clause; "as the argument predicts" is doing work the
  argument does not do.
- **N4** — Accept — One-line change to the documented predicate.
- **N5** — Decline — The guideline is about comments that restate the code;
  these record measurements that would otherwise be unrecoverable. Moving them
  is churn for a prototype, and the docstring is already long.
