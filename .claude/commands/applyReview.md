---
description: Triage reviewer findings — accept and implement, or decline with rationale
model: sonnet
---

# Apply Review

Your task is to triage every finding in `review.md` with the user, record each decision inline, and implement the accepted fixes via a fresh `implementer` subagent. The annotated `review.md` is the handoff to `/submitForPullRequest`.

You delegate code changes; you do not write them yourself. You are responsible for orchestration, verification, annotation, and commits.

## Step 1 — Identify the feature

- Ask the user which feature directory to triage (e.g. `features/csv-export/`).
- The current branch must be the feature branch `/executePlan` created for this feature. If it isn't, stop and ask.

## Step 2 — Preflight

1. `git status` must be clean. If not, stop and ask the user to commit or discard first.
2. `review.md` must exist in the feature directory. If not, stop — `/executePlan` did not complete.
3. If `review.md` has no findings at all (Blockers, Suggestions, and Nitpicks are all empty), report this and exit — nothing to triage.

## Step 3 — Read context

Read, for context when implementing fixes:

- `spec.md` — what was originally requested
- `implementation.md` — how it was planned
- `learnings.md` — notes from execution
- `review.md` — all findings

Parse out every finding. Each one has an ID (`B1`, `B2`, ..., `S1`, ..., `N1`, ...) and a `**Decision:**` line whose current value is `— _(pending)_`.

## Step 4 — Offer fast-path

Before entering the per-finding loop, check whether `review.md` contains a **Recommended Decisions** section with a recommendation for every pending finding.

If it does, present the full list of recommendations to the user and ask: **"Apply all recommended decisions?"**

- **Yes — apply all:** treat each recommendation as the user's choice and proceed through the triage loop without prompting per finding. Declined findings still need a rationale — use the reviewer's one-line reason from the Recommended Decisions section.
- **No — go one by one:** fall through to the per-finding loop below, ignoring the recommendations.
- **Partial — adjust first:** show the list and let the user change individual items before proceeding; then apply.

If any pending finding has no recommendation, skip the fast-path entirely and go one by one.

## Step 5 — Triage loop

For each finding whose `Decision:` is still `— _(pending)_`, in order (Blockers first, then Suggestions, then Nitpicks):

1. Show the finding to the user.
2. Ask: **Accept**, **Decline**, or **Skip**.

### If Accepted

1. Build the commit subject: `Address <finding-ID>: <short title from the finding>` (e.g. `Address B1: null handling in detect.py`).
2. Update `review.md`: replace this finding's `— _(pending)_` with `Accepted — addressed in "<commit subject>"`.
3. Delegate the fix to a fresh `implementer` subagent via the `Agent` tool (`subagent_type: "implementer"`). The prompt must include:
   - The finding, verbatim, including its ID
   - The path to the spec file
   - Any learnings from `learnings.md` that look relevant
   - An explicit instruction: this is a fix for one specific reviewer finding; scope is limited to what the finding describes. Do not expand scope.
   - The feature directory path (so the subagent can append to `learnings.md` itself)
4. When the subagent reports back, verify:
   - The project builds
   - Tests relevant to the change pass
   - No tests were skipped
     If any of these fail, tell the user and do not commit. They choose to retry, decline instead, or stop.
5. Stage: the code files the fix changed, plus `review.md` (and `learnings.md` if the subagent appended to it). If `git status` shows changes to any other files, stop and investigate — that is scope drift.
6. Commit with the subject you built in step 1: `git commit -m "<commit subject>"`.

### If Declined

1. Ask the user for a one-line rationale. Do not accept "declined" without one.
2. Update `review.md`: replace this finding's `— _(pending)_` with `Declined — <rationale>`.
3. Do not commit yet — declined annotations are batched in Step 6.

### If Skipped

Leave the `Decision:` field as `— _(pending)_` and move on. The user can run `/applyReview` again later to finish.

## Step 6 — Commit declined annotations

If the triage loop recorded any Declined decisions, commit the pending `review.md` changes:

```
git add <feature-dir>/review.md && git commit -m "Record declined review findings"
```

If nothing was Declined, skip this step.

## Step 7 — Report

Tell the user:

- Findings Accepted (count, with commit subjects)
- Findings Declined (count, with rationales)
- Findings Skipped (count) — remind them to re-run `/applyReview` to finish, and that `/submitForPullRequest` will refuse to open a PR while anything is still pending

## Rules

- Never change a finding's Decision without the user's explicit choice in this session.
- Never delegate a Declined finding to the implementer — only Accept triggers subagent work.
- Never force-push or amend.
- Never `git add -A` or `git add .`.
- If an implementer subagent breaks the build or tests, stop and ask the user how to proceed. Do not commit a broken state.
