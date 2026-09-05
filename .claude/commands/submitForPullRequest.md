---
description: Assemble and open a pull request from a completed feature branch
model: sonnet
---

# Submit Feature Branch for Review

Your task is to assemble a PR description from the artifacts produced by `/executePlan` and open the pull request. The goal is to let the human reviewer trust the work at a glance — by surfacing the acceptance-criteria verdicts, linking the right artifact for each question, and making declined review findings explicit.

You do not merge the PR or resolve reviewer comments. Your job ends once the PR is open.

The development artifacts (`implementation.md`, `progress.md`, `learnings.md`,
`implementation-report.md`, `review.md`) stay on the branch and in the PR — this project's
other readers benefit from seeing how the Claude-driven workflow actually ran, not just the
summarized PR body.

## Step 1 — Identify the feature

Confirm with the user:

- The feature directory (e.g. `features/csv-export/`). Used to locate the artifacts and to build the links in the PR body.
- The base branch for the PR (usually `main`). Do not assume.
- The current branch is the feature branch that `/executePlan` worked on. If it isn't, stop and ask.

## Step 2 — Preflight

1. `git status` must be clean. If not, stop and ask the user to commit or discard first.
2. The feature directory must contain all four review artifacts: `spec.md`, `implementation.md`, `implementation-report.md`, `review.md`. If any are missing, stop — `/executePlan` did not complete.

## Step 3 — Read the artifacts

1. `spec.md` — lift the Acceptance Criteria list verbatim.
2. `review.md` — for each acceptance criterion, find the reviewer's verdict (MET / PARTIAL / NOT MET) and the evidence cited. Also pull out the Blockers, Suggestions, and Nitpicks.
3. `implementation-report.md` — pull the Summary (for the PR summary) and the Tests section (for the PR's Tests section). When drafting the PR summary, rewrite any jargon-heavy sentences: no class names, method names, field names, table names, column names, SQL identifiers, or other hyper-specific technical detail. Describe behaviour and outcomes only.
4. `implementation.md` — skim Approach & Key Decisions for additional context if the report Summary is thin.
5. **Key files** — run `git diff main...HEAD --name-only` to see every changed file. From that list (and from the plan/report), identify the 1–3 files a human reviewer must read to understand what the PR actually does. Prefer production code over tests and generated files. For each chosen file:
   - Get the repo URL: `gh repo view --json url -q .url`
   - Construct a GitHub blob link: `{repoUrl}/blob/{branch}/{relativePath}` — if the meaningful edit is localised within a large file, append `#L{N}` (single line) or `#L{N}-L{M}` (range) for the most important change in that file.
   - Write one sentence explaining why this file is the right place to start.

## Step 4 — Read triage decisions from review.md

Every Blocker, Suggestion, and Nitpick in `review.md` must already have a `**Decision:**` line that is **not** `— _(pending)_`. Parse each one:

- `Accepted — addressed in "<commit subject>"` → this finding goes under **Addressed** in the PR body.
- `Declined — <rationale>` → this finding goes under **Declined** in the PR body.

If any finding is still `— _(pending)_`, stop and tell the user to run `/applyReview` first. Do not open a PR with pending decisions — that defeats the point of the Reviewer-findings section.

## Step 5 — Build and open the PR

1. Draft a short PR title (<70 chars), natural-language, matching the repo's existing commit style. The feature name is usually the right seed.
2. Build the PR body using the template below.
3. Show the proposed title and body to the user. Accept edits before pushing.
4. Push the branch: `git push -u origin <branch>`.
5. Open the PR as a draft, so the user can sanity-check the rendered markdown in GitHub before marking it ready for review:

   ```
   gh pr create --draft --base <base> --title "<title>" --body "$(cat <<'EOF'
   <body>
   EOF
   )"
   ```

6. Print the PR URL.

## PR body template

```markdown
## Summary

[One paragraph describing what the feature does and why it matters, written for a developer who is broadly familiar with the project but has not read any branch artifact. Use plain language: describe behaviour and user-visible outcomes, not implementation details. No class names, method names, field names, table names, column names, SQL identifiers, or other hyper-specific technical jargon — those belong in the Key files section or the artifacts. Seed the content from `implementation-report.md`'s feature summary, but rewrite any jargon-heavy sentences before including them.]

## How to review

Scan the Acceptance Criteria checklist below. If every criterion is ✅ and nothing under "Declined findings" looks wrong, approve. Otherwise drill into the code for the items you are not satisfied with.

## Key files

- [`FileName`](url) — one sentence on why this is the right place to start.
- (1–3 files total; omit this section if there are no non-trivial code changes)

## Acceptance Criteria

- [✅ | ⚠️ | ❌] <AC1 text, expressed in plain behavioural terms — no class names, method names, or file paths>
- [✅ | ⚠️ | ❌] <AC2 text>
- ...

<details>
<summary>Evidence</summary>

- **AC1** — <verdict from review.md + cited file:line or test name>
- **AC2** — <verdict + evidence>
- ...

</details>

## Tests

N tests added · all passing

<details>
<summary>Detail</summary>

- **Unit** — `<file>` (<N> tests): <one-line coverage summary>
- **Integration** — `<file>` (<N> tests): <one-line coverage summary>
- ... (one bullet per test file, repeated for each category as needed)

</details>

## Reviewer findings

<details>
<summary>Addressed (N) · Declined (N)</summary>

**Addressed** (N)
- <finding> — <optional commit subject>
- ...

**Declined** (N)
- <finding> — <one-line rationale>
- ...

(Omit either subheading if it is empty. If the reviewer flagged nothing, write "Reviewer flagged no issues.")

</details>
```

The Tests section headline (`N tests added · all passing`) must be accurate: count every new test method across all test files touched, then confirm all pass. The detail block is lifted straight from `implementation-report.md`'s Tests section. If there were no test changes, omit the section entirely.

When writing Acceptance Criteria bullets, describe observable behaviour only — what the system does or does not do. Never name specific classes, methods, parameters, or file paths in the checklist itself; those belong in the Evidence detail block.

## Rules

- Never push without showing the title and body to the user first.
- Never force-push.
- Never omit a reviewer finding. Every Blocker, Suggestion, and Nitpick in `review.md` must appear in "Reviewer findings" as either Addressed or Declined.
- Default to a draft PR. Drop `--draft` only if the user asks.
- If an AC verdict pulled from `review.md` looks wrong on sanity check (e.g. the cited `file:line` does not exist in the diff), stop and raise the concern before opening the PR.
- AC checklist bullets must describe observable behaviour only — no class names, method names, parameters, or file paths. Move any such specifics to the Evidence detail block.
- All reviewer findings (Addressed and Declined) must be inside the `<details>` block — never above it.
- Test file names, per-file counts, and coverage notes must be inside the `<details>` block. Only the headline totals (`N tests added · all passing`) appear outside it.
