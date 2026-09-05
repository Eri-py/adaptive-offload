---
description: Execute Implementation Plan
model: opus
---

# Execute Implementation Plan

Your task is to execute an implementation plan by delegating each task to a fresh subagent in sequence, tracking progress, and keeping the project in a healthy state between tasks.

You DELEGATE tasks; you do not implement them yourself. You ARE responsible for orchestration, verification, progress tracking, and commits.

## Step 1 — Identify the plan

Ask the user which `implementation.md` to execute. Plans live alongside their spec under [features/](/features/). If the feature was split into numbered sub-plans (`01-.../`, `02-.../`), confirm which sub-plan the user means — run one sub-plan per invocation.

## Step 2 — Preflight

Before starting any task:

1. Ask the user which base branch to branch from. Do not assume `main` or `master`.
2. Confirm the working tree is clean (`git status`). If not, stop and ask the user to commit or stash first.
3. Ask the user what branch-naming convention this repo uses (check recent branch names or the project's instruction file if unsure), and propose a branch name that follows it, seeded from the plan directory name (e.g. a plan in `features/csv-export/` → `feature/csv-export`, or the project's own prefix in place of `feature/`). For nested sub-plans, include the parent directory (e.g. `feature/csv-export-01-download-button`). Confirm the name with the user before creating.
4. If the branch already exists:
   - If the user confirms they are resuming, check it out (`git checkout <branch>`) and continue.
   - Otherwise, stop and ask them to pick a different name.
5. If the branch does not exist, create it: `git checkout -b <branch> <base>`.
6. Ensure `progress.md` exists in the plan directory. If missing, create it with one entry per task from the plan's Tasks section. Suggested per-task format:

   ```markdown
   ## Task N — <name>
   - Status: not started
   - Started: —
   - Completed: —
   - Notes: —
   ```

7. Ensure `learnings.md` exists in the plan directory. If missing, create it empty. If the plan lives inside a parent feature directory (multi-spec feature), also locate sibling sub-plan `learnings.md` files so they can be consulted during delegation.
8. Commit the implementation plan. `spec.md` is already on the base branch (it went through its own PR), but `implementation.md` was written to the working tree by `/generateImplementationPlan` and is untracked. Stage it and commit with message `"Add implementation plan"`. This puts the plan on the branch as the first commit so it's traceable alongside the code that executes it. Do not stage `progress.md` or `learnings.md` here — they piggyback onto task commits in Step 4.

## Step 3 — Build the task list

Read the plan's Tasks section and build a to-do list. Tasks run one at a time, in order. Never skip a task. Never run tasks in parallel.

## Step 4 — Per-task loop

For each task, in order:

1. **Check status in `progress.md`:**
   - **Not started** — proceed normally.
   - **In progress** — the previous attempt was interrupted. Examine the working tree, prior commits, and the Notes field to understand what was already done, then continue from there. Do not restart from scratch. You may undo or adjust prior partial work if that produces a cleaner attempt, but do not silently discard it.
   - **Completed** — skip to the next task.
   - **Failed** — stop and report. Do not proceed without user instruction.
2. Verify the previous task was fully successful — all its success criteria met, project building, tests passing. If not, stop and report.
3. Mark the task `in progress` and record a start timestamp. Get the timestamp from `date '+%Y-%m-%d %H:%M:%S'`, not by estimation.
4. Read `learnings.md` (and any parent/sibling `learnings.md` files from Step 2.7). Collect entries relevant to this task.
5. Delegate the task to a fresh `implementer` subagent via the `Agent` tool (`subagent_type: "implementer"`). The prompt must include:
   - The task entry from the plan, copied verbatim
   - The path to the spec file
   - Any relevant learnings collected in the previous step
   - The feature directory path (so the subagent can append to `learnings.md` itself)
6. When the subagent reports back, verify:
   - The project builds
   - All tests the task introduced or touches pass
   - No tests were skipped
   If any of these fail, mark the task `failed` in `progress.md` with a note on what went wrong, and stop. Report to the user.
7. Append any new learnings from the subagent's report to `learnings.md`. If the subagent reported something worth capturing but didn't log it themselves, add it yourself.
8. Stage the files listed in this task's `Files` entry in the plan, plus `progress.md` and `learnings.md` if they were modified. If `git status` shows changes to any files outside those, stop and investigate — that indicates scope drift. If nothing is staged (e.g. a regression-test task that changed nothing), skip commit and move on. Otherwise commit with `git commit -m "Task <N>: <task name>"`.
9. Mark the task `completed` with an end timestamp.

Continue until every task is done. Do not stop between tasks unless blocked. If blocked, report the issue and wait for user instruction.

## Step 5 — Write the implementation report

After the final task completes successfully, write `implementation-report.md` in the feature directory. This is a neutral, descriptive record of what the plan produced — no critique, no suggestions (the reviewer in Step 6 handles judgement).

The report must include:

- **Feature:** name (from the plan) and feature directory path.
- **Tasks:** each task with its outcome (completed / skipped), taken from `progress.md`.
- **Files changed:** every file added, edited, or deleted on this branch since it forked, grouped by action. Derive from `git diff --name-status <base>...<feature>`, and exclude workflow artifacts (`progress.md`, `learnings.md`, `implementation-report.md`, `review.md`).
- **Tests:** test files added or modified, categorised as **Unit** or **Integration** based on the repo's conventions (test project suffix, filename, etc.). For each: filename, test count, and a one-line description of what it covers. End the section with a single line confirming all tests pass (guaranteed by the per-task build/test gate in Step 4).
- **Commits:** the commit range on the feature branch (e.g. `<base>..HEAD`).
- **Notable events:** anything during execution that required resume-from-partial handling or a stop-and-report. Omit the section if there were none.

Commit the report: `git add <feature-dir>/implementation-report.md && git commit -m "Add implementation report"`.

## Step 6 — Review

Once the report is committed, delegate to a fresh `reviewer` subagent via the `Agent` tool (`subagent_type: "reviewer"`). The prompt must include:

- The feature directory path
- The feature branch name
- The base branch the feature was cut from

The reviewer writes `review.md` in the feature directory and makes no code changes. Do not skip this step, even if all tasks reported success — the reviewer is the independent check on that.

Once the reviewer returns, commit `review.md` to the feature branch: `git add <feature-dir>/review.md && git commit -m "Add review report"`.

## Step 7 — Report to user

1. Summarise: which tasks ran, the commit range on the feature branch, and the reviewer's verdict (blocker count, any acceptance criteria marked partial or not met).
2. Point the user at `implementation-report.md` (what was done) and `review.md` (what the reviewer flagged).
3. Remind the user that the reviewer's suggestions are advisory — they choose which to act on.

## Rules

- Never skip a task.
- Never run tasks in parallel.
- Never continue after a failed build, a failing test, or a skipped test.
- Never `--amend`, `reset --hard`, or delete branches without explicit user consent.
- Record all timestamps from a bash `date` command, never by estimation.
- If blocked, stop and report. Do not improvise a workaround.
