# Rules for adaptive-offload

Senior Seminar semester project — adaptive on-device/server offloading for
real-time object detection on mobile devices. See
`Senior Seminar/Project Technical Notes.md` (Windows side, School repo) for
the full research design; this repo is the implementation.

## Infrastructure — never start or stop it yourself

- Never start, stop, or restart the Postgres instance, the FastAPI server,
  the data-gen simulator, or any other long-running process on your own
  initiative — always ask the user to do it.
- Never run database migrations or create/drop a database yourself; the user
  runs the migration script and manages the database themselves.
- This applies even mid-task: if a task would be easier to verify with
  infrastructure running (e.g. hitting a live endpoint, querying the DB),
  stop and ask the user to start it rather than starting it yourself.

## Git workflow

- Feature branch + PR; never push directly to `main`, even for small fixes.
- An explicit go-ahead to push straight to `main` covers that one commit
  only — it is not standing permission for later commits in the same
  session. Ask again each time.

## Code quality

Follow `.claude/coding-guidelines.md` for repo layout, stack conventions,
and the testing bar. Applies to every line the `implementer` and `reviewer`
agents touch.

## Workflow

Non-trivial work goes through the spec-driven flow in `.claude/commands/`,
not ad hoc edits: `bootstrap` → `generateFeatureSpec` →
`generateImplementationPlan` → `executePlan` (which delegates tasks to the
`implementer` agent) → `applyReview` (using the `reviewer` agent's report).
`refreshDocs` and `submitForPullRequest` cover doc upkeep and PR submission.
The `implementer`/`reviewer` agents are only useful in the context of this
workflow — check `.claude/commands/` before assuming a piece of tooling is
missing.
