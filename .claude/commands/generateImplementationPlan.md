---
description: Implementation Plan Generator
model: opus
---

# Implementation Plan Generator

Your task is to create an implementation plan from an existing feature spec. The plan has two audiences:

1. A human reviewer who should be able to tell at a glance whether the plan is correct.
2. A sequence of executing subagents, one per task, each starting with fresh context. A separate command orchestrates this.

Each task entry must therefore be self-contained enough that a subagent with only the spec, that task entry, and access to the codebase can execute it correctly.

YOU DO NOT IMPLEMENT THE FEATURE. Your output is `implementation.md` only.

## Step 1 — Identify the spec

Ask the user which spec file to use. Specs live under [features/](/features/).

## Step 2 — Learn the codebase context

Before writing anything, read:

- Any agent-facing architecture/structure reference the project maintains (e.g. `PROJECT_OUTLINE.md`, `ARCHITECTURE.md`) — skip if the project has none
- Any other reference docs relevant to the feature area
- Source files in the areas the feature touches
- The spec's UI mockup (`ui-mocks/<feature-name>.html`), if one exists, so the plan's tasks match what was shown to the reviewer

The project's instruction file (`CLAUDE.md` / `AGENTS.md`) is already in context; apply its constraints when designing the plan, and reference the specific rule in the plan wherever it shaped a decision.

## Step 3 — Resolve ambiguities before writing

If the spec leaves anything unclear that would block a complete plan — which component owns the change, where a hook slots into an existing pipeline, what data shape is assumed, whether a CLAUDE.md constraint conflicts with a requirement — ask the user before drafting. Do not write the plan until either every section can be filled, or the user explicitly defers a question (record those under "Open Questions").

## Step 4 — Write the plan file

Create `implementation.md` in the same subdirectory as the spec, using the template below exactly.

### Plan file template

```markdown
# [Feature Name] — Implementation Plan

## Summary

[2–3 sentences restating what's being built, for the reviewer.]

## Approach & Key Decisions

[3–5 bullets covering the non-obvious decisions this plan commits to. This is the at-a-glance review section — a reviewer should be able to reject a wrong approach from this alone. Cover things like: where in the pipeline the change lives, what's reused vs. new, which layer owns the logic, what was deliberately NOT done. Reference `CLAUDE.md` / `PROJECT_OUTLINE.md` sections where they shape a decision.]

## Out of Scope

[Carried forward from the spec, plus anything this plan further defers. Lets the reviewer confirm scope has not drifted between spec and plan.]

## Dependencies and Configuration

[New packages, config entries (`appsettings.json`, `usersettings.json`), firewall/port changes, DB migrations. Write "none" if there are none.]

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `path/to/file.py` | add / edit / delete | One-line purpose | One-line reason |

[Every file the plan will touch belongs in this table. Prefer new files over repurposing existing ones unless the existing file is clearly the right home.]

## Tasks

[Numbered, ordered tasks the executor will perform in sequence. Each task is executed by a fresh subagent — include the context that subagent needs (which existing types to use, where the new code plugs in, what success looks like) so the task is executable without reading the rest of the plan. Do not restate the overall approach in every task; do state the local context each task depends on.]

### Task 1 — [Short name]

- **Objective:** [One sentence.]
- **Files:** [Paths from the Files Changed table that this task touches.]
- **Details:** [What the task produces. Name any interface or base class to implement, if applicable.]
- **Success criteria:**
  - [Observable outcome]
  - [Test that must pass]

### Task 2 — [Short name]

...

### Task N — Regression test run

- **Objective:** Run every test that exercises code added or modified in this plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - All relevant tests pass

## Open Questions

[Anything the user chose to defer. Omit the section if none.]
```

## Rules for plan content

- Default to the simplest plan that satisfies the spec — do not invent additional robustness, abstractions, extensibility, or configuration unless the spec or the user asked for it.
- Keep the plan as short as possible while still covering approach, scope, files, ordered tasks, and success criteria. The plan is a review artifact, not a training doc for new joiners.
- Put the reasoning in "Approach & Key Decisions". Task entries should be as short as possible while still being self-contained for a fresh subagent — context needed to execute the task belongs in the task; justification for the overall approach does not.
- Never expand scope beyond the spec. If a constraint in the project's instruction file or architecture reference would force scope expansion, raise it as an Open Question in Step 3 rather than silently widening the plan.
- If the spec has a UI mockup, the Files Changed table and tasks should account for building toward it — don't silently deviate from what the reviewer already signed off on.
- Do not include time estimates.
- Do not include execution-agent protocol — progress tracking, build/test cadence between tasks, learnings capture, or post-run reports all live in the execute command, not here. The plan only defines *what* to build and the order; *how the executor runs* is out of scope.

### Task-design rules

- Tasks run sequentially. Each task must leave the project in a building, test-passing state.
- Tests written in a task only exercise code implemented in that task or an earlier one.
- Add an integration-test task at each point where component boundaries have just been wired together.
- The final task is always the regression run described in the template.
