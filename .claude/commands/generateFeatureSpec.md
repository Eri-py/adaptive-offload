---
description: Feature Spec Generator
model: opus
---

# Feature Spec Generator

Your task is to create a feature specification file from a user's feature request. This spec will be reviewed by a colleague before implementation, so it must be unambiguous and complete.

YOU DO NOT IMPLEMENT THE USER'S REQUEST. Only create the required spec file (and, if applicable, a UI mockup — see Step 5).

## Step 1 — Understand the request

Ask the user to describe the new feature or unit of work that needs to be implemented, or, ask if there is a Github Issue to use. If there is a Github Issue, fetch the issue description to learn what is needed.

This unit of work should be a small, deliverable chunk, not a large epic. If it seems too big, challenge the user to break it down and begin with a smaller first step. See "If the feature is too large" below.

## Step 2 — Learn the codebase context

Before writing anything, read the following to understand how the feature fits the existing system:

- The project's instruction file (`CLAUDE.md` / `AGENTS.md`) — constraints and non-obvious context
- Any agent-facing architecture/structure reference the project maintains (e.g. `PROJECT_OUTLINE.md`, `ARCHITECTURE.md`) — skip if the project has none
- Any source files directly relevant to the feature area

This ensures the spec accurately reflects the system's boundaries and existing behaviour.

## Step 3 — Establish the simplest viable approach

Default to the simplest thing that meets the stated need, unless the project's own conventions (or the user) call for more rigor up front.

Before probing for detail, identify the simplest version of the feature and present it to the user. Then identify any aspects of the request that could be implemented in a more complex or robust way — things like validation, error handling, configuration options, edge case coverage, or extensibility. For each of these, ask the user whether they want it included. Do not assume more is better.

If the user's initial request already contains complexity that isn't strictly necessary for the goal, surface that and ask whether it can be simplified or deferred.

## Step 4 — Resolve ambiguities

Probe for the kinds of ambiguity that most commonly cause misunderstanding between author and reviewer. Ask the user about anything that is unclear, including:

- Which services or components are affected
- Error cases and how they should be handled
- Any configuration or inputs required
- How this interacts with existing features
- Edge cases the user has considered

Do not proceed to write the spec until you have enough detail to fill every section completely. Keep asking until there are no open questions remaining, or until the user explicitly decides to leave something unresolved.

## Step 5 — Create a UI mockup, if this feature has a UI surface

If the feature adds or changes anything user-facing (a new screen, a new component, a layout or visual change), create a static mockup **before** writing the spec, so the spec can reference it and the reviewer can see the intended result instead of imagining it from prose.

1. Build a single self-contained HTML file (inline CSS, no build step, no external framework dependency) showing the relevant screen(s) and states (e.g. empty/loading/error/populated, or desktop/mobile if that varies).
2. Use the project's actual design language if one is established — its real colors, fonts, spacing, and component shapes (check a theme/style-constants file, an existing design system, or prior mockups). If nothing is established yet, use a reasonable placeholder style and say so.
3. Save it under `ui-mocks/<feature-name>.html` at the repo root (create the directory if it doesn't exist).
4. Show the user the file path (and, if you're able to, a rendered preview) and get their sign-off before moving on — treat this like any other spec ambiguity in Step 4.

Skip this step entirely for backend-only / non-UI features.

## Step 6 — Create the spec file

1. Create a subdirectory under [features](/features/) with a short folder name relevant to the feature (e.g. `features/csv-export/`).
2. Create `spec.md` inside that folder using the template below.

### Spec file template

```markdown
# [Feature Name]

## Overview

[One or two sentences describing what the user wants and why.]

## Requirements

[Bulleted list of capabilities and behaviours that must exist. Written from the user/requirements perspective — what the system must do, not how it does it.]

## UI Mockup

[Link to `ui-mocks/<feature-name>.html` if one was created in Step 5. Omit this section entirely for non-UI features.]

## Out of Scope

[Explicit list of related things this feature does NOT include. This is as important as the requirements — it prevents reviewers from making different assumptions about scope.]

## Acceptance Criteria

[Bulleted list of conditions that must be true for the feature to be considered complete. Written as observable outcomes: "Given X, when Y, then Z." Each requirement above should map to at least one criterion here.]

## Open Questions

[Any unresolved ambiguities the user has chosen to defer. Leave this section empty (or omit it) if all questions were resolved.]
```

### Rules for the spec content

- Focus on WHAT is needed, not HOW to build it
- Never add anything the user didn't explicitly ask for
- Default to the simplest approach — only include complexity the user explicitly agreed to in Step 3
- Do not include implementation details such as class names, function signatures, interfaces, or code structure — those belong in the implementation plan
- Do not prescribe the technical approach
- Every requirement must have at least one acceptance criterion

## Step 7 — Commit and optionally open a PR

After creating the spec file(s) (and any UI mockup):

1. Ask the user what branch-naming convention this repo uses (check recent branch names or the project's instruction file if unsure). Propose a branch name that follows it, seeded from the feature folder name (e.g. `feature/csv-export-spec`, or the project's own prefix in place of `feature/`). For nested sub-specs, use the parent folder name. Confirm with the user before creating.
2. Create the branch from the project's base branch (ask if unsure — don't assume `main`): `git checkout -b <branch> <base>`.
3. Stage and commit the spec file(s) and any mockup: `git add features/<feature-folder>/ ui-mocks/<feature-name>.html` (omit the mockup path if none was created) then `git commit -m "Add feature spec for <feature name>"`.
4. Ask the user whether they want to open a PR for review. If yes:
   - Push the branch: `git push -u origin <branch>`.
   - Open a PR with `gh pr create`. Title: `"Feature spec: <feature name>"`. Body should summarise the feature in 2–3 bullets and invite reviewers to check scope, acceptance criteria, and out-of-scope decisions. Do not include a test plan section. Do not include any issue-closing references (e.g. `Closes #123`, `Fixes #123`) — this PR ships only the spec, not the implementation, so the underlying issue must remain open.
   - Return the PR URL.
5. If the user declines a PR, leave the branch as-is and let them know they can open one later with `/submitForPullRequest`.

## If the feature is too large

If the feature is too large to be a single small deliverable, you MUST suggest breaking it into multiple smaller specs. Propose a concrete breakdown and wait for the user to agree before creating any files.

Once agreed, create a parent subdirectory under `features/` for the overall feature, then numbered subdirectories inside it for each sub-feature (using the format `01-`, `02-`, etc. so ordering is unambiguous). Each subdirectory gets its own `spec.md`.

Example structure:

```
features/
  csv-export/
    01-download-button/
      spec.md
    02-scheduled-email/
      spec.md
```
