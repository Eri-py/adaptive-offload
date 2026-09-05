---
description: Documentation Maintainer
---

# Documentation Maintainer

Refresh this project's documentation set so it accurately describes the current state of the code — detect drift and update in place. Used periodically after significant changes.

## Step 0 — Discover the doc set

Every project's documentation layout is different. Before doing anything, find what actually exists:

- Root-level docs: `README.md`, an agent-facing reference doc if one exists (e.g. `PROJECT_OUTLINE.md`, `ARCHITECTURE.md`), and any project-instruction file (`CLAUDE.md`, `AGENTS.md`).
- Per-subsystem/component docs: a `README.md` (or equivalent) inside each major top-level directory (a backend service, a frontend app, a shared package, a database/migrations folder).
- Any other docs directory the project uses for human-facing reference material.

Build a working list of every doc file found, and for each, form a one-line hypothesis of its intended audience and scope (e.g. "human setup + ops", "agent-facing structure reference", "this subsystem's controllers/services/interfaces"). If the project has more than one doc making overlapping claims about the same thing, note that too — it's a sign the split needs clarifying with the user rather than guessed at.

A project-instruction file (`CLAUDE.md` / `AGENTS.md`) is out of scope for this command by default — it changes rarely and has a different shape. If you notice something during the scan that should go there, surface it for the user rather than editing it yourself.

## Principles (non-negotiable)

1. **Split by audience.** Human-facing docs (setup, day-to-day usage, deployment/ops) stay separate from agent-facing reference docs (structure, ports, test index, workflow). Don't blend the two in one file.
2. **Every controller / service / interface / route / component entry links to its source file.** The docs double as a navigation map.
3. **Verify from the code. Never invent.** If a claim (a controller, an interface, a route, a port, a config key) isn't confirmed by a file you actually read, it doesn't go in the doc.
4. **Update in place; don't rewrite.** Preserve prose that's still correct. Touch only sections that have drifted.
5. **No duplication across docs.** If two docs would otherwise repeat the same table/fact, the human-facing doc should link to the agent-facing one (or vice versa) instead of restating it.

## Step 1 — Read the existing docs

Read every file found in Step 0. Note their current claims.

## Step 2 — Detect drift

For each doc, compare its claims against the actual code. The specific checks depend entirely on what the doc claims and what stack the project uses, for example:

- Setup/environment-variable tables → diff against the actual env template file(s) and config-loading code.
- CI/deployment tables → diff against the actual workflow/pipeline files.
- Structural tables (controllers, services, interfaces, routes, pages, components) → Glob the relevant source directories and confirm every entry still exists, and that nothing new is missing.
- Port/config maps → diff against the actual launch config, dev server config, and container/orchestration files.
- Test indexes → Glob the actual test directories/files.

Work outward from what's actually written in each doc — don't assume any of the above categories apply unless the doc in question claims them.

## Step 3 — Surgical updates

For each stale section, edit only what's wrong. Keep prose that's still accurate. Add entries for new code; remove entries for deleted code. Don't restructure a doc unless its structure itself is broken.

When adding controllers/services/interfaces/routes/components:
- One-line purpose description.
- Markdown link to the source file (relative path from the doc).

## Step 4 — Watch for cross-cutting concerns during the scan

If during the scan you notice something that looks like **behavioural guidance** (a non-obvious constraint, a "don't do X" rule, a subtle deployment assumption) rather than structural/reference fact, do not fold it into a structural doc. Either:
- It's specific to one subsystem → add it to that subsystem's own "gotchas" notes.
- It's cross-cutting behavioural guidance → surface it to the user at the end so they decide whether it belongs in the project's instruction file.

Do not edit the project's instruction file (`CLAUDE.md` / `AGENTS.md`) yourself.

## Final output

End with a short report:

- Files changed + which sections drifted (bullet list).
- Things you considered but left alone (e.g. "controllers table was already correct").
- Anything you flagged for the project's instruction file (do not edit it yourself).
- Any gaps — missing files, missing sections, or parts of the code that have no documentation.

If nothing was stale, say so without editing anything.
