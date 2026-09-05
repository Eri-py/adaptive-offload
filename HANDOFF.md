# Handoff — adaptive-offload

Written 2026-09-05, closing out the session that scaffolded this repo. Read
this once, then `CLAUDE.md` and `.claude/coding-guidelines.md` for the
standing rules — this file is a snapshot, those are the rules.

## Where this project sits

Senior Seminar (CSCI-411-01) semester project. The research design lives in
the Windows-side School repo, not here — a `SessionStart` hook already reads
`School/Senior Seminar/Project Technical Notes.md` and the assignment folder
listing into context automatically at the start of every session in this
repo, so you don't need to go read it manually. If that hook output isn't in
context for some reason, read that file first before touching anything —
it's the source of truth for what this project is actually trying to do
(per-request local-vs-offload routing decision, not continuous streaming).

A3 ("Progress Report 1") is due **2026-09-15** and explicitly asks for
evidence of real code progress (a GitHub link, code examples, screenshots) —
not just a status update. That's the immediate deadline pressure on this
repo.

## What exists right now

**Nothing is implemented yet.** This session was scaffolding only, at the
user's explicit choice. What's here:

```
app/                          empty
server/
  api/{controller,contracts,services,core,dependencies}/   empty
  common/                     empty
  tests/                      empty
training/
  datagen/                    empty
  router/                     empty
CLAUDE.md                     infra + git-workflow rules
.claude/
  agents/implementer.md       adapted from CSHub, quality gate updated for this stack
  agents/reviewer.md          copied from CSHub, unmodified
  skills/grill-me/            copied from CSHub, unmodified
  commands/                   copied from CSHub (applyReview, bootstrap, executePlan,
                               generateFeatureSpec, generateImplementationPlan,
                               refreshDocs, submitForPullRequest) — the spec-driven
                               workflow implementer/reviewer are meant to run under.
                               Two illustrative example filenames swapped .cs -> .py;
                               otherwise unmodified.
  coding-guidelines.md        written for this stack (see below)
  settings.json                SessionStart hook (Senior Seminar context)
  hooks/senior_seminar_context.py
```

No `package.json`, no `pyproject.toml`/`requirements.txt`, no FastAPI app,
no RN app, no Alembic setup, no database tables. All of that is next-session
work.

## Stack decisions made this session

- **App:** React Native, targeting **iOS only** (user only has an iPhone).
- **Server:** Python + FastAPI + YOLOv8, laid out like a layered dotnet
  solution (see `coding-guidelines.md` for the full `server/api/*` breakdown
  — `controller/` not `routers/`, `contracts/` not `schemas/`).
- **Training/research code:** one `training/` folder, subfoldered into
  `datagen/` (simulation/condition-sweep harness) and `router/`
  (decision-layer models), not split into separate top-level folders.
- **Database:** a single Postgres instance **the user already runs
  themselves** — no docker-compose, no SQLite, no CSV/Parquet. Backs both
  server request logs and the data-gen sweep results. `server/common/` will
  own the SQLAlchemy engine/models; `training/` imports from there.
  Migrations are authored in-repo (Alembic) but the user runs them by hand.
- **Infrastructure control:** never start/stop Postgres, the server, or any
  long-running process yourself, and never run migrations or create/drop
  the database — always ask the user. This is a hard rule in `CLAUDE.md`,
  not a suggestion.

## Tooling copied/adapted from CSHub

`C:\Users\eriol\Desktop\Projects\CSHub` is the user's template project for
`.claude/` conventions. `agents/`, `skills/grill-me/`, and `commands/` were
copied as-is (commands only got copied after the user caught the omission —
check that folder before assuming something's missing next time). The
`commands/` workflow (`bootstrap` → `generateFeatureSpec` →
`generateImplementationPlan` → `executePlan` → `applyReview`) is what
actually drives the `implementer`/`reviewer` agents; they're not much use
without it. `implementer.md`'s quality-gate step was rewritten for this
stack (RN lint+typecheck, Python ruff+mypy for both `server/` and
`training/` separately). `coding-guidelines.md` was written from scratch for
this repo (CSHub doesn't actually have one — its agents reference a file
that doesn't exist there either).

## Hooks in place

- **This repo** (`~/projects/adaptive-offload/.claude/settings.json`):
  `SessionStart` → `.claude/hooks/senior_seminar_context.py` reads the
  Windows-side tech notes + assignment folder list into context on every
  new session here.
- **School repo** (`C:\Users\eriol\Desktop\School\.claude\settings.json`):
  `PreToolUse` on `Edit`/`Write`, scoped to `Senior Seminar/**` via the `if`
  filter, runs `Senior Seminar\.scripts\adaptive-offload-check.ps1` — checks
  this WSL repo for recent changes before a School-side session edits/writes
  anything in the Senior Seminar folder, so progress-report writing doesn't
  drift from actual code state. Both were pipe-tested and fire-tested
  working as of this session.

## Suggested next step

Per the tech notes' own feasibility section, the data-gen/router loop is the
highest-risk, most iterative part — worth tackling before the two inference
paths. But that was never confirmed for *this* session; it was scoped as
scaffolding-only. Whoever picks this up next should confirm with the user
what to build first rather than assuming.
