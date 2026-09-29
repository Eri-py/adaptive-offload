# Handoff — adaptive-offload

Snapshot as of 2026-09-29. This file describes *where the project is*; the
standing rules live in `CLAUDE.md` (infrastructure and git workflow) and
`.claude/coding-guidelines.md` (layout, stack conventions, testing bar). Read
those for rules, this for orientation.

## Where this project sits

Senior Seminar (CSCI-411-01) semester project. The research design lives in the
Windows-side School repo, not here — a `SessionStart` hook reads
`School/Senior Seminar/Project Technical Notes.md` and the assignment folder
listing into context automatically at the start of every session. If that hook
output isn't in context, read that file before touching anything: it is the
source of truth for what this project is trying to do (per-request
local-vs-offload routing, not continuous streaming).

A3 ("Progress Report 1") was due 2026-09-15 and has passed. Check the
assignment folders for what's next — this repo doesn't track deadlines.

## What exists now

| Area | State |
|---|---|
| `training/coco/` | Attempt 1. Data-gen simulator + router experiments. The substance of the research so far. |
| `training/birds/` | Attempt 2. MobileNetV3-Large / ConvNeXt-Base on CUB-200-2011, both trained and evaluated. |
| `training/shared/` | MLflow tracking, used by every attempt. |
| `database/` | SQLAlchemy engine/session, shared models, Alembic migrations, ephemeral-test-DB fixture. |
| `app/` | React Native inference benchmark. Runs on a real iPhone; has produced numbers. |
| `server/` | **Scaffolding only.** `server/api/` holds one `__init__.py`. No endpoints, no tests. |
| `findings/` | `coco-router.md`, `birds.md` — where results are recorded. |
| `features/` | Ten feature directories, each with spec, plan, implementation report and review. |

Tests: `training/` 222 passing, `database/` 1 passing, `server/` none (nothing
to test yet). Ruff clean; mypy has 9 pre-existing `ndarray` type-arg errors in
`training/coco/` that predate current work.

Run history is in Postgres (the `mlflow` database, separate from
`adaptive_offload`), not a folder. View it with `mlflow-ui` — the user runs it,
never an agent.

## Where the research actually stands

Read `findings/coco-router.md` before planning anything. The short version:

- **Soft-utility routers failed.** None of the direct-classifier or
  utility-regression variants beat "always offload". Root cause recorded:
  `scene_complexity` carries no signal about the local-vs-offload margin.
- **Budget-only routing looks strong.** Under a hard latency budget, routing on
  predicted network latency alone — decided before any inference runs — lands
  within 0.1–1.6 points of the within-budget oracle and beats both static
  baselines at every budget tested.
- **But that result is simulated, and its own findings name the catch.**
  It holds largely because simulated offload latency is nearly deterministic
  (offload-latency predictor reaches R² = 0.9951). Whether real network latency
  is anywhere near that predictable is, in the findings' own words, "the central
  risk this simulated result carries, not a secondary check."
- **The cascade verdict is provisional, not settled.** `coco-router.md` records
  cascade as "not useful at any budget", then immediately qualifies it: the
  cascade's disadvantage is the local-first latency tax, and if real on-device
  latency via Core ML is much cheaper than the simulated ~74 ms, that tax
  shrinks and the comparison must be re-run. Don't treat the cascade as dead.

## The one thing blocking the next decision

**Real-hardware latency.** Both open questions — is budget-only's predictability
real, and is the cascade actually dead — resolve the same way: measure on the
phone and over a real network.

Partial data exists. `features/mobile-inference-benchmark/learnings.md` records
an iPhone 15 Pro at **17.34 ms mean via Core ML** vs **106.01 ms on CPU**. Two
caveats, both recorded there: it was a Debug build (the notes say re-run from a
Release/`preview` build before reporting), and it measures the forward pass
only.

That number already undercuts a comparison elsewhere in the repo:
`birds/evaluate.py` times the "phone" model on the desktop CPU
(`5.877 ms/photo` at last run) while the real phone CPU measured 106 ms. The
desktop stand-in flatters local by a large factor, and any win/loss label
derived from it inherits the bias.

## Next step

`features/server-served-benchmark/01-server-frame-endpoints/spec.md` is written
and unimplemented — server-side frame endpoints, so the app can offload over a
real network. That is the natural next feature, and it is what unblocks
everything above. It has a spec but no plan; start at
`generateImplementationPlan`.

## Branch and PR state

Everything is on `main` as of PR #12 (2026-09-29). Eleven PRs merged, numbered
#1–#12 — there is no #9. For a long stretch PRs merged into each other rather than into
`main`, which left the default branch showing almost nothing while the work
lived in a five-deep branch stack — PRs #11 and #12 landed all of it. GitHub
auto-deletes branches on merge here, so a merged branch disappears from origin;
its commits remain reachable via `refs/pull/<n>/head` if ever needed.

Keep future work landing on `main` directly rather than restacking.

## Conventions worth knowing before you start

- **Infrastructure is the user's.** Never start/stop Postgres, servers or any
  long-running process; never run migrations or create/drop databases. The one
  exception is test fixtures creating their own disposable database. See
  `CLAUDE.md` — it's a hard rule.
- **Feature branch + PR always**, even for one-line fixes.
- **Non-trivial work goes through the spec-driven flow** in `.claude/commands/`:
  `generateFeatureSpec` → `generateImplementationPlan` → `executePlan` →
  `applyReview` → `submitForPullRequest`.
- Each top-level area owns its own `pyproject.toml` and lint/type config, but
  all three Python areas share the root `.venv/`.
