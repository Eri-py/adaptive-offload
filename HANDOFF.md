# Handoff — adaptive-offload

Snapshot as of **2026-11-27**, written at the end of a long session that merged
two specs. Standing rules live in `CLAUDE.md` (infrastructure, git workflow) and
`.claude/coding-guidelines.md` (layout, stack conventions, testing bar). This
file is orientation only.

**This file has gone stale before.** The previous version still claimed the
server's frame endpoints were built, from a PR that was closed unmerged, and it
named a next step that had already been abandoned. If it disagrees with a more
recent conversation, the conversation wins — then fix this file.

## The deadline

**A8, "Final Evaluation: Report and Presentation", is due 2026-12-08, 2:30 PM.**
100 points: a written report (60) and a presentation with slides and a recorded
video (40). Unlike every other assignment this term, `lock_at` is null. The
user confirmed they are **not** graduating this semester, so the Nov 30 cutoff
in the assignment text does not apply to them.

A3 and A5 are zeros. A6 and A7 were submitted on time and are ungraded. See the
`project-senior-seminar-deadlines` memory; check Canvas rather than assuming.

## Where the research stands

The project asks: should a request run on-device or be offloaded? The
contribution is the decision layer.

- **The positive result is budget-only routing.** Under a hard latency budget,
  routing on predicted network latency alone — decided *before* any inference
  runs — lands within 0.1-1.6 points of the within-budget oracle and beats both
  static baselines at every budget tested.
- **The cascade is a characterised negative**, with a structural argument, not
  just a bad measurement: escalation costs `L + O` and offloading costs `O`, so
  since `L + O >= O` the escalation band always sits above the offload band. A
  deadline can never make the cascade the only thing that fits.
- **`scene_complexity` is not a router feature.** Correlation with the
  local-vs-offload margin is 0.002.
- **Routing rules stay deterministic** — thresholds and formulas, never a
  learned model. A learned router tied with a plain threshold.

Read `findings/birds.md` before planning anything. `findings/coco-router.md`
covers attempt 1.

## What exists now

| Area | State |
|---|---|
| `training/` | Attempts 1 (COCO) and 2 (birds). `birds.export` produces the phone model. 302 tests. |
| `database/` | Shared models and Alembic migrations. **At revision 0005**; `classify_requests` exists. 1 test. |
| `server/` | FastAPI: `GET /health`, `POST /classify` (raw JPEG body) running ConvNeXt-Base on CUDA, logging every request. 32 tests. |
| `app/` | The scan app. Pick from camera, library or 15 bundled samples; both paths run on identical bytes; both results shown with costs. **No test runner** — `npx tsc --noEmit` is the whole gate. |
| `findings/` | Where results go. The write-up draws on these, not on `learnings.md`. |
| `features/final-demo/` | `01-inference-paths` and `02-scan-both-paths`, both merged. `03` is not written. |

Merged this session: **PR #19** (spec 01), **PR #20** (the arrival-pattern
finding), **PR #21** (spec 02). `main` is at `ca37ea4`.

Gate: `training/` 302, `database/` 1, `server/` 32 — 335 tests, none skipped.
`ruff` clean everywhere; `mypy` clean in `database/` and `server/`; `training/`
carries exactly 9 pre-existing `ndarray` type-arg errors in `coco/`. Note
`mypy .` from `training/` also sweeps the gitignored `scratch/`, which has its
own pre-existing errors — judge against the 9 outside it.

## What the deployed system actually costs

These are measured, and three of them contradict the simulation the analysis was
built on. All from a **development (Debug) build**; `app/README.md` says only a
`preview` build's latency is quotable.

| Quantity | Simulation assumed | Measured |
|---|---|---|
| Server compute | 6.731 ms | **~35 ms** |
| Local cost | 5.877 ms desktop / 4.20 ms model | **82-108 ms total** |
| Payload | 11.39 KB | **11,926 B exact** for prepared input |

- **Server compute depends on request arrival pattern.** Idle GPU at ~292 MHz
  gives 33-43 ms; after a burst the clock reaches 825 MHz and it drops to
  6.5-8 ms. With requests 20 s apart the clock is back at idle before *every*
  one. The 6.731 ms figure is the offline `evaluate` number, measured as a tight
  loop. **The project's own per-request interaction model selects the bad case.**
- **The on-device path is dominated by JPEG decoding**, 75-93 ms against 5.5-18
  ms of model. `jpeg-js` is pure JavaScript and the offload path never pays it,
  so this is a property of the client, not of on-device inference. A native
  decoder would remove most of it.
- **Decode time tracks payload size**: roughly a 70 ms floor plus ~1 ms/KB. So
  the encoder setting costs twice — bytes on the wire *and* decode before the
  local model runs — and the two paths' costs are **not independent**. Payload
  size is knowable before either path runs, which makes it a far better router
  feature than `scene_complexity` ever was.
- **On-device confidence drifts ~0.05** from the desktop on the same bytes,
  because `jpeg-js` and PIL decode differently. Any on-device confidence
  threshold has that as a floor on its precision.

## Next step: spec 03, the router

The remaining research work. It should cover the background network monitor
(cached rolling estimate, never measured on the critical path), the
deterministic budget rule choosing one path *before* inference, and the paired
record of the decision and its outcome.

Build it on the measured constants above, not the simulated ones. Spec 02
deliberately runs both paths on every scan, which produces exactly the paired
"which path won" record the simulator computed — that is the substrate spec 03's
router should be judged against.

## Running it

Nothing below is ever started by an agent. See `CLAUDE.md`.

```bash
# server (user runs)
cd server && uvicorn api.core.app:app --host 0.0.0.0 --port 8000
# app (user runs)
cd app && npx expo start --tunnel
```

- The server is reachable on the tailnet at **`http://100.97.204.46:8000`**.
  `0.0.0.0` is what makes that work.
- **Tailscale** is installed in WSL (systemd is PID 1, `/dev/net/tun` exists, so
  full TUN mode). WSL node `100.97.204.46`, phone `eri-iphone` 100.77.49.10.
  `tailscale up --accept-dns=false` is deliberate — MagicDNS rewrites
  `/etc/resolv.conf` in WSL and breaks everything else.
- **The tailnet link is relayed, not direct** (`via DERP`, ~34 ms). `netcheck`
  shows `MappingVariesByDestIP: true` — symmetric NAT, which hole punching
  cannot beat — and no port mapping. The deeper cause is WSL2's own NAT layer;
  mirrored networking would likely fix it but needs `wsl --shutdown`, which
  stops Postgres. Offload round trips include this relay and are reported as
  measured.
- `--tunnel` was needed because iOS App Transport Security blocks cleartext HTTP
  to Tailscale's CGNAT range. The current dev build now sets
  `NSAllowsArbitraryLoads`, so `REACT_NATIVE_PACKAGER_HOSTNAME=100.97.204.46
  npx expo start` may now work without the tunnel — untested.
- The installed dev build includes the camera, the photo library and the ATS
  exception. Those are native, so changing them means another EAS build; all
  JS reloads over Metro.

## Open questions

- **Does `compress: 0.85` match PIL quality 85?** Still unmeasured cleanly. The
  one camera scan gave 1.92x, but on a screen photo full of text — high entropy
  and larger than every CUB crop. Needs several camera scans of bird-like
  scenes. Only the camera and library paths encode; prepared images pass
  through untouched.
- **Release-build numbers.** The decode cost especially — it is the most
  JavaScript-heavy step and Debug is where JS is slowest.
- **Would a native decoder collapse the 75-93 ms?** If local is really ~25 ms,
  the routing boundary moves a long way.
- The resize/crop path has **no bundled coverage** now that prepared images
  pass through; camera and library photos are its only exercise.

## Conventions worth knowing

- **Infrastructure is the user's.** Never start or stop Postgres, the server or
  any long-running process; never run a migration. The one exception is test
  fixtures creating their own disposable database.
- **Feature branch + PR always**, even for one-line fixes.
- **Non-trivial work goes through `.claude/commands/`**: `generateFeatureSpec`
  → `generateImplementationPlan` → `executePlan` → `applyReview` →
  `submitForPullRequest`. The review step earns its keep: it caught a blocker in
  each of the last two specs, one of which would have left `classify_requests`
  permanently empty, and one of which was silently corrupting the measurements.
- **Verify what subagents report.** Several claims this session were wrong or
  unverified until checked — a status line that depended on an uninstalled
  binary, a parity gate that tested the wrong artifact, a "no-op" that was
  resampling every image.
- **`git add` aborts entirely on a bad pathspec.** Naming an
  already-`git rm`-ed file silently stages nothing.
- Each top-level area owns its own `pyproject.toml` and lint config; all three
  Python areas share the root `.venv/`. Run each area's gate from its own
  directory.
