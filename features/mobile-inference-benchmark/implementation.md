# Mobile Inference Benchmark — Implementation Plan

## Summary

Build a minimal Expo (React Native) iOS app in `app/` — the first code in that
folder — that loads YOLOv8n (exported to TFLite) and a bundled set of real
COCO photos, runs on-device inference over them, and displays real,
device-measured latency stats on screen. Built and deployed via EAS Build
(cloud), no local Xcode involved.

## Approach & Key Decisions

- **Inference library: `react-native-fast-tflite`**, with its Expo config
  plugin (`enableCoreMLDelegate: true` for iOS GPU acceleration). Chosen over
  a custom Core ML native module during spec discussion — no custom native
  Swift code to write or maintain.
- **Model + images are committed to git, not gitignored.** The repo's
  existing convention gitignores `*.pt`/`*.joblib` as regenerable Python-side
  build output — but the exported `.tflite` model and bundled photos aren't
  optional build output here, they're assets the app ships with. EAS Build
  needs them present in the uploaded project; a nano-model TFLite export is a
  few MB, and 15 photos are a few hundred KB each, both small enough that
  committing them is the safer, lower-surprise choice over relying on exact
  `.gitignore`/`.easignore` upload semantics.
- **One-off Python export script, not a registered CLI tool.**
  `.claude/coding-guidelines.md` deliberately caps `training/datagen/cli/`'s
  registered tools at two (`run-simulation`, `score-complexity`) — this
  export script is unrelated to that pipeline (a rare, one-off conversion
  step, not part of the simulation loop), so it's a plain script, run
  directly, not added to `training/pyproject.toml`'s `[project.scripts]`.
- **Structure follows `.claude/coding-guidelines.md`'s `app/` conventions**:
  functional components + hooks, one component per file (`PascalCase.tsx`),
  co-located styles, typed props, no `any`. The actual benchmark-running
  logic (load model, loop over images, time each inference, compute stats)
  lives in a hook (`useBenchmark.ts`), not inline in the screen component —
  per the guideline that anything non-presentational belongs in a hook or
  service module.
- **No automated tests for the app code itself.** This is a one-screen debug
  harness with no pipeline-correctness stakes — `coding-guidelines.md`'s
  testing bar explicitly exempts "UI polish and unlikely error states"
  without pipeline impact. The Python export script is simple enough
  (one `ultralytics` call, one file copy) that a smoke-test run stands in
  for a unit test.

## Out of Scope

Carried from the spec: camera/live capture, any call to the offload/FastAPI
server or simulated network conditions, the router/routing decision logic,
on-device accuracy measurement, persisting results anywhere off-device,
Android support, production polish.

Also out of scope for this plan specifically: setting up the user's Expo
account or Apple Developer credentials — those are account-level actions
only the user can do (`eas login`, accepting Apple's terms, etc.), not
something a task here can complete unattended. The plan gets the project to
the point where `eas build` is ready to run; actually running an
authenticated build against the user's accounts happens with the user
present, not as an unattended task.

## Dependencies and Configuration

- New Expo/React Native project scaffolded in `app/` — first `package.json`
  in this repo.
- `react-native-fast-tflite` + its Expo config plugin.
- An Expo account (free) for EAS Build — the user sets this up themselves
  (`eas login`) if they haven't already.
- The user's existing Apple Developer account, used by EAS for device
  provisioning/signing.
- No new Python dependency — `ultralytics` (already a `training/` dependency)
  supports `model.export(format="tflite")` directly.
- No database, no server dependency, no new backend configuration.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/export_tflite_model.py` | add | One-off script: exports `training/models/yolov8n.pt` to TFLite, copies the result into `app/assets/models/` | Bridges the existing Python-side weights to the app; not part of the datagen pipeline, so not a registered CLI tool |
| `app/package.json` | add | Expo project manifest | First JS/TS content in the repo |
| `app/app.json` | add | Expo app config, incl. the `react-native-fast-tflite` config plugin | Required for the TFLite native module + Core ML delegate |
| `app/eas.json` | add | EAS Build profile(s) (development/internal, iOS) | Needed for `eas build` to run at all |
| `app/tsconfig.json` | add | TypeScript config | Per guidelines, typed props/no `any` |
| `app/App.tsx` | add | Entry point, renders the benchmark screen | Expo's default app entry |
| `app/screens/BenchmarkScreen.tsx` | add | The single screen: bundled-set info, run button, results display | One component per file, per guidelines |
| `app/hooks/useBenchmark.ts` | add | Loads the TFLite model, runs inference over bundled images N times, computes mean/median/min/max/stdev | Non-presentational logic belongs in a hook, per guidelines |
| `app/assets/models/yolov8n.tflite` | add | The exported model, bundled with the app | Produced by `export_tflite_model.py`, committed (see Key Decisions) |
| `app/assets/images/*.jpg` | add (15 files) | Real COCO photos, bundled with the app | Copied from `training/data/coco/val2017/`, committed (see Key Decisions) |

## Tasks

### Task 1 — Export the model and bundle the test images

- **Objective:** Produce `app/assets/models/yolov8n.tflite` and 15 real COCO
  photos under `app/assets/images/`, both committed to git.
- **Files:** `training/export_tflite_model.py`, `app/assets/models/yolov8n.tflite`, `app/assets/images/*.jpg`
- **Details:** `training/export_tflite_model.py` is a plain script (not a
  CLI entry point) that loads `training/models/yolov8n.pt` via `ultralytics`
  and calls `model.export(format="tflite")`, then copies the resulting
  `.tflite` file to `app/assets/models/yolov8n.tflite`. Separately (can be
  done by the same script or a short inline step), copy 15 real photos from
  `training/data/coco/val2017/` into `app/assets/images/` — any 15 is fine,
  determinism doesn't matter here since this benchmark isn't compared
  frame-for-frame against anything.
- **Success criteria:**
  - `app/assets/models/yolov8n.tflite` exists and is a valid TFLite file
    (loadable by `tf.lite.Interpreter` or equivalent, sanity-checked once
    from the Python side).
  - `app/assets/images/` contains exactly 15 real `.jpg` files.

### Task 2 — Scaffold the Expo app

- **Objective:** A minimal, building Expo/TypeScript project in `app/`, with
  no benchmark logic yet — just the empty shell.
- **Files:** `app/package.json`, `app/app.json`, `app/tsconfig.json`, `app/App.tsx`
- **Details:** Standard Expo TypeScript template (`npx create-expo-app`or
  equivalent), targeting iOS. `App.tsx` can render a placeholder ("Mobile
  Inference Benchmark") — no screen logic yet. Follow
  `.claude/coding-guidelines.md`'s TypeScript conventions from the start
  (explicit types, no `any`).
- **Success criteria:**
  - `npx tsc --noEmit` (or the project's equivalent type-check command)
    passes cleanly.
  - The app starts in Expo's dev client / simulator without crashing (a
    local sanity check; doesn't require EAS or a physical device yet).

### Task 3 — Add on-device TFLite inference

- **Objective:** Wire `react-native-fast-tflite` into the project and confirm
  it can load `yolov8n.tflite` and run one inference against one bundled
  image.
- **Files:** `app/package.json` (dependency), `app/app.json` (config plugin)
- **Details:** Install `react-native-fast-tflite`; add the config plugin
  entry with `enableCoreMLDelegate: true`. This task's success criterion is
  deliberately narrow (load + one inference), isolating "does the native
  module work at all" from the benchmark logic built in Task 4.
- **Success criteria:**
  - A throwaway test call (removed or left minimal before Task 4) loads
    `app/assets/models/yolov8n.tflite` and runs inference against one bundled
    image without throwing.

### Task 4 — Build the benchmark hook

- **Objective:** `useBenchmark.ts` — loads the model once, runs inference
  over all 15 bundled images, times each call, and exposes aggregate stats.
- **Files:** `app/hooks/useBenchmark.ts`
- **Details:** Exposes something like `{ status, results, runBenchmark }`
  where `results` is `{ mean, median, min, max, stdev }` in milliseconds once
  a run completes, and `status` reflects idle/running/done so the screen can
  show progress. Wrap the inference calls in try/catch per guidelines — a
  failure should surface as an error state, not fail silently.
- **Success criteria:**
  - Calling `runBenchmark()` runs inference on all 15 bundled images exactly
    once each and produces a populated stats object.
  - An inference failure (e.g. forced by a bad path in manual testing)
    surfaces as a visible error state, not a silent no-op.

### Task 5 — Build the benchmark screen

- **Objective:** `BenchmarkScreen.tsx` — the single screen from the spec's UI
  mockup (`ui-mocks/mobile-inference-benchmark.html`): bundled-set info, a
  "Run Benchmark" button, and the results block once a run completes.
- **Files:** `app/screens/BenchmarkScreen.tsx`, `app/App.tsx` (renders it)
- **Details:** Purely presentational — reads `status`/`results` from
  `useBenchmark` and renders accordingly (idle / running / done). Visual
  styling is not a priority (per the spec) — structure over polish.
- **Success criteria:**
  - Tapping "Run Benchmark" triggers `useBenchmark`'s `runBenchmark()` and
    the screen reflects the running state, then displays the final stats.

### Task 6 — EAS Build configuration

- **Objective:** `app/eas.json` configured for an iOS development/internal
  build profile, so the project is ready for the user to run an authenticated
  `eas build` themselves.
- **Files:** `app/eas.json`
- **Details:** This task configures the build profile only — it does not run
  an authenticated build (that needs the user's own Expo login and Apple
  Developer credentials live, which can't be done unattended; see Out of
  Scope). Success here is "the config is correct and complete," not "a build
  was produced."
- **Success criteria:**
  - `app/eas.json` defines a profile suitable for an iOS physical-device
    development build.
  - The task's completion note tells the user exactly what command to run
    next (`eas build --platform ios --profile development`) and that it
    requires their own `eas login` and Apple Developer account.

### Task 7 — Regression test run

- **Objective:** Confirm the whole project still type-checks and starts
  cleanly after all prior tasks.
- **Files:** none changed
- **Success criteria:**
  - `npx tsc --noEmit` passes with no errors.
  - The app still starts in Expo's dev client without crashing.

## Open Questions

- Exact Expo SDK / React Native version to target — left to whatever
  `create-expo-app`'s current template installs at execution time, rather
  than pinning a version now that may already be stale by the time this
  plan executes.
- Whether the user wants the 15 bundled photos to be a fixed, checked-in
  list (reproducible across rebuilds) or freshly picked each time
  `export_tflite_model.py` runs — deferred to Task 1; either is fine since
  this benchmark isn't compared frame-for-frame against anything.
