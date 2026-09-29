# Mobile Inference Benchmark — Implementation Report

**Feature:** Mobile Inference Benchmark
**Directory:** `features/mobile-inference-benchmark/`

## Tasks

| Task | Outcome |
|------|---------|
| Task 1 — Export the model and bundle the test images | completed (blocked twice, resolved on the third attempt) |
| Task 2 — Scaffold the Expo app | completed (blocked once, resolved) |
| Task 3 — Add on-device TFLite inference | completed |
| Task 4 — Build the benchmark hook | completed |
| Task 5 — Build the benchmark screen | completed |
| Task 6 — EAS Build configuration | completed |
| Task 7 — Regression test run | completed |

## Files changed

### Added

- `training/export_tflite_model.py` — one-off script exporting `training/models/yolov8n.pt` to TFLite and bundling 15 real COCO photos into the app
- `app/assets/models/yolov8n.tflite` — exported model (~12.3MB), committed
- `app/assets/images/*.jpg` (15 files) — bundled real COCO photos, committed
- `app/package.json`, `app/package-lock.json`, `app/app.json`, `app/tsconfig.json`, `app/eas.json`, `app/metro.config.js`, `app/.gitignore` — Expo project configuration
- `app/App.tsx`, `app/index.ts` — app entry point
- `app/screens/BenchmarkScreen.tsx` — the single screen (bundled-set info, run button, results)
- `app/hooks/useBenchmark.ts` — loads the model, runs inference over all bundled images, computes latency stats
- `app/types/jpeg-js.d.ts` — hand-written ambient type declaration (the published `@types/jpeg-js` targets an older, incompatible API)
- `app/assets/icon.png`, `app/assets/splash-icon.png` — template-provided app icons
- `features/mobile-inference-benchmark/{spec,implementation,progress,learnings}.md`
- `ui-mocks/mobile-inference-benchmark.html` — UI mockup (idle/running states)

## Tests

No automated test suite was added for `app/` — per the plan's own key decisions, this is a one-screen debug harness with no pipeline-correctness stakes, and the repo's coding guidelines exempt UI code without pipeline impact from the testing bar. Verification instead relied on:

- `npx tsc --noEmit` — clean after every task, reverified independently by the orchestrator (not just trusted from each subagent's own report) after Tasks 4, 5, and again at the final regression run.
- `npx expo start` reaching a ready state (`Waiting on http://localhost:8081`) with no crash — reverified independently after Tasks 2, 5, and the final regression run.
- For the native TFLite module (Task 3) and the hook that calls it (Task 4): the exact `loadTensorflowModel`/`model.run()` call shape was verified against the installed library's real `.d.ts` types, and `expo prebuild` was used to confirm the native iOS project generates correctly with the CoreML delegate wired into the Podfile. Full on-device execution (does inference actually run and return real output) could not be verified in this environment — no Mac, Xcode, CocoaPods, or physical iOS device is available here. This is expected and was anticipated by the plan; real confirmation happens when the user runs an EAS build on their own device.
- `training/export_tflite_model.py`'s output was independently verified: the exported `.tflite` file was loaded with `tf.lite.Interpreter` and confirmed to report the correct `[1, 640, 640, 3]` input / `[1, 84, 8400]` output tensor shapes, matching the PyTorch model's own reported export shape. The 15 bundled images were verified genuinely openable/valid via PIL (`Image.verify()` plus a full re-decode).
- `app/eas.json`'s config was validated against the real EAS schema/merge-resolution logic (`@expo/eas-json`, driven directly, no login required) rather than just hand-inspected.

All checks pass; nothing was skipped.

## Commits

`feature/real-model-inference..feature/mobile-inference-benchmark` (10 commits): spec, plan, and one commit per task (Task 1 required three commits across its multi-attempt resolution — see Notable events).

## Notable events

- **Task 1 required three attempts to resolve**, each a real, distinct blocker rather than a retry of the same fix:
  1. `ultralytics`' modern TFLite export path (`litert_torch`) conflicted with the shared venv's pinned `torch`/`torchvision` versions.
  2. Pinning `ultralytics` below that cutover fixed conflict #1 but surfaced a second, unrelated one — the legacy TensorFlow-based exporter's auto-installed `tensorflow` crashed against the shared venv's `numpy`/`scipy` versions.
  3. Rather than continue pinning the shared venv (used by the rest of the training/router pipeline), the export was moved into a throwaway, isolated virtualenv that never touches the shared one — this fully resolved it, and was verified not to regress the existing pipeline (86/86 tests still passing in `training/` throughout).
  This is documented in detail in `learnings.md` for anyone who needs to re-run the export later.
- **Task 2 hit a real, structural environment blocker**: the only Node.js available at the start of this feature was a Windows-side install reached through WSL, and Expo's Metro dev server crashed (`EISDIR`) trying to watch the project directory across that filesystem boundary. Fixed by installing a native Linux Node.js and switching the project to use it — this also removed a set of `.cmd`/UNC-path workarounds Task 2 had initially needed for `npx`-based commands, which no longer apply after the fix (documented as now-obsolete in `learnings.md` for context, but the actual dependency on them is gone).
- **The branch's base was changed mid-feature**: `feature/mobile-inference-benchmark` was originally branched from `main` (correct per the spec's own "no dependency on training/, database/, server/" framing for the *shipped app*), but Task 1's export script legitimately needs `training/`'s real dependency-managed environment (`training/pyproject.toml`) to exist and be editable, which only exists on the not-yet-merged `feature/real-model-inference` branch. The branch was rebased onto `feature/real-model-inference` before Task 1's corrective work — safe, since it had never been pushed. This is why the Files Changed list above doesn't include any `training/` files beyond the one new export script.
- No task was abandoned or left in a `failed` state at the end — every blocker encountered was resolved before moving to the next task, per the plan-execution rule against continuing past a failure.
