## Task 1 — Export the model and bundle the test images
- Status: completed
- Started: 2026-09-23 19:47:39
- Completed: 2026-09-23 22:59:37
- Notes: Image bundling succeeded (15 real, verified photos in app/assets/images/).
  Two prior corrective attempts pinned `training/pyproject.toml`'s ultralytics
  version in the shared venv and got progressively further (litert/torch
  conflict, then a tensorflow/numpy/scipy conflict) but never produced a
  `.tflite` file — see learnings.md for that history. Final resolution:
  stopped trying to make the shared venv's stack coexist with the tflite
  export toolchain and isolated the export into a throwaway virtualenv
  (`/tmp/tflite-export-venv`, created with `uv venv` since `python3 -m venv`
  had no working `ensurepip` here) outside `.venv/`. The shared venv's
  `ultralytics` pin was reverted to unpinned `ultralytics>=8.0` since it was
  only ever needed for the abandoned shared-venv approach. In the isolated
  venv, pinning `ultralytics<8.4.83` (forces the legacy TF exporter) plus
  `numpy==2.0.2`/`scipy==1.13.1` (avoids a real numpy/scipy internal
  incompatibility, not shared-venv-specific) plus `onnx<1.18` (protobuf
  gencode/runtime compatibility) plus explicit `tf_keras`/`sng4onnx`/
  `onnx_graphsurgeon` installs (ultralytics' own auto-install for these hit
  an unreachable NVIDIA package mirror) got a clean export.
  `app/assets/models/yolov8n.tflite` now exists (12,865,855 bytes, ~12.3 MB),
  verified as a real, loadable TFLite model via `tf.lite.Interpreter`:
  input `[1, 640, 640, 3]` float32, output `[1, 84, 8400]` float32, matching
  the PyTorch model's own reported output shape. `training/export_tflite_model.py`'s
  module docstring now documents the isolated-venv setup steps. Confirmed
  the shared venv is untouched: `ruff check . && mypy . && pytest -q` from
  the shared `.venv/` is still clean/86-passing, with the same pre-existing
  mypy gaps as before (unrelated `ndarray` generic-type and `ultralytics`
  stub-coverage errors, not introduced by this task). See learnings.md
  follow-up entry for full diagnosis and final package versions.

## Task 2 — Scaffold the Expo app
- Status: completed
- Started: 2026-09-23 23:00:33
- Completed: 2026-09-23 23:15:44
- Notes: Expo TypeScript template scaffolded into app/ (iOS-only config,
  placeholder App.tsx per coding-guidelines style). Type-check clean. The
  `expo start` success criterion initially failed — Metro's file watcher
  crashed (`EISDIR`) watching the project directory through the Windows-side
  Node's WSL/UNC path. Fixed by installing a native Linux Node.js (v24.21.0)
  to ~/.local/node, symlinked into ~/.local/bin (reliably on PATH regardless
  of shell invocation mode — see learnings.md for why .bashrc/.profile edits
  alone weren't enough), and reinstalling app/node_modules clean under it.
  Verified: `expo start` now reaches "Waiting on http://localhost:8081"
  with no crash. This also unblocks Tasks 3/4/5/7, which all need a working
  dev-server run to verify.

## Task 3 — Add on-device TFLite inference
- Status: completed
- Started: 2026-09-23 23:16:10
- Completed: 2026-09-23 23:23:36
- Notes: react-native-fast-tflite + react-native-nitro-modules installed,
  config plugin wired in app.json (enableCoreMLDelegate: true), metro.config.js
  added (required by the library's own install steps for .tflite asset
  resolution — outside the plan's original Files list but a necessary part
  of "wire the library in", flagged rather than silent). Verified: Expo
  plugin system actually applies the config (confirmed via `expo config`),
  `expo prebuild` generates a real iOS project with CoreML delegate wired
  into the Podfile, and the exact load/run call Task 4 will make type-checks
  cleanly against the library's real .d.ts signatures. Full native execution
  (does inference actually run and return real output) is NOT verifiable in
  this environment — no Mac/Xcode/CocoaPods/physical device — expected per
  the plan; real confirmation comes from EAS Build (Task 6) on the user's
  phone. No regression: expo start still reaches ready state, tsc clean.

## Task 4 — Build the benchmark hook
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 5 — Build the benchmark screen
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 6 — EAS Build configuration
- Status: not started
- Started: —
- Completed: —
- Notes: —

## Task 7 — Regression test run
- Status: not started
- Started: —
- Completed: —
- Notes: —
