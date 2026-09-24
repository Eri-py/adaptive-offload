# Mobile Inference Benchmark

## Overview

The router training data's `local_latency_ms` values were measured by actually
running YOLOv8n on the dev machine's CPU (a desktop-class Ryzen 5 9600X), not
on real mobile hardware. A quick check showed that workload is single-core
speed bound (thread count doesn't move the number), and desktop single-core
performance is nowhere near representative of a phone's. We need a real,
on-device latency measurement from the user's actual iPhone to know how far
off the desktop-measured numbers are — there's no way to simulate real mobile
silicon on desktop (an emulator still executes on the host CPU).

## Requirements

- A React Native (Expo) app, under `app/`, buildable via EAS Build (cloud
  build — no local Xcode/Mac involved) and installable on the user's physical
  iPhone via their existing Apple Developer account.
- The app bundles a fixed set of real COCO validation images (at least 10,
  drawn from `training/data/coco/val2017/`) and the YOLOv8n model exported to
  TFLite format (via ultralytics' `model.export(format="tflite")`, from the
  existing `training/models/yolov8n.pt`).
- The app runs on-device TFLite inference (via `react-native-fast-tflite`)
  over the bundled images and measures wall-clock latency per inference.
- A single screen with a button that triggers the benchmark run over all
  bundled images, and displays aggregate latency stats — mean, median, min,
  max, standard deviation, in milliseconds — once the run completes.
- No camera capture, no network calls, no calls to the offload/FastAPI
  server — purely local, on-device inference timing.

## UI Mockup

See `ui-mocks/mobile-inference-benchmark.html` (idle and running states).
Visual styling is not a priority for this feature — the mockup exists to fix
the screen's structure (bundled-set info, run button, results block), not its
appearance.

## Out of Scope

- Camera / live capture.
- Any call to the FastAPI offload server, or simulating network conditions.
- The router/routing decision logic — this measures latency only, it does not
  make or evaluate a local-vs-offload decision.
- On-device accuracy measurement (no ground truth bundled or compared) —
  latency-only benchmark.
- Persisting results anywhere outside the device (results are read directly
  off the phone's screen, not written to Postgres or exported).
- Android support.
- Production polish — styling, exhaustive error handling, accessibility.
  Matches this project's stated "prototype bar, not production bar."

## Acceptance Criteria

- Given the app is installed on the user's physical iPhone via an EAS build,
  when the user taps "Run Benchmark", then the app runs local TFLite
  inference on every bundled image and displays mean/median/min/max/stdev
  latency in milliseconds on-screen.
- Given the benchmark has been run at least once, when reading the on-screen
  results, then the numbers are real, device-measured latencies — directly
  comparable to the desktop-measured numbers already gathered (the ~20-24ms
  figures from the ad hoc desktop timing script run earlier in this project).
- Given the app is built via EAS Build using the user's Apple Developer
  account, with no local Mac involved at any point, then the build succeeds
  and the resulting binary installs and runs on the physical iPhone.
- Given the app has no network/camera/server dependency, when run with the
  phone in airplane mode, then the benchmark still completes successfully.
