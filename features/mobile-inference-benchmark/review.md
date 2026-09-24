# Review — Mobile Inference Benchmark

## Verdict

The implementation is structurally sound. The app is laid out the way the plan describes, and the TFLite calls match `react-native-fast-tflite@3.0.1`'s real API: `loadTensorflowModel(source, delegates[])` and `run(ArrayBuffer[])`. The input tensor is built with the right shape and layout (NHWC float32 `[1,640,640,3]`, RGB, normalised to [0,1]), and every failure path shows up as a visible error. Two problems stop it from meeting the spec as written, and both are cheap to fix:

- **B1:** The benchmark has no warm-up inference. The first timed sample includes one-off initialisation cost, which skews mean/max/stdev over only 15 samples. The Python side fixed exactly this defect in an earlier review.
- **B2:** The build command the implementation tells the user to run (`--profile development`) produces a dev-client/Debug build. That build loads its JS from a Metro server, so it cannot run in airplane mode (AC4) and is not a representative standalone binary. The `preview` profile, which already exists, is the right one.

There is also an unflagged design drift: the Core ML delegate is enabled in the build config, but the hook passes `[]`, so inference runs CPU-only (S1). And the "matches the Python-side timing" claim is wrong: the desktop number includes preprocessing and NMS (S2).

## Acceptance Criteria

1. **Tapping "Run Benchmark" runs inference on every bundled image and shows mean/median/min/max/stdev in ms. PARTIAL (structurally met).** The button calls `runBenchmark` (`app/screens/BenchmarkScreen.tsx:35-39`). The hook loops over all 15 `require`d images (`app/hooks/useBenchmark.ts:16-32`, `:77-87`), and `computeStats` (`:146-161`) produces all five stats, which the screen renders at `BenchmarkScreen.tsx:58-82`. The stats maths is correct: sorted median with even/odd handling, and population stdev. On-device execution is unverified, which was accepted up front. Marked PARTIAL only because of B2: the documented build path does not give a standalone app.
2. **The numbers are real device latencies, directly comparable to the ~20-24 ms desktop figures. PARTIAL.** The latencies are real `performance.now()` deltas around `model.run` (`useBenchmark.ts:79-86`). But the first sample includes cold-start cost (B1). The timed scope is also narrower than the Python side's, which covers JPEG read, letterbox, forward pass and NMS (S2). The delegate choice (CPU only) is implicit and not shown on screen (S1).
3. **The EAS build succeeds and the binary installs and runs on the iPhone. PARTIAL.** `app/eas.json` is valid (checked with `@expo/eas-json`) and `app.json` wires the TFLite plugin. However, the recommended `development` profile sets `developmentClient: true` while `expo-dev-client` is not a dependency (`app/package.json:5-15`), so the build will not run without a Metro server (B2). The auto-generated bundle ID `com.anonymous.mobile-inference-benchmark` (`app/app.json:11`) may fail App ID registration (S4).
4. **The benchmark completes in airplane mode. PARTIAL.** The code itself has no network, camera or server dependency. Model and images are bundled through Metro assets, and `Asset.downloadAsync()` resolves bundled assets locally in a release build. But a `development`-profile build needs Metro at launch, so it fails in airplane mode (B2). A `preview` build would meet this criterion.

## Scope

The diff matches the plan's Files Changed table. The additions outside the table are all justified and flagged in learnings.md:

- `app/metro.config.js`: needed for `.tflite` assets.
- `app/index.ts`, `app/.gitignore`, `app/assets/icon.png`, `app/assets/splash-icon.png`: Expo template files.
- `app/package-lock.json`.
- `app/types/jpeg-js.d.ts`: the DefinitelyTyped package targets the wrong API version.
- The `expo-asset` plugin entry in `app.json`.
- `ui-mocks/mobile-inference-benchmark.html`: a spec artifact.

New runtime dependencies not listed in the plan: `expo-asset`, `expo-file-system`, `jpeg-js` and `react-native-nitro-modules`. All are necessary. `training/pyproject.toml` is correctly unchanged, because the ultralytics pin was reverted. No unrelated changes.

## Blockers

#### B1 — No warm-up inference; the cold-start sample is included in the stats

- **File:** `app/hooks/useBenchmark.ts:70-87`
- **Issue:** The first `model.run()` pays one-off costs (TFLite/XNNPACK weight packing and allocation, plus Core ML compilation if S1 turns that delegate on). That sample is recorded like any other, which inflates mean/max/stdev over n=15. This is the same defect the Python side fixed as review finding B1 (`training/datagen/simulate/yolo_inference.py`, the discarded warm-up call). Without the fix the phone numbers are not comparable to the desktop ones.
- **Fix:** Right after the model loads (or on every run), do one untimed `await model.run([...])` on a zero-filled `Float32Array(640*640*3)` (or the first image's tensor) before the timed loop. Optionally, show "1 warm-up discarded" in the bundled-set card.
- **Decision:** Accepted — addressed in "Address B1: warm-up inference before the timed loop"

#### B2 — The recommended `development` profile gives a Metro-dependent Debug build that breaks AC4

- **File:** `app/eas.json:7-13` (and the next-step instruction in progress.md, Task 6)
- **Issue:** `developmentClient: true` builds a dev client in the Debug configuration, and `expo-dev-client` is not even in `app/package.json`. Such a build fetches its JS bundle from a running `expo start`, so it cannot launch in airplane mode (AC4). It also runs JS in `__DEV__` mode, which is not a representative binary. Depending on the answer to EAS's interactive prompt, it either installs `expo-dev-client` or produces an app that shows "No bundle URL present".
- **Fix:** Point the user to `eas build --platform ios --profile preview`, which uses internal distribution, the Release config and an embedded JS bundle. Mention that internal distribution first needs `eas device:create` to register the iPhone's UDID. Either remove the `development` profile or add `expo-dev-client` if a dev-client build is actually wanted. Put the command in a short `app/README.md` or in the export-script docstring so it is not lost in progress.md.
- **Decision:** Accepted — addressed in "Address B2: build with the preview profile, not development"

## Suggestions

#### S1 — Core ML delegate is enabled in the build but never used; inference runs CPU-only

- **File:** `app/hooks/useBenchmark.ts:72`
- **Issue:** `loadTensorflowModel(..., [])` selects the default CPU delegate (per `Tflite.nitro.d.ts`: "If delegates is empty, the default CPU delegate will be used"). Meanwhile `app.json:14-19` sets `enableCoreMLDelegate: true`, which the plan chose "for iOS GPU acceleration". Nothing records whether CPU-only was a deliberate choice, and the screen does not say which delegate the numbers came from.
- **Fix:** Make the delegate an explicit named constant and show it on screen (for example "Delegate: CPU"). Better still, run the benchmark once per delegate (`[]` and `['core-ml']`, each with a warm-up) and report both rows: CPU-vs-CPU is the fair comparison with the desktop, and Core ML is what a real local path would use.
- **Decision:** Accepted — addressed in "Address S1: benchmark both CPU and Core ML delegates explicitly"

#### S2 — The timed scope differs from the Python side, but the docstring claims they match

- **File:** `app/hooks/useBenchmark.ts:51-57`
- **Issue:** The docstring says the timing matches `yolo_inference.py`'s approach. In fact the Python side times `model(image_path)`, which includes JPEG read/decode, letterbox, the forward pass and NMS. The app times only the raw forward pass. The desktop ~20-24 ms figure therefore covers more work than the phone number, and comparing them directly (AC2) biases the result in the phone's favour.
- **Fix:** Correct the docstring to say "raw forward pass only". Note the scope difference in learnings.md. For a like-for-like comparison, compare against the desktop's `results[0].speed['inference']`, or time the same `.tflite` through `tf.lite.Interpreter.invoke()` on the desktop.
- **Decision:** Accepted — addressed in "Address S2: correct the timed-scope claim vs. the Python side"

#### S3 — The export script adds a new mypy error to `training/`

- **File:** `training/export_tflite_model.py:46`
- **Issue:** `mypy export_tflite_model.py` reports `Module "ultralytics" does not explicitly export attribute "YOLO" [attr-defined]`. That is a new error, not a pre-existing one as progress.md implies. The coding guidelines require mypy to run clean.
- **Fix:** Copy the existing suppression pattern from `datagen/simulate/yolo_inference.py`: a one-line why-comment plus `# type: ignore[attr-defined]`.
- **Decision:** Accepted — addressed in "Address S3: silence the ultralytics YOLO attr-defined mypy error"

#### S4 — Placeholder `com.anonymous.*` bundle identifier

- **File:** `app/app.json:11`
- **Issue:** `expo prebuild` auto-generated `com.anonymous.mobile-inference-benchmark`. App IDs are globally unique across Apple teams, and `com.anonymous.<slug>` is Expo's default pattern, so EAS's App ID registration can fail if someone else already claimed it. It is also not an identifier the user owns.
- **Fix:** Change it to a reverse-domain ID the user controls, such as `com.<username>.adaptiveoffload.bench`.
- **Decision:** Accepted — addressed in "Address S4: use an owned bundle identifier"

## Nitpicks

#### N1 — Comments exceed the guideline's one- to two-line limit

- **File:** `app/hooks/useBenchmark.ts:12-15`, `:51-57`, `:80-83`, `:100-107`; `app/screens/BenchmarkScreen.tsx:5-6`; `app/metro.config.js:6-9`
- **Issue:** `.claude/coding-guidelines.md` "Comments" says one line under ~100 chars, two only when truly needed. Several of these blocks run 4-7 lines and partly restate the code.
- **Fix:** Trim each to a single "why" line. For example, `// Metro needs literal require() paths, so images are listed explicitly.`
- **Decision:** Accepted — addressed in "Address N1: trim comments to the one-line guideline"

#### N2 — Image count is hard-coded in two places

- **File:** `app/screens/BenchmarkScreen.tsx:7`
- **Issue:** `BUNDLED_IMAGE_COUNT = 15` duplicates `BUNDLED_IMAGE_MODULES.length` in the hook. Adding or removing an image would make the screen's label wrong without any error.
- **Fix:** Export the count (or `BUNDLED_IMAGE_MODULES.length`) from `useBenchmark.ts`, or return it from the hook, and use that value in the screen.
- **Decision:** Accepted — addressed in "Address N2: derive the image count from the bundled image list"

#### N3 — `eslint-disable` comment with no ESLint configured

- **File:** `app/metro.config.js:1`
- **Issue:** `// eslint-disable-next-line @typescript-eslint/no-var-requires` does nothing: `app/` has no ESLint setup, and a `.js` file would not hit that rule anyway.
- **Fix:** Delete the line.
- **Decision:** Accepted — addressed in "Address N3: drop the dead eslint-disable comment"

#### N4 — Export-script docstring is inaccurate on two points

- **File:** `training/export_tflite_model.py:3-4`, `:36-37`, `:83-88`
- **Issue:** It says the rationale is in `spec.md`, but it is in `implementation.md`. It also says `bundle_sample_images()` can run from the shared venv, but `__main__` always runs `export_tflite_model()` first, which fails there.
- **Fix:** Point to `implementation.md`. Either add a `--images-only` argument (or split `__main__`) or drop the shared-venv claim.
- **Decision:** Accepted — addressed in "Address N4: fix the export script docstring"

#### N5 — Running state has no per-image progress, unlike the mockup

- **File:** `app/screens/BenchmarkScreen.tsx:41-46`
- **Issue:** The mockup shows "Running inference 9 / 15…". The screen shows a generic "Running benchmark…". On-device JPEG decoding in pure JS makes a run take several seconds, so progress would be useful.
- **Fix:** Add a `progress` counter to the hook's state, update it after each image, and render `Running inference {n} / {total}…`.
- **Decision:** Declined — Cosmetic; the spec says styling and polish are not a priority.

#### N6 — Executable bit set on non-executable files

- **File:** `app/.gitignore`, `app/App.tsx`, `app/index.ts`, `app/tsconfig.json` (mode 100755)
- **Issue:** The template files were committed with `+x`, a leftover from the Windows/WSL toolchain.
- **Fix:** `git update-index --chmod=-x` on those four files.
- **Decision:** Accepted — addressed in "Address N6: clear the executable bit on app template files"

## Tests

No automated tests were added for `app/`, which matches the plan's decision. Verification consisted of:

- `tsc --noEmit`
- `expo start` reaching the ready state
- `expo prebuild` confirming the Core ML Podfile flag
- `@expo/eas-json` schema resolution
- Python-side `tf.lite.Interpreter` shape checks

That is reasonable given there is no device. `computeStats` and `resizeAndNormalize` are pure functions and would be trivial to unit-test. I checked both by reading them and they are correct: median handles even/odd counts, stdev is the population (ddof=0) form, and the NHWC index maths and alpha-skip via `sourceChannels` are right. Adding Jest just for them is not warranted at this bar. The main gap is not a test at all: the first real EAS build will be the first time the model loads. Doing B1 and B2 before that build avoids spending a build cycle on numbers that would have to be thrown away.

## Recommended Decisions

- **B1** — Accept — A few lines of code. Without it, the headline numbers are skewed by a cold start the Python side already excludes.
- **B2** — Accept — The documented build path cannot meet AC4 and gives a non-representative Debug binary. Switching to `preview` is a one-line change to the instructions.
- **S1** — Accept — Making the delegate explicit and visible costs little and removes ambiguity about what the numbers measure. Running both delegates is the best version.
- **S2** — Accept — The docstring is factually wrong about comparability, which is the whole point of this feature (AC2).
- **S3** — Accept — One-line fix that restores clean mypy per the guidelines.
- **S4** — Accept — Cheap to change now, and avoids a likely App ID registration failure on the first EAS build.
- **N1** — Accept — Direct coding-guidelines violation, trivial to trim.
- **N2** — Accept — Removes an easy-to-miss duplication at no cost.
- **N3** — Accept — Deleting a dead line.
- **N4** — Accept — Doc accuracy for whoever reruns the export; small edit.
- **N5** — Decline — Cosmetic. The spec says styling and polish are not a priority.
- **N6** — Accept — Trivial cleanup.
