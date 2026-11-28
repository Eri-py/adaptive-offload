# Scan: Both Inference Paths on the Phone — Implementation Plan

## Summary

Builds the scan: choose a photo from the camera, the photo library or a bundled
sample; prepare it exactly as the research pipeline does; run it through the
on-device model and the server at the same time; and show both answers with
what each cost. No routing decision is made — both paths always run, which is
what produces the paired "which path won" record.

## Approach & Key Decisions

- **One picker package, not two.** `expo-image-picker` provides both the camera
  and the library through the system UI, so it replaces what would otherwise be
  `expo-camera` plus a picker, and carries one set of permission strings.
  `expo-camera` would mean building a camera viewfinder we do not need.
- **Preparation is a two-pass render.** `expo-image-manipulator`'s context API
  cannot report a photo's dimensions before loading it, and the centre-crop
  origin depends on them. So: render once to read width and height, compute the
  shorter side and the crop origin, then resize and crop in a second pass. The
  docs do not state how `compress` maps to JPEG quality, so Task 2 verifies the
  produced bytes against the bundled crops, which were made at PIL quality 85.
- **The payload is the file's bytes, not base64.** `saveAsync` returns a file
  URI; `expo-file-system`'s `File(...).arrayBuffer()` (already used by the
  spike) gives the bytes to POST directly. Base64 would inflate the payload by a
  third and break the correspondence with the measured `PAYLOAD_KB = 11.39`.
- **Both paths run concurrently and report independently.** The on-device result
  (~4 ms) must not wait for the server (~70 ms+). Each path has its own
  try/catch so an offload failure cannot cost the local result — the guidelines'
  rule that a silent failure here corrupts the latency record applies directly.
- **All native configuration lands in one task** so the user does a single EAS
  development build: the picker plugin with its permission strings, and the ATS
  exception without which iOS blocks requests to a tailnet address before they
  are sent. Everything after Task 1 is JavaScript and needs no further rebuild.
- **The server address is a field in the app, persisted to a file.** The spec
  requires changing it without a rebuild, which rules out `app.json` extras.
  `expo-file-system` is already a dependency, so no storage package is added.

## Out of Scope

Carried from the spec: any routing decision, latency budget, cascade or
confidence threshold; the background network monitor; recording scans on the
phone or in the database; changes to the server; fixing the relayed network
path; release-build measurements; changes to the model or its export; batch
modes; saving or sharing results.

Further deferred by this plan: no test runner is added to `app/` — there is
none today and `npx tsc --noEmit` is the whole gate, so the plan favours small
pure modules that a later task could test rather than adding a framework now.
No offline queue, no retry of a failed offload, and no persistence of past
scans.

## Dependencies and Configuration

- New packages, installed with `npx expo install` so SDK 57 resolves compatible
  versions (the newest published are SDK 58 and must not be pinned):
  `expo-image-picker`, `expo-image-manipulator`.
- `app.json` gains:
  - the `expo-image-picker` plugin with camera and photo-library permission
    strings,
  - `ios.infoPlist.NSAppTransportSecurity` allowing cleartext HTTP, because the
    server is reached at a `100.64.0.0/10` tailnet address which iOS does not
    treat as a local network.
- **One EAS development build is required** after Task 1 and before anything can
  run on the device: `npx eas-cli build --platform ios --profile development`.
  The user runs it. No later task needs another.
- No database migration. No change to `metro.config.js`, which already handles
  `.tflite`.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `app/package.json` | edit | Add picker and manipulator | New capabilities |
| `app/app.json` | edit | Picker plugin, permission strings, ATS exception | Native config, one rebuild |
| `app/services/serverConfig.ts` | add | Read/write the server base URL | Changeable without a rebuild |
| `app/services/prepareImage.ts` | add | Resize 232, centre-crop 224, JPEG, bytes | The payload contract |
| `app/services/localInference.ts` | add | Load the model once, run, derive confidences | On-device path |
| `app/services/offloadClient.ts` | add | `GET /health`, `POST /classify`, timeout | Server path |
| `app/services/samples.ts` | add | Bundled photos and their known species | Correctness without a human |
| `app/services/pickPhoto.ts` | add | Camera and library selection | One async boundary per source |
| `app/hooks/useScan.ts` | add | Run both paths, expose partial results | Orchestration out of the view |
| `app/hooks/useServerHealth.ts` | add | Reachability before scanning | Mockup's idle state |
| `app/screens/ScanScreen.tsx` | add | The screen in the mockup | Replaces the spike |
| `app/components/SourceSelector.tsx` | add | Samples / Camera / Library control | One component per file |
| `app/components/PathResultCard.tsx` | add | One path's answer, confidence, latency | Used twice per scan |
| `app/components/ScanSummaryCard.tsx` | add | Winner and payload size | The research view |
| `app/App.tsx` | edit | Render the scan screen | Spike removed |
| `app/hooks/useBirdSpike.ts` | delete | Superseded | Spike replaced |
| `app/screens/BirdSpikeScreen.tsx` | delete | Superseded | Spike replaced |

## Tasks

### Task 1 — Dependencies, native configuration and server address

- **Objective:** Install the two packages, add all native configuration in one
  go, and provide the changeable server address.
- **Files:** `app/package.json`, `app/app.json`, `app/services/serverConfig.ts`
- **Details:**
  - Install with `npx expo install expo-image-picker expo-image-manipulator`.
    Do **not** pin the newest published versions: those target SDK 58 and this
    app is on Expo SDK 57.
  - `app.json`: add the `expo-image-picker` plugin with camera and
    photo-library permission strings written for a bird-identification app.
    Keep the existing `react-native-fast-tflite` and `expo-asset` entries and
    `ITSAppUsesNonExemptEncryption` exactly as they are.
  - `app.json`: add an ATS exception under `ios.infoPlist` permitting cleartext
    HTTP. iOS blocks plain HTTP to Tailscale's `100.64.0.0/10` range because it
    is not treated as a local network, and Expo's default local-networking
    exception does not cover it. Without this the server is unreachable from the
    app regardless of JavaScript.
  - `serverConfig.ts`: read and write a base URL, persisted with
    `expo-file-system` (already a dependency — add no storage package), with a
    default of `http://100.97.204.46:8000`. Both functions wrapped so a missing
    or corrupt file falls back to the default rather than throwing.
- **Success criteria:**
  - `npx tsc --noEmit` passes from `app/`.
  - `app.json` parses and retains every pre-existing plugin and iOS key.
  - Reading the address before anything has been written returns the default.
  - **Stop and tell the user to run one EAS development build before any later
    task can be exercised on the device.**

### Task 2 — Image preparation

- **Objective:** Turn any photo into the exact bytes the research pipeline would
  produce.
- **Files:** `app/services/prepareImage.ts`
- **Details:**
  - Given a photo URI, return the prepared JPEG bytes, the byte count and the
    final pixel dimensions.
  - Sequence: render once with `ImageManipulator.manipulate(uri).renderAsync()`
    to read `width` and `height`; resize so the **shorter** side is 232, passing
    only that dimension and `null` for the other so the aspect ratio is kept;
    compute the centre-crop origin from the resized dimensions; `crop` to
    224x224; `saveAsync({ format: SaveFormat.JPEG, compress: 0.85 })`.
  - Read the saved file's bytes with `expo-file-system`'s
    `new File(uri).arrayBuffer()` — the same call the spike uses. Do **not** use
    base64: it inflates the payload by about a third and would break the
    correspondence with the measured 11.39 KB payload the cost model uses.
  - The context API is current; `manipulateAsync` is deprecated. Verify the
    installed version exposes `manipulate`/`useImageManipulator` and adapt if
    SDK 57's version differs.
  - **Verify the quality mapping:** `compress: 0.85` is not documented as
    equalling JPEG quality 85. Prepare one of the bundled photos in
    `app/assets/birds/` — which were produced at PIL quality 85 and are 3.8–17.8
    KB, mean 11.2 KB — and compare the byte count. Report the comparison; if it
    is wildly off, say so rather than adjusting silently.
- **Success criteria:**
  - `npx tsc --noEmit` passes.
  - The function returns 224x224 and a byte count in a plausible range for
    quality 85.
  - The comparison against a bundled crop's size is reported.

### Task 3 — On-device inference

- **Objective:** Run the exported model on prepared bytes and derive the same
  values the server returns.
- **Files:** `app/services/localInference.ts`
- **Details:**
  - Load `app/assets/models/small.tflite` once with the `core-ml` delegate and
    reuse it across scans; discard one warmup run before the first timed one, as
    the spike does.
  - Input is **NCHW** `[1, 3, 224, 224]` float32 — planar, not interleaved.
    Reuse `buildInputTensor`'s logic from `app/hooks/useBirdSpike.ts`: decode
    with `jpeg-js` (which yields RGBA, stride 4, alpha skipped), normalise per
    channel with ImageNet mean `[0.485, 0.456, 0.406]` and std
    `[0.229, 0.224, 0.225]`, and write planar at `channel * 224 * 224 + y * 224
    + x`. This is verified correct on device; do not rewrite it from scratch.
  - Output is `[1, 200]` logits. Softmax them, then return the top-1 species
    name and class index, the max probability, the top-2 margin and the top-3,
    plus the time for the `model.run()` call only.
  - Species names come from `app/assets/models/cub200-species.json`, indexed by
    class index.
  - Shape the result to match the server's fields so the two can be displayed by
    one component.
- **Success criteria:**
  - `npx tsc --noEmit` passes.
  - Timing covers only the model call, not decoding or tensor building.
  - The model is loaded once, not per scan.

### Task 4 — Server client

- **Objective:** Talk to the two endpoints, with failures that surface.
- **Files:** `app/services/offloadClient.ts`
- **Details:**
  - `GET /health` returning whether the server is reachable and its model
    loaded.
  - `POST /classify` sending the prepared bytes as the **raw body** with
    `Content-Type: image/jpeg` — not multipart, because the research's payload
    figure is the image bytes alone.
  - Measure the total round trip on the phone, and return it alongside the
    server's own reported `compute_ms`, so network and server cost stay
    distinguishable.
  - Response fields to type: `species`, `class_index`, `max_probability`,
    `top2_margin`, `top3[]`, `compute_ms`, `geometric_transform_applied`.
  - Apply a request timeout so an unreachable server ends the scan instead of
    hanging it, and return a typed failure (timeout, network error, HTTP error
    with status) rather than throwing a bare error.
  - Base URL comes from `serverConfig.ts`, read per request so a changed address
    takes effect without a restart.
- **Success criteria:**
  - `npx tsc --noEmit` passes.
  - A timeout produces a typed failure, not an unhandled rejection.
  - No multipart encoding anywhere.

### Task 5 — Photo sources

- **Objective:** Provide the three ways to choose a photo.
- **Files:** `app/services/samples.ts`, `app/services/pickPhoto.ts`
- **Details:**
  - `samples.ts`: expose the 15 bundled photos in `app/assets/birds/` with their
    known species, read from `app/assets/birds/expected.json` (fields `file`,
    `trueLabel`, `trueSpecies`). Metro needs literal `require()` paths, so list
    them explicitly and key by filename, as the spike does. These are already
    224x224 at quality 85, so note that preparing them again is close to a
    no-op.
  - `pickPhoto.ts`: launch the system camera and the system library through
    `expo-image-picker`, each wrapped so a denied permission or a cancelled
    pick returns a clear outcome rather than throwing. A refusal of one source
    must leave the others usable.
  - A photo from the camera or library has no known species; make that explicit
    in the returned shape rather than defaulting to something.
- **Success criteria:**
  - `npx tsc --noEmit` passes.
  - A cancelled pick and a denied permission are distinguishable from a success.
  - Bundled photos carry their known species; picked photos carry none.

### Task 6 — Scan orchestration

- **Objective:** Run both paths on one prepared photo and expose results as they
  arrive.
- **Files:** `app/hooks/useScan.ts`, `app/hooks/useServerHealth.ts`
- **Details:**
  - `useScan`: given a chosen photo, prepare it once, then start both paths on
    the **same bytes** concurrently. Expose the prepared photo, its byte count,
    each path's state (pending, done, failed with a reason) and, when both have
    settled, a summary naming the latency winner and — only where the species is
    known — the accuracy winner.
  - The on-device result must render as soon as it is ready, without waiting for
    the server. Do not await both together.
  - Each path keeps its own try/catch. An offload failure leaves the local
    result intact and records the reason.
  - `useServerHealth`: check reachability so the idle state can show it before a
    scan. A failed check is a normal state, not an error to surface loudly.
- **Success criteria:**
  - `npx tsc --noEmit` passes.
  - A rejected offload cannot reject the scan.
  - Both paths receive identical bytes.

### Task 7 — The screen

- **Objective:** Build the four states in the mockup and remove the spike.
- **Files:** `app/screens/ScanScreen.tsx`, `app/components/SourceSelector.tsx`,
  `app/components/PathResultCard.tsx`, `app/components/ScanSummaryCard.tsx`,
  `app/App.tsx`, `app/hooks/useBirdSpike.ts` (delete),
  `app/screens/BirdSpikeScreen.tsx` (delete)
- **Details:**
  - Follow `ui-mocks/final-demo-scan.html`: idle (source selector, empty frame,
    Scan disabled, server reachability shown), running (prepared photo with its
    dimensions and byte count, on-device card done, server card waiting),
    result (ground truth when known, both cards marked correct or wrong, the
    winning card outlined, a summary with winner and payload bytes), and offload
    failed (error block, local result intact, server card marked failed).
  - Server latency is shown split into the server's model time and the network
    remainder, since those are separate terms in the research's cost model.
  - A field for the server address, so it can be changed on the device.
  - Existing palette: background `#fff`, cards `#f2f2f7`, muted `#6e6e73`,
    accent `#007aff`, correct `#248a3d`, wrong `#d70015`.
  - One component per file, filename matching the component, explicit
    `<Component>Props` interfaces, `StyleSheet.create` at the bottom of each
    file, no `any`.
  - Delete both spike files and point `App.tsx` at the scan screen.
- **Success criteria:**
  - `npx tsc --noEmit` passes.
  - No reference to the deleted spike files remains anywhere.
  - All four mockup states are reachable in the code.

### Task 8 — Static verification and device checklist

- **Objective:** Confirm the project builds clean, and hand the user a checklist
  for what only a device can prove.
- **Files:** none changed
- **Details:** `app/` has no test runner; `npx tsc --noEmit` is the entire
  automated gate, so this task runs it and then writes the device checklist into
  the feature's learnings: both paths return on a bundled photo; the on-device
  answer matches what the desktop model predicts for that photo; the payload
  byte count is close to the bundled crop's; the server result arrives after the
  local one; airplane mode leaves the local result intact and marks the server
  failed; and the camera and library paths each work once.
- **Success criteria:**
  - `npx tsc --noEmit` passes from `app/`.
  - No test was skipped, because none exists to skip — stated explicitly rather
    than implied.
  - The device checklist is recorded for the user's run.

## Open Questions

- Whether `compress: 0.85` produces bytes equivalent to PIL's quality 85 is
  undocumented. Task 2 measures it against the bundled crops. If it diverges
  materially, the payload figure the cost model uses would describe the
  simulation but not the app, and the discrepancy should be recorded in
  `findings/birds.md` rather than quietly corrected.
