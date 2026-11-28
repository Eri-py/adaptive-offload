
## Task 1
- `expo install` resolved expo-image-picker ~57.0.22 and expo-image-manipulator ~57.0.23 (SDK 57 line).
- expo-file-system 57 `File` API: `new File(Paths.document, name)`, `.exists`, `.create()`, `.write(string)`, `await .text()`. No legacy readAsStringAsync needed.
- ATS: `NSAllowsArbitraryLoads` chosen because per-domain `NSExceptionDomains` entries match hostnames, not IP literals/CIDR ranges, so they cannot cover 100.64.0.0/10. Broad but fine for a course demo.
- Requires one new EAS development build (new native modules + Info.plist changes).

## Task 3
- `localInference.ts` caches the load+warmup as a shared promise (cleared on failure so a retry works); self-contained, no import from useBirdSpike.
- `compute_ms` brackets only `await model.run()`; decode, tensor build and softmax are outside it. Untested on device here.

## Task 2
- SDK 57 expo-image-manipulator exposes `ImageManipulator.manipulate(uri)` (chainable `resize`/`crop`, `renderAsync()` -> `ImageRef` with `width`/`height`/`saveAsync`), plus `useImageManipulator` and deprecated `manipulateAsync`. `resize` accepts `null` for the other dimension.
- Centre-crop origin uses Math.round of (resized - 224)/2 (matches torchvision CenterCrop); long side after resize is computed locally with Math.round and may differ by 1px from native rounding, shifting the crop by at most 1px.
- `compress: 0.85` vs PIL quality 85 is unverified (no device): bundled crops are 3.8-17.8 KB, mean 11.2 KB. Check on device that a prepared bundled photo lands near its `expected.json` bytes.

## Task 4
- `offloadClient.ts` returns `ClientResult<T>` (ok/failure union) and never throws; failure kinds: timeout, network, http (status + server `detail`), invalid-response. Timeout 5000 ms via AbortController, timer cleared in `finally`; a `timedOut` flag tells abort-by-timer from other network errors.
- Round trip is `Date.now()` around fetch until headers arrive (excludes local JSON parse); `serverComputeMs` is the server's `compute_ms`, so network is the difference. Body is a raw `Uint8Array`; never exercised against the live server here.

## Task 5
- expo-image-picker 57: `launchCameraAsync`/`launchImageLibraryAsync` return a union where `canceled: true` has `assets: null`; success has `assets[]` with `uri`. `request*PermissionsAsync` return `PermissionResponse` with `granted` and `canAskAgain`. `mediaTypes` takes `['images']` (the `MediaTypeOptions` enum is the older form).
- Picks return a `PhotoOutcome` union (picked/cancelled/denied/error) instead of throwing; bundled samples reuse it via `loadSample` so the hook handles one shape. `trueSpecies` is `null` for camera/library, never a default.
- Unverified on device: whether a camera/library URI is accepted directly by expo-image-manipulator.
- Added `preloadLocalModel()`: joins the same cached load+warmup promise, resolves a `PreloadStatus` instead of rejecting; failure clears the cache so a scan can retry.
- Task 2 revision: crop origin now comes from the real resized `ImageRef` dimensions via a chained `manipulate(resizedRef)` (SDK 57 types accept `SharedRef<'image'>`), replacing the locally predicted long side. Untested on device.

## Task 6
- Both paths share one `new Uint8Array(prepared.jpegBytes)` view (no copy, no second prepare); `runLocalInference` accepts it and `classifyImage` copies it itself for fetch.
- Local throws and offload returns a result; each is wrapped in its own try/catch and normalised to `PathState`, so `Promise.all` over them cannot reject. Each path patches state in its own `.then`, so local renders before the server answers.
- Latency compares local `compute_ms` to offload `roundTripMs` (ties to local). Accuracy compares `class_index` to `trueSpecies.label` only when known; if a path failed the summary says `unavailable`. A scan id ref discards results from superseded scans.
- Task 6 revision: latency now compares like with like. `LocalInferenceResult.total_ms` (decode + tensor + model.run + softmax, started after the one-time model load) is the local `latencyMs`; `compute_ms` stays model-only and becomes `PathResult.modelMs` (offload: server compute). Replaces `serverComputeMs`.

## Task 7
- `git rm` of the last file in `screens/` removes the directory; recreate it before writing the new screen.
- The screen keeps the chosen photo and selected sample file in its own state (a cancelled pick keeps the previous photo); `useScan.reset()` is called on every real pick so stale results never sit under a new photo.
- Winner outline rule: single accuracy winner, else latency winner, else the path that answered. Mockup state 4 (local outlined, server failed) falls out of the last rule.
- Not run on device: layout is unverified against the mockup visually; the health tag shows no latency (the hook exposes none), unlike the mockup's "34 ms".

## Task 8 — device checklist (orchestrator)

`app/` has no test runner and no working ESLint config (`npx eslint` fails: no
`eslint.config.*`), so `npx tsc --noEmit` is the entire automated gate and it
passes. No test was skipped because none exists to skip. `server/` (32) and
`database/` (1) are unaffected and still pass.

Everything below can only be settled on the device, against the new development
build. Run `npx expo start --tunnel` from `app/`.

1. **Payload size, the open question.** Scan a bundled sample and read the byte
   count on screen. `1116.jpg` was 11,926 bytes at PIL quality 85; the 15 crops
   average 11,216. If `compress: 0.85` lands far from that, the app's payload is
   not the 11.39 KB the cost model assumes, and that belongs in
   `findings/birds.md` rather than being tuned away quietly.
2. **On-device correctness.** For a bundled sample, the device prediction should
   match `expectedSpecies` in `app/assets/birds/expected.json` — that is what the
   desktop model predicted from the same JPEG bytes. Near-ties may differ because
   `jpeg-js` and PIL decode differently, worth up to 4.8e-02 in max probability.
3. **The ordering claim.** The on-device card must appear visibly before the
   server card. If both appear together, results are not streaming as intended.
4. **Latency split.** The device card should read roughly "4 ms model · N ms
   decode"; the server card roughly "35 ms model · 83 ms network". A server model
   time near 6-7 ms instead of ~35 would mean the GPU was still at boost clocks
   from a previous request — see the arrival-pattern finding.
5. **Offload failure.** Put the phone in airplane mode and scan: the device
   result must survive, the server card must show a timeout reason, and the scan
   must end rather than hang (5 s timeout).
6. **The two picked sources.** Camera and library each once. Confirm a cancelled
   pick returns to idle silently and a denied permission leaves the others
   usable. Confirm a picked photo reports its species as unknown rather than
   showing a correctness verdict.
7. **Non-224 input.** A camera photo is not 224x224, so preparation must crop it.
   The server's `geometric_transform_applied` should be false either way, since
   the phone sends an already-cropped 224 image.
8. **Server address.** Change it to something invalid and confirm the error shows
   under the field; change it back and confirm the reachability tag recovers.

**The demo photo:** `1116.jpg` — ground truth Eastern Towhee, where the device
model says American Redstart and the server says Eastern Towhee at 0.80. One
scan that shows why the offload path earns its keep.

## Review fixes B1, S7, S8
- B1: `prepareImage` returns the original file bytes untouched when the source is exactly 224x224 (mirrors the server's preprocess rule). Bundled samples no longer exercise resize/crop; camera and library photos are its only coverage.
- S7: resize now chains from the rendered `source` ImageRef instead of re-opening the URI, removing one full decode per scan.
- S8: `cropOrigin` clamps to >= 0 and a resized side under 224 throws a message naming the dimensions. Geometry is in exported pure functions (`shortSideResize`, `cropOrigin`, `isAlreadyPrepared`).
- N2 (Math.round vs torchvision banker's rounding, at most 1 px) was considered and declined; rounding left as is.

## Review fixes (S1-S6, N1, N3, N7)

- **Reordering two statements does not put the request on the wire first.** `classifyImage`
  awaits a native file read (`readServerUrl`) before `fetch`, so local's already-resolved
  continuation would still run its 80 ms decode first. `classifyImage` now takes an
  `onDispatched` callback fired right after `fetch(...)` is called, and `useScan` awaits it
  before starting the local path. `runOffloadPath` also fires it in `finally`, so a request that
  fails before dispatch cannot hang the scan.
- **One clock: `performance.now()`.** Monotonic and fractional, already used by the local path.
- **The offload remainder is `network + server overhead`.** It includes server decode and
  normalisation. `findings/birds.md` should use the same wording.
- **The review's S5 is `usedDefault`, not the preload status.** The preload status belongs to S1.
  Both are now shown: `modelError` from `useScan`, and a "using the default" line on the address field.
