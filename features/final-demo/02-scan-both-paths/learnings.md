
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
