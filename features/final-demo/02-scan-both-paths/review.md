# Review — Scan: Both Inference Paths on the Phone

## Verdict

The feature does what spec 02 asked for: three photo sources, one preparation
step, both paths on the same bytes, results that stream independently, a summary
that names both winners, and an offload failure that never costs the local
result. The module boundaries are right — `prepareImage`, `localInference` and
`offloadClient` are plain services, orchestration is in `useScan`, and the
screen is presentational. `npx tsc --noEmit` passes, no `any` appears anywhere in
`app/`, and no routing logic leaked in (grep for budget/threshold/router/deadline/
cascade across `app/` is empty), so the spec-03 boundary held.

One acceptance criterion is measurably not met, and it is the accuracy-critical
one: the bundled samples are *already* the research pipeline's 224x224 q85 crops,
and the app runs the full `Resize(232) + CenterCrop(224) + JPEG` transform over
them a second time. The bytes therefore differ from the research pipeline's by
1.55x, and the second pass is a resample as well as a re-encode — it discards the
outer 3.5% of each edge (224 -> 232 -> crop 224 keeps 224/232 of the frame) and
moved the server's confidence on `1116.jpg` from 0.804 to 0.70. That is one
blocker (B1). `findings/birds.md` records the discrepancy honestly and flags the
double-generation caveat itself, which is exactly right — but the fix belongs in
the code as well as in the write-up, because the whole point of the bundled set
is that it is the research pipeline's own output.

Everything else is advisory. The geometry is otherwise correct, the identical-bytes
guarantee holds with no copy and no mutation, and the state machine has no stuck
path I can find. The latency comparison is close to symmetric but has three small
honesty problems (S2, S3, S4), and one failure reason is computed and then never
displayed (S1).

## Acceptance Criteria

### Choosing and preparing

- **Bundled source offers a photo whose species the app knows — MET.**
  `app/services/samples.ts:37-40` builds `BUNDLED_SAMPLES` from
  `expected.json`'s `trueLabel`/`trueSpecies`, and
  `app/components/SourceSelector.tsx:50-63` lists all 15 with the species as the
  row label.
- **Camera/library photo is scannable and the app says no correct answer is
  known — MET.** `app/services/pickPhoto.ts:58` sets `trueSpecies: null` with no
  default, `app/screens/ScanScreen.tsx:24-27` maps that to `'unknown'` so no
  verdict is shown, and `ScanScreen.tsx:180-184` prints "Unknown for this photo,
  so no correct/wrong verdict". Note it is stated at result time (the block is
  gated on `settled`), not while the photo sits unscanned.
- **A refused permission leaves the bundled photos scannable — MET.**
  `pickPhoto.ts:44-47` returns `{kind:'denied'}` and touches nothing else;
  `ScanScreen.tsx:46-48` states "The other sources still work." The sample tab
  is independent of both permissions.
- **Prepared photo shown at 224x224 with the payload size in bytes — MET.**
  `ScanScreen.tsx:147-154` overlays `prepared.width × prepared.height` and a KB
  pill; `app/components/ScanSummaryCard.tsx:65-69` prints the exact byte count
  (`18,489 bytes`). Both appear as soon as preparation finishes, before either
  path answers.
- **Prepared bytes match the research pipeline's in size and prediction —
  NOT MET.** Measured on device: 18,489 B vs 11,926 B for `1116.jpg` and
  18,319 B vs 11,608 B for `1176.jpg` (`findings/birds.md`, the payload table).
  The species still matched, but the server's max probability moved 0.804 ->
  0.70, so "produce the same prediction" holds only on the argmax. Cause is
  `app/services/prepareImage.ts:16-39` being applied to an input that the
  research pipeline had already transformed — see B1.

### Scanning

- **Both paths run on the same bytes — MET.** `app/hooks/useScan.ts:186` takes
  one `new Uint8Array(prepared.jpegBytes)` view (no copy) and hands the same
  reference to both calls at `:196` and `:200`. `localInference.ts:79` reuses the
  view as-is; `offloadClient.ts:134-137` copies it for `fetch` without touching
  the original; `jpeg-js` does not mutate its input. Nothing re-prepares or
  re-derives per path.
- **On-device result reports species, confidence, top-2 margin and model time —
  MET.** `localInference.ts:120-145` derives all four;
  `PathResultCard.tsx:63-72` renders species, `confidence 0.84 · margin 0.31`,
  the total, and the `model` split.
- **Server result reports the same three plus server model time and round trip —
  MET.** `offloadClient.ts:153-160` returns `roundTripMs` and `serverComputeMs`
  separately, mapped to `latencyMs`/`modelMs` at `useScan.ts:115-116`. The
  server's `compute_ms` is the forward pass only
  (`server/api/services/classification/classify.py:38-42`), which is the same
  bracket as the local `model.run()` — the model-vs-remainder split is derived
  correctly on both sides. The remainder's *label* is the problem, not the
  number (S2).
- **Local result shown without waiting for the server — MET.**
  `useScan.ts:196-203` publishes each path in its own `.then`; nothing awaits
  both before rendering. `Promise.all` at `:205` only computes the summary.
- **Each path marked correct or wrong where the species is known — MET.**
  `ScanScreen.tsx:24-27` and `PathResultCard.tsx:32-34`.
- **Summary names the latency winner, and the accuracy winner when known —
  MET.** `useScan.ts:126-147` plus `ScanSummaryCard.tsx:56-64`; the
  `unknown`/`unavailable` cases are distinct strings rather than a silent blank.
- **Scanning the same bundled photo twice gives the same species and
  confidences — MET by construction.** No randomness on either side: the local
  path is a deterministic decode + fixed normalisation + `model.run`, and the
  server runs under `torch.inference_mode()` with fixed weights. Preparation is
  re-run per scan but is deterministic for identical input bytes. This is the one
  criterion the recorded device run did not exercise (one scan per sample).

### Failure

- **Offload failure keeps the local result and shows a reason — MET.**
  `useScan.ts:102-123` wraps the offload path in its own try/catch and
  `classifyImage` never throws; `ScanScreen.tsx:188-193` renders the reason plus
  "The on-device result is unaffected." The local path cannot be rejected by the
  offload path — `Promise.all` is over two promises that both resolve to a
  `PathState`.
- **A non-responding server ends the scan by itself — MET.** 5 s
  `AbortController` timeout at `offloadClient.ts:71-92`, with `clearTimeout` in
  `finally` and a `timedOut` flag that distinguishes a timeout abort from any
  other network error.
- **Reachability shown before any scan — MET.** `useServerHealth.ts:41-47` checks
  at mount; `ScanScreen.tsx:228-237` shows checking/reachable/unreachable with
  the failure reason underneath. See S6 for staleness after a failed scan.
- **A changed address is used without a rebuild — MET.**
  `offloadClient.ts:67` calls `readServerUrl()` per request, and
  `serverConfig.ts:30-39` persists to the documents directory. No rebuild and no
  restart needed.

## Scope

The diff matches the plan's Files Changed table exactly — every added, edited and
deleted path is listed there, and nothing outside it except three expected items:
`app/package-lock.json` (a consequence of the two installs),
`ui-mocks/final-demo-scan.html` (the mockup the spec references), and the
`findings/birds.md` entry, which the plan's Open Questions explicitly directed to
be written there rather than "quietly corrected".

Two commits sit beyond the eight tasks and both are in scope: `a5c4dc2` removes a
`maxHeight` that was hiding 10 of 15 samples (`SourceSelector.tsx:83-84`), and
`f06bf65` is the findings entry. Both spike files are gone and no reference to
`useBirdSpike`/`BirdSpikeScreen` survives anywhere in `app/`.

Out-of-scope boundaries held: no routing, budget, threshold or deadline code; no
on-phone scan persistence (the only file written is the server address); no
change under `server/` or `database/`; the `app.json` edits are confined to the
picker plugin, its two permission strings and the ATS exception, with every
pre-existing plugin and iOS key intact.

## Blockers

#### B1 — Bundled samples are prepared a second time, so neither path sees the research pipeline's bytes

- **File:** `app/services/prepareImage.ts:16` (with `app/services/samples.ts:42`)
- **Issue:** The 15 files in `app/assets/birds/` are already the research
  pipeline's `Resize(232) + CenterCrop(224)` q85 crops, so running the same
  transform over them upscales 224 -> 232 and crops back to 224, keeping only the
  central 96.5% of the frame and adding a second JPEG generation. Measured
  consequence: 18,489 B vs 11,926 B (1.55x) and the server's confidence on
  `1116.jpg` dropping 0.804 -> 0.70. This fails the acceptance criterion that the
  app's bytes match the research pipeline's in size, and it means the recorded
  payload ratio conflates three effects (`compress` vs PIL quality, a resample,
  and a second encode) that the findings entry then has to caveat. The comment at
  `samples.ts:42` ("close to a no-op") is not accurate.
- **Fix:** In `prepareImage`, skip the resize and crop when the rendered source
  is already exactly `CROP_SIZE x CROP_SIZE`, and read the bytes straight from
  the source (for a bundled sample, read the asset file's bytes directly). The
  bundled path then sends byte-identical input to both models and the criterion
  is met; the `compress: 0.85` question narrows to camera photos, where it is the
  real question, and `findings/birds.md` can drop the double-generation caveat
  for the sample numbers. Trade-off to note in `learnings.md`: the sample path
  then no longer exercises the resize/crop code, so the camera path becomes the
  only coverage of it (device checklist item 7 already covers that).
- **Decision:** Accepted — addressed in "Address B1, S7, S8: Stop re-transforming images that are already 224"

## Suggestions

#### S1 — A failed on-device path computes a reason that is never shown

- **File:** `app/components/PathResultCard.tsx:76`
- **Issue:** `useScan.ts:98` builds `reason` for a local failure, and
  `ScanScreen.tsx:188-193` renders the reason for the *offload* path only. A
  local failure shows the `failed` tag and the generic "No result recorded for
  this scan." — so a model-load or decode failure reaches the UI with its cause
  stripped, which is the silent-failure case the guidelines call out. Related:
  `useScan.ts:157` discards `preloadLocalModel()`'s `PreloadStatus`, a value
  built specifically so a load failure could be reported rather than thrown.
- **Fix:** Render `state.reason` in `PathResultCard` when `status === 'failed'`
  (the state is already a prop, so no plumbing is needed), and keep the preload
  error in `useScan` state so a model that never loads is visible before the
  first scan.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### S2 — The offload remainder is labelled "network" but includes the server's non-compute cost

- **File:** `app/screens/ScanScreen.tsx:208`
- **Issue:** `PathResultCard.tsx:69-72` prints `latencyMs - modelMs` under the
  caller's label, and the server card passes `remainderLabel="network"`. That
  remainder also contains PIL decode, `ToTensor`/`Normalize`, FastAPI dependency
  resolution and response serialisation — the server's own code says as much
  (`server/api/services/classification/request_log.py:53-57`). The research cost
  model has a separate transfer term, so "83 ms network" in the findings table
  overstates transfer by the server's overhead.
- **Fix:** Label it `network + server overhead` (or `round trip − model`), and
  use the same wording in `findings/birds.md`'s per-scan table.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### S3 — The two latencies being compared are measured on two different clocks

- **File:** `app/services/offloadClient.ts:77`
- **Issue:** The round trip uses `Date.now()` (wall clock, 1 ms granularity,
  movable by an NTP adjustment) while the local path uses `performance.now()`
  (monotonic, fractional) at `localInference.ts:78`. `useScan.ts:137` then decides
  the winner by comparing the two directly. For a latency record that is the
  project's output, both sides should come off the same monotonic clock.
- **Fix:** Use `performance.now()` for `startedAt`/`roundTripMs` in
  `offloadClient.ts:77,84`.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### S4 — The offload request is only dispatched after the local path's ~80 ms of blocking decode

- **File:** `app/hooks/useScan.ts:196`
- **Issue:** `runLocalPath` is called first and its `await loadModelOnce()`
  resolves from cache, so its continuation — the `jpeg-js` decode and the
  150k-pixel normalisation loop, measured at 75-89 ms — is queued ahead of the
  offload path's first continuation and blocks the JS thread. The server request
  therefore goes on the wire ~80 ms into the scan. `roundTripMs` itself is not
  inflated (the clock starts immediately before `fetch`), but the wall-clock
  arrival of the server answer is, and the response callback competes with
  whatever local work is still running.
- **Fix:** Start `runOffloadPath(bytes)` before `runLocalPath(bytes)` so the
  request is issued before the thread is blocked. The local card still renders
  first by a wide margin, so the demo's point is unaffected.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### S5 — `usedDefault` is computed and never read, so a corrupt config falls back silently

- **File:** `app/services/serverConfig.ts:18`
- **Issue:** `ServerUrlResult.usedDefault` exists precisely so a missing or
  unreadable config can be reported, but both callers (`ScanScreen.tsx:67` and
  `offloadClient.ts:67`) read only `.url`. An unreadable config writes
  `console.warn` and then presents the default address as if it had been stored —
  the user sees an address that is not the one they saved.
- **Fix:** Show it (e.g. "using the default address" under the field when
  `usedDefault` is true after a save has happened), or delete the field so the
  API does not promise something no caller uses.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### S6 — Server health is never re-checked after a failed offload

- **File:** `app/hooks/useServerHealth.ts:41`
- **Issue:** The check runs at mount and on "Save and check" only. After an
  offload timeout the screen can show the red "Offload failed" block while the
  server card below still reads `reachable`, which is the opposite of the spec's
  "a failure during a demonstration is anticipated rather than discovered".
- **Fix:** Call `recheck()` from `ScanScreen` when a scan settles with
  `scan.offload.status === 'failed'`.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### S7 — `prepareImage` decodes the source image twice

- **File:** `app/services/prepareImage.ts:18`
- **Issue:** `ImageManipulator.manipulate(uri).renderAsync()` at `:18` loads and
  decodes the file to read its dimensions, then `:22` loads and decodes the same
  file again to resize it. The crop stage already chains from the previous
  `ImageRef`, so the pattern is available.
- **Fix:** Chain the resize from the already-rendered ref:
  `ImageManipulator.manipulate(source).resize(...)`. Preparation is on the
  critical path of every scan, and for a full-resolution camera photo one decode
  is not cheap.
- **Decision:** Accepted — addressed in "Address B1, S7, S8: Stop re-transforming images that are already 224"

#### S8 — The crop is not clamped to the resized image's bounds

- **File:** `app/services/prepareImage.ts:27`
- **Issue:** The geometry is right whenever the native resize honours
  `{width: null}`/`{height: null}` and the ref's `width`/`height` reflect applied
  EXIF orientation. If either assumption fails for a camera photo — an
  orientation-tagged portrait reported as landscape would resize the *long* side
  to 232 and leave the short side at ~174 — `originY` goes negative and the crop
  exceeds the image, so the native call throws and the whole scan reports "Could
  not prepare the photo". The failure is surfaced rather than swallowed, but a
  whole source would be unusable. The camera path is the only one affected and
  the recorded device run covered bundled samples only.
- **Fix:** After the resize, assert `resized.width >= CROP_SIZE &&
  resized.height >= CROP_SIZE` and re-resize on the actual shorter axis if not;
  or clamp `originX`/`originY` to `>= 0` and raise a message that names the
  resized dimensions.
- **Decision:** Accepted — addressed in "Address B1, S7, S8: Stop re-transforming images that are already 224"

## Nitpicks

#### N1 — `style: object` where a style type belongs

- **File:** `app/components/PathResultCard.tsx:23`
- **Issue:** `tagFor` returns `{ label: string; style: object }`. It is not
  `any`, but `object` accepts anything non-primitive and defeats the type check
  on a style value.
- **Fix:** Type it `StyleProp<TextStyle>` from `react-native`.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### N2 — Centre-crop rounding differs from torchvision at the half-pixel case

- **File:** `app/services/prepareImage.ts:29`
- **Issue:** torchvision's `center_crop` uses `int(round((h - th) / 2.0))` with
  Python's banker's rounding, so an odd difference rounds *down* (`round(4.5)` is
  4); `Math.round(4.5)` is 5. For any resized dimension that is odd the crop sits
  1 px off the research pipeline's. Also, torchvision computes the long side with
  truncation (`int(size * long / short)`) while the native resize rounds, which
  can shift the frame by another pixel. Bundled samples are unaffected
  ((232-224)/2 is exact) and camera photos have no ground truth, so nothing
  measurable depends on it.
- **Fix:** Record it in `learnings.md` as a known sub-pixel divergence, or match
  the convention with `Math.floor((d - CROP_SIZE) / 2 + 0.5 - Number.EPSILON)`
  if exactness is ever wanted.
- **Decision:** Declined — at most 1 px, provably zero for the bundled set once B1 lands, and unobservable for photos with no ground truth; recorded in learnings.md instead.

#### N3 — `checkHealth` throws away the round trip it already measured

- **File:** `app/services/offloadClient.ts:122`
- **Issue:** `request()` returns `roundTripMs` for every call, but `checkHealth`
  discards it, so the health tag shows no latency where the mockup shows
  "34 ms" next to the address (`ui-mocks/final-demo-scan.html`, state 1).
- **Fix:** Return `roundTripMs` from `checkHealth`, expose it on
  `useServerHealth`, and render it in the server card.
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

#### N4 — The local path has no timeout, so a hung native call freezes the screen

- **File:** `app/hooks/useScan.ts:196`
- **Issue:** The offload path cannot hang (5 s abort), but if
  `loadTensorflowModel` or `model.run` never settles, `phase` stays `running`,
  `busy` stays true and `ScanScreen.tsx:116` keeps the Scan button disabled with
  no way out but a restart. Unlikely, and outside what the spec asked for (the
  "not indefinitely" requirement is about the server).
- **Fix:** Either race the local path against a generous timeout, or leave it and
  note the limitation.
- **Decision:** Declined — a hung native call is not a realistic demo risk, and racing the local path adds a branch to the number the research record depends on.

#### N5 — Each scan leaves a JPEG behind in the cache directory

- **File:** `app/services/prepareImage.ts:35`
- **Issue:** `saveAsync` writes to the cache directory and the file is never
  deleted after its bytes are read, so a demo session accumulates one file per
  scan. iOS reclaims the cache directory, so this costs nothing in practice.
- **Fix:** `new File(saved.uri).delete()` after `arrayBuffer()` resolves.
- **Decision:** Declined — iOS reclaims the cache directory; not worth code in a prototype.

#### N6 — The library path requests a permission the system picker does not need

- **File:** `app/services/pickPhoto.ts:29`
- **Issue:** The comment acknowledges that recent iOS needs no permission for the
  system picker, yet the code requests one and returns `denied` without
  launching, so a refusal blocks a path that would have worked.
- **Fix:** For the library, launch first and report `denied` only if the launch
  itself fails.
- **Decision:** Declined — deliberate and commented, the permission string already ships, and the demo grants access.

#### N7 — Two unhandled async boundaries in the component body

- **File:** `app/screens/ScanScreen.tsx:67`
- **Issue:** `readServerUrl().then(...)` has no `.catch` and no mounted guard,
  and `onPress={() => chosen !== null && startScan(chosen)}` at `:162` drops the
  returned promise. Neither function rejects today, so this is a latent rather
  than an actual failure — but the guidelines ask every async boundary to be
  wrapped with the failure surfaced, and a rejection in either place would leave
  the screen stuck with nothing on screen to explain it.
- **Fix:** Add `.catch` handlers that set `notice`/`addressMessage`, and mark the
  `onPress` promise explicitly (`void startScan(chosen)`).
- **Decision:** Accepted — addressed in "Address S1-S6, N1, N3, N7: Surface failures and make the timings honest"

## Tests

No tests were added and none were skipped, because `app/` has no runner and no
working ESLint config — `npx tsc --noEmit` is the entire automated gate and it
passes clean (verified in this review). `server/` (32) and `database/` (1) are
untouched by the diff. That matches the plan, which explicitly deferred adding a
framework, and it is the right call at this bar; the device run recorded in
`findings/birds.md` covers what a test could not.

Worth noting for later, since the plan promised "small pure modules that a later
task could test":

- `summarise` (`useScan.ts:126`) is exported and pure — the winner logic,
  including the tie-to-local rule and the `unknown`/`unavailable` cases, is
  testable today with no native modules. Good shape.
- The two accuracy-critical pure functions are *not* reachable:
  `summarize` (`localInference.ts:120`, the softmax/top-3/margin derivation) and
  `buildInputTensor` (`:105`, the NCHW planar layout and ImageNet
  normalisation) are module-private. Both are plain array maths with no native
  dependency, so exporting them would make the most consequential code in the
  local path unit-testable.
- `prepareImage`'s geometry (the shorter-side choice and the crop origin,
  `prepareImage.ts:19-33`) is inlined inside an `async` function that awaits the
  native module three times, so it cannot be tested without a device. Extracting
  `shortSideResize(w, h)` and `cropOrigin(w, h)` as pure helpers would make the
  part most likely to be wrong — and the part B1 and N2 are about — checkable
  from a desk.
- Coverage gap that only a device can close, and is still open: a repeat scan of
  the same bundled photo (the determinism criterion), and the camera path with an
  EXIF-rotated photo (S8).

## Recommended Decisions

- **B1** — Accept — an acceptance criterion is measurably failing, the fix is a
  single early-return in `prepareImage`, and it removes the confound that
  `findings/birds.md` currently has to caveat.
- **S1** — Accept — the reason is already computed and already a prop; dropping it
  is the exact silent-failure the guidelines forbid.
- **S2** — Accept — a one-word label, but it is a mislabelled term in a research
  figure that the write-up will quote.
- **S3** — Accept — two lines, and the two numbers being compared should come off
  one monotonic clock.
- **S4** — Accept — a statement reorder that puts the request on the wire before
  the thread is blocked, with no effect on the demo's visible ordering.
- **S5** — Accept — either surface it or delete it; an unread field that exists to
  report a failure is worse than no field.
- **S6** — Accept — a stale `reachable` tag beside a timeout undercuts the one
  thing the health check is there for.
- **S7** — Accept — removes a full image decode from every scan for a one-line
  change, using a pattern the same function already uses.
- **S8** — Accept — three defensive lines on the only path the device run did not
  cover, where the failure mode is a dead source rather than a bad number.
- **N1** — Accept — trivial, and `object` is the one weak type left in the diff.
- **N2** — Decline — at most 1 px, provably zero for the bundled set once B1
  lands, and unobservable for photos with no ground truth. Record it in
  `learnings.md` instead.
- **N3** — Accept — the value is already measured and thrown away, and it restores
  the mockup's server card.
- **N4** — Decline — a hung native call is not a realistic demo risk, and racing
  the local path adds a branch to the number the research record depends on.
- **N5** — Decline — iOS reclaims the cache directory; not worth code in a
  prototype.
- **N6** — Decline — deliberate and commented, the permission string is already
  shipped, and the demo grants access.
- **N7** — Accept — cheap, and the guidelines are explicit that every async
  boundary is wrapped with the failure surfaced.
