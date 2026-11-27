# Scan: Both Inference Paths on the Phone

## Overview

Spec 01 produced a phone-runnable model and a working server endpoint, but
nothing on the phone has ever sent a photo to the server, and nothing has ever
compared the two paths on real hardware. This feature builds the scan itself:
pick or take a photo, prepare it exactly as the research pipeline does, run it
through both the on-device model and the server, and show both answers with
what each one cost.

It deliberately makes no routing decision. Running both paths on every scan is
what the simulator did to produce clean "which path actually won" labels, so
this produces the same record from a real phone over a real network — the live
data the project's simulated results have so far lacked. Choosing between the
paths is spec 03.

This is spec 02 of the final demo.

## Requirements

### Choosing a photo

- The user can scan a photo from three sources: the device camera, the device
  photo library, and a bundled set of test photos shipped with the app.
- The bundled photos are held-out test images whose correct species is known, so
  a scan can be judged right or wrong without a human identifying the bird.
- Photos from the camera or the library have no known correct answer, and the
  app says so rather than implying a correctness it cannot establish.
- The app asks for camera and photo-library access with a stated purpose, and a
  refusal leaves the other sources usable.

### Preparing the photo

- Before anything is classified or sent, the photo is prepared the same way the
  research evaluation prepares one: shorter side resized to 232, centre-cropped
  to 224, encoded as JPEG at quality 85.
- Both paths see the same prepared bytes, so a difference between their answers
  is the models differing and never the input differing.
- The prepared size in pixels and the payload size in bytes are both shown to
  the user before and after the scan.
- The payload is exactly the image bytes, with no enclosing envelope, because
  the research's cost model is built on a measured payload of 11.39 KB.

### Running both paths

- Every scan runs the on-device model and the server, on the same prepared
  photo.
- The on-device result reports the predicted species, its confidence, the
  top-2 margin, and the time the model itself took.
- The server result reports the same three values, the time the server says its
  model took, and the total round-trip time measured on the phone.
- The round trip and the server's own model time are reported separately, so
  the network cost and the server cost can be told apart.
- Results appear as they arrive rather than after both finish, so the gap
  between a few milliseconds on device and a round trip over the network is
  visible rather than described.
- Where the correct species is known, each path is marked correct or wrong
  against it.
- Each scan reports which path won: on accuracy where the correct answer is
  known, and on latency always.

### When the server cannot be reached

- A scan whose offload fails still returns and keeps its on-device result.
- The failure is shown with a reason, and never silently swallowed.
- An unreachable or slow server cannot leave a scan waiting indefinitely.
- Before scanning, the app shows whether the server is currently reachable, so
  a failure during a demonstration is anticipated rather than discovered.

### Reaching the server

- The server's address is configuration that can be changed without rebuilding
  the app.
- The app can reach the server over plain HTTP at a private network address.
  (This needs a native allowance; without it the platform blocks the request
  before it is sent.)

### Replacing the spike

- The exploratory screen added while testing the delegate is replaced by this
  one; the app has a single purpose after this feature.

## UI Mockup

`ui-mocks/final-demo-scan.html` — four states: idle, running, a completed scan
with both results, and a scan whose offload failed.

## Out of Scope

- **Any routing decision.** No choice between the paths, no latency budget, no
  deadline, no cascade and no confidence threshold. Both paths always run. This
  is spec 03.
- **The background network monitor** and any rolling estimate of network
  conditions or device load.
- **Recording scans anywhere on the phone or in the database.** The server
  already records what it classifies; pairing that with the on-device result is
  spec 03.
- Any change to the server, its endpoints or its response shape.
- Fixing the relayed network path between the phone and the server. Round-trip
  times measured here include that relay, and are reported as measured.
- Measurements from a release build. This feature is developed and demonstrated
  on a development build, whose timings are not reportable figures.
- Any change to how the model was exported, or to the model itself.
- Running more than one photo per scan, or any batch or benchmark mode.
- Saving, sharing or exporting scan results.

## Acceptance Criteria

### Choosing and preparing

- Given the app is open, when the user chooses the bundled source, then they can
  select a test photo whose correct species the app knows.
- Given the user chooses the camera or the library and grants access, when they
  select or take a photo, then it can be scanned and the app states that no
  correct answer is known for it.
- Given access to the camera or the library is refused, when the user returns to
  the app, then the bundled photos remain scannable.
- Given any photo is chosen, when it has been prepared, then the app shows it at
  224 by 224 and shows the payload size in bytes.
- Given a photo has been prepared, when its bytes are compared against the same
  photo prepared by the research pipeline, then they are the same size and
  produce the same prediction from the server.

### Scanning

- Given a prepared photo, when the user scans, then both the on-device model and
  the server run on the same bytes.
- Given a scan completes, when its on-device result is shown, then it reports a
  species, a confidence, a top-2 margin and the model's own time.
- Given a scan completes, when its server result is shown, then it reports a
  species, a confidence, a top-2 margin, the server's reported model time and
  the total round-trip time measured on the phone.
- Given a scan is running, when the on-device result is ready before the server
  result, then it is shown without waiting for the server.
- Given a scanned photo whose correct species is known, when both results are
  shown, then each is marked correct or wrong against it.
- Given a completed scan, when its summary is shown, then it names which path
  won on latency, and on accuracy when the correct species is known.
- Given the same bundled photo is scanned twice with the server reachable, then
  both scans report the same two species and the same two confidences.

### Failure

- Given the server is unreachable, when the user scans, then the on-device
  result is still shown and the server result is shown as failed with a reason.
- Given the server does not respond, when a scan is run, then the scan ends by
  itself rather than waiting indefinitely.
- Given the server is unreachable, when the user opens the app, then its
  reachability is shown as unavailable before any scan is attempted.
- Given the configured server address is changed, when a scan is run, then the
  new address is used without rebuilding the app.

## Related specs

- **01 — real inference paths** (merged): the exported on-device model, the
  species list, and the server's classify and health endpoints this feature
  calls.
- **03 — the router** (not yet written): the background network monitor, the
  deterministic budget rule choosing one path before inference runs, and the
  paired record of the decision and its outcome.
