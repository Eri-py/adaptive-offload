# Real Inference Paths: Phone Model Export and the Server Classify Endpoint

## Overview

The final demo app has to route a request between a small model on the phone
and a large model on a server, but neither path can actually run today: the
trained bird checkpoints exist only as desktop PyTorch files, and `server/` is
an empty skeleton. This feature makes both paths real and independently
runnable — an exported phone-ready copy of the small model, and a server
endpoint that classifies a photo with the large model — so the next spec can
build the routing app on top of two working ends.

This is spec 01 of the final demo (see Related specs below).

## Requirements

### Phone model export

- A runnable export step turns the trained small checkpoint (MobileNetV3-Large
  with the 200-way CUB head) into a file the mobile app can run on-device.
- The export refuses to run with an actionable error if the checkpoint is
  missing, naming the command that produces it, rather than exporting an
  untrained model.
- The exported model file lands where the mobile app bundles its assets, and is
  committed to version control. This is a requirement, not an accident of
  layout: the app's cloud build only uploads files tracked by git, so an
  untracked export would be silently absent from the installed app.
- The exported model keeps full float32 weights. No quantization of any kind,
  so the accuracy figures already recorded in `findings/birds.md` continue to
  describe the model the phone runs.
- The export also produces a species-name list the app can display, with one
  entry per class, derived from the dataset's own class list rather than
  hand-written.
- That list is ordered so that entry *i* is the name of model class index *i*.
  The dataset numbers classes from 1 and the training split subtracts 1, so
  entry *i* corresponds to the dataset's class *i+1*. A mismatch here would
  mislabel every prediction while still looking plausible, so the ordering is
  derived from the same source the training labels came from.
- Species names are presentable as-is in a UI: the dataset's numeric prefixes
  and underscores are not shown to the user.
- Before the export is accepted, it is verified against the PyTorch checkpoint
  on real test photos — not only on synthetic input — covering agreement on the
  predicted class and bounded drift in reported confidence. A verification
  failure is reported as a failure; it never produces a silently-wrong artifact.
- The export runs without a GPU, so it is not gated on the training machine.
- Re-running the export on an unchanged checkpoint produces a functionally
  identical artifact.

### Server classify endpoint

- The server exposes a health endpoint that reports whether the service is up
  and whether its model is loaded. It is cheap enough to be called repeatedly
  as a round-trip-time probe, since that is what the app's network monitor will
  use it for.
- The server exposes a classify endpoint accepting a single JPEG image.
- If the submitted image is already at the model's input size, the server only
  normalizes it. If it is any other size, the server first applies the same
  geometric transform the training evaluation used (resize the shorter side to
  232, then center-crop to 224). Either way the pixels reaching the model have
  been through the same pipeline the reported accuracy figures were measured
  with.
- The classify response reports: the top-1 species name and its class index,
  the maximum probability, the top-2 margin, the top-3 species with their
  probabilities, and the measured model compute time in milliseconds.
- The two confidence measures (maximum probability and top-2 margin) are the
  same two the cascade experiment used, so a live result can be read against
  the thresholds already recorded in `findings/birds.md`.
- The reported compute time measures the model forward pass only — not image
  decoding, transforming, transport or serialization — because the experiment's
  offload cost model treats server compute as a separate term from transfer and
  round trip.
- The large model is loaded once when the server starts, never per request.
- The server requires a CUDA GPU and refuses to start with a clear error
  without one. The server-compute figure the research uses is a GPU
  measurement, and a silent CPU fallback would report a number that looks
  valid but describes different hardware.
- A request whose body is not a decodable image is rejected with a clear client
  error rather than a server crash.
- The checkpoint location and the dataset class-list location are
  configuration, not values embedded in request handling.

### Request logging

- Every classify request is recorded as one row in Postgres, holding: when it
  happened, the submitted payload size in bytes, whether the server had to
  apply the geometric transform, the predicted class index and name, the
  maximum probability, the top-2 margin, the model compute time, and the total
  time spent handling the request.
- Payload size and the compute-versus-total split are recorded because they are
  the live counterparts of the two assumptions the offload cost model is built
  on — the 11.39 KB measured payload and server compute as a term distinct from
  transfer.
- A logging failure never turns a successful classification into a failed
  request: the result is still returned and the logging failure is surfaced in
  the server's own output.
- The server starts and serves classify requests even if Postgres is
  unreachable, so a database problem cannot take down a live demo.
- The new table is created by a migration authored in the shared database
  package. Authoring the migration is in scope; running it is the user's.

## Out of Scope

- The mobile app itself — bundling the exported model, on-device inference, the
  camera or photo picker, the network monitor, and anything the user sees. That
  is spec 02.
- The routing decision. No local-versus-offload choice, no latency budget and
  no cascade logic appears in this spec; the server classifies whatever it is
  sent.
- Phone-to-server networking setup (Tailscale, reachability from the device).
- Reviving `features/server-served-benchmark/` — the frame-list and
  frame-image endpoints from that spec are not built here, and
  `scene_complexity` plays no part in this feature.
- Quantization, and any attempt to shrink the exported model.
- Measuring what JPEG quality-85 transport costs in accuracy, and any
  re-measurement of the accuracy figures already in `findings/birds.md`.
- Running the small model on the server, and any batching of requests.
- Authentication, rate limiting, request size limits, and serving more than one
  dataset or model pair.
- Starting or stopping the server, and running the migration.

## Acceptance Criteria

### Export

- Given the trained small checkpoint is present, when the export step is run,
  then a phone-runnable model file and a species-name list are written to the
  app's bundled-asset location, and both are tracked by git.
- Given the small checkpoint is absent, when the export step is run, then it
  fails with an error naming the command that trains it, and writes no output.
- Given an export has been produced, when its predictions are compared against
  the PyTorch checkpoint over real test photos, then the predicted class agrees
  on every photo compared and the reported confidence differs by no more than a
  small stated tolerance.
- Given the exported species-name list, when entry *i* is read for any class
  index *i* the model can predict, then it is the name of the dataset class the
  training split maps to index *i*.
- Given the exported species-name list, when any entry is displayed, then it
  contains no numeric class prefix and no underscores.
- Given a machine with no GPU, when the export step is run, then it completes
  successfully.
- Given an unchanged checkpoint, when the export is run twice, then the two
  resulting models produce the same predictions.

### Server

- Given the server is running, when the health endpoint is called, then it
  responds successfully and reports that the model is loaded.
- Given the server is running, when a 224x224 JPEG is posted to the classify
  endpoint, then the response contains a top-1 species name and class index, a
  maximum probability, a top-2 margin, three ranked species with
  probabilities, and a compute time in milliseconds.
- Given a JPEG that is not 224x224 is posted, when it is classified, then the
  server applies the resize-and-center-crop transform before inference and the
  response reports that it did so.
- Given the same image is posted twice, when both are classified, then the
  predicted class, maximum probability and top-2 margin are identical.
- Given a request body that is not a decodable image, when it is posted to the
  classify endpoint, then the server responds with a client error describing
  the problem and does not crash.
- Given no CUDA GPU is available, when the server is started, then it fails at
  startup with a clear error rather than serving requests on CPU.
- Given the server has started, when several classify requests are handled,
  then the model was loaded once and not reloaded per request.
- Given a classify response, when its compute time is compared against the
  total handling time recorded for the same request, then compute time is the
  smaller of the two and excludes decode and transform work.

### Logging

- Given Postgres is reachable and the migration has been run, when a photo is
  classified, then exactly one row is recorded holding the timestamp, payload
  size in bytes, whether the transform was applied, predicted class index and
  name, maximum probability, top-2 margin, compute time and total handling
  time.
- Given Postgres is unreachable, when a photo is classified, then the caller
  still receives a successful classification and the server reports the
  logging failure in its own output.
- Given Postgres is unreachable, when the server is started, then it starts and
  serves classify requests.

## Related specs

- **02 — the phone app** (not yet written): bundles this export, runs it
  on-device under the Core ML delegate, monitors network conditions, makes the
  deterministic budget routing decision, and calls this classify endpoint when
  it routes to offload.

## Open Questions

- Whether the Core ML delegate covers MobileNetV3-Large's operator set well
  enough to run the whole graph on the Neural Engine is unverified. The only
  on-device delegate measurement this project has (17.34 ms) was YOLOv8n, a
  different architecture. This resolves in spec 02, where the model actually
  runs on the phone; it does not affect anything in this spec, which produces a
  standard float32 model file.
