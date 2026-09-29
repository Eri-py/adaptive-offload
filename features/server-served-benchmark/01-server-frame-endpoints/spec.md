# Server Frame Endpoints

## Overview

The mobile benchmark app currently ships its test images inside the app, so
the image set can only change with a new build and has no link to the frames
the simulator measured. This adds the FastAPI server's first endpoints: one
that returns a benchmark frame set chosen from the frames Postgres already
knows about, and one that serves each frame's image file from the dataset
folder on disk. It is the first of three specs (see Related specs below).

## Requirements

- The server exposes a frame-list endpoint that returns a set of benchmark
  frames from one configured dataset (`coco_val2017` today).
- A frame is eligible for the set only if all of the following are true:
  - Postgres has a scene-complexity score for it.
  - Postgres has a desktop inference timing for it on the local path
    (YOLOv8n, the same model the phone runs).
  - Its image file exists in the dataset's image folder on disk.
- The caller can ask for a number of frames. If they don't, the server
  returns 15. If fewer eligible frames exist than were asked for, the server
  returns all of them.
- The set is spread evenly across the eligible frames' scene-complexity
  range, from the least to the most complex.
- The set is deterministic: the same request against the same database
  contents always returns the same frames, in the same order, so separate
  benchmark runs stay comparable.
- Each frame in the list includes:
  - an identifier the caller can use to download that frame's image
  - its scene-complexity score
  - its desktop YOLOv8n inference time in milliseconds, exactly as stored in
    Postgres (the full-pipeline figure)
- The server exposes an image endpoint that returns a listed frame's original
  image file, byte-for-byte unchanged, so the phone decodes the same JPEG the
  desktop measurement did.
- The image endpoint only serves images of frames in the configured dataset
  that Postgres knows about. It never serves arbitrary files from disk, and
  it rejects a request that names anything else.
- The dataset's image folder location is configuration, not hard-coded in
  request handling.
- If Postgres is unreachable, both endpoints return a clear error response
  rather than crashing or hanging.
- Image bytes stay on disk. Postgres decides which frames are served but does
  not store images.

## Out of Scope

- Any change to the mobile app: fetching frames, removing the bundled images,
  and the phone-to-server network setup (Tailscale inside WSL) are spec 02.
- Full-pipeline and per-stage timing on the phone, and the phone-vs-desktop
  comparison view, are spec 03.
- Storing image bytes in Postgres or any object store.
- Serving more than one dataset, or choosing the dataset per request.
- Choosing frames by simulation run, or random sampling.
- Authentication or access control (the server is only reachable on the
  developer's own machine and tailnet).
- The offload inference endpoint, and receiving or storing benchmark results
  from the phone.
- On-device or HTTP caching of images.
- Any database schema change or new migration.

## Acceptance Criteria

- Given the server is running and Postgres holds complexity scores and
  YOLOv8n desktop timings for frames whose images are on disk, when a client
  requests the frame list without a count, then it receives 15 frames, each
  with an identifier, a complexity score and a desktop YOLOv8n time in ms.
- Given a count is requested, when there are at least that many eligible
  frames, then exactly that many are returned; when there are fewer, then all
  eligible frames are returned.
- Given a returned frame set, when its complexity scores are inspected, then
  they span the eligible range, including frames at or near the lowest and
  highest complexity, rather than clustering at one end.
- Given the database contents haven't changed, when the same frame-list
  request is made twice, then both responses list the same frames in the same
  order.
- Given a frame that has a complexity score but no YOLOv8n desktop timing, or
  whose image file is missing from disk, when the frame list is requested,
  then that frame is never included.
- Given a frame from the list, when its image is requested, then the response
  body is byte-for-byte identical to the file on disk and is served as a JPEG.
- Given an identifier that isn't a known frame of the configured dataset
  (including one that tries to reach a path outside the image folder), when
  its image is requested, then the server returns a not-found error and
  serves no file.
- Given Postgres is unreachable, when either endpoint is called, then the
  server returns an error response that says the database is unavailable.
- Given this feature is complete, then no image bytes have been written to
  Postgres and no migration has been added.

## Related specs

- `features/server-served-benchmark/02-app-fetches-frames/`: the app
  downloads its frames from these endpoints on every run; the bundled images
  are removed; the phone reaches the server over Tailscale running inside WSL.
- `features/server-served-benchmark/03-full-pipeline-timing/`: the phone
  times decode, preprocess, model and postprocess (including NMS) separately
  plus a total, and shows its total next to the desktop YOLOv8n time from the
  frame list.

## Open Questions

None.
