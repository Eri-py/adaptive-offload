# Review — Real Inference Paths: Phone Model Export and the Server Classify Endpoint

## Verdict

The export half of this spec is done well and is verifiably correct: the gate
re-reads the staged file before publishing it, the species list I checked
independently against `classes.txt` matches entry-for-entry in the right order,
and the committed artifact is genuinely float32 (input `(1, 3, 224, 224)`,
output `(1, 200)`, 275 float32 tensors and 9 int32 shape constants). The server
half is clean, correctly layered, and the two accuracy-critical decisions — the
224-or-else transform and timing only the forward pass between CUDA syncs — are
right. One blocker: the production request-logging path cannot resolve
`DATABASE_URL`, because `server/api/core/config.py:12` loads only `server/.env`,
which does not exist, while the URL lives in `database/.env`. Logging is
best-effort by design, so as shipped every classify request would warn once and
record nothing — the spec's primary logging requirement would quietly not
happen, and no test covers that path because every test overrides the logger
seam. Everything else I raise is polish or documentation. Of the orchestrator's
six questions, I agree with five of the design calls (details in Suggestions)
and give a ranked view on the 12.5 ms vs 6.08 ms gap in S7.

Gates I re-ran myself: `server/` 29 passed, `ruff` and `mypy --strict` clean (30
files); `database/` 1 passed, `ruff`/`mypy` clean; `training/`
`tests/birds/test_export.py` 10 passed, `ruff` clean. I did not re-run the full
`training/` suite (it exceeds the sandbox timeout).

## Acceptance Criteria

### Export

1. **Artifacts written to the app's bundled-asset location and git-tracked —
   MET.** `training/birds/export.py:18` defaults to `app/assets/models/`;
   `git ls-files` lists both `app/assets/models/small.tflite` and
   `app/assets/models/cub200-species.json`, and `git check-ignore` matches no
   rule for either.
2. **Missing checkpoint fails with the training command and writes nothing —
   MET.** `export.py:106` goes through
   `birds/modeling/build.py:40-48`, whose message is "Run `python -m birds.train
   --model small` first." `output_dir.mkdir` is at `export.py:109`, after that
   load, so nothing is created. See N1 for an ordering wrinkle in the message
   the user actually sees.
3. **Parity against PyTorch on real test photos, class agreement plus bounded
   drift — MET.** `export.py:101-123`; tolerance stated at `export.py:22`
   (1e-3) and quoted in the failure text at `export.py:63-67`. The gate reloads
   the staged file via `litert_torch.load` (`export.py:117`), so the artifact
   that ships is the one verified — a genuinely important mid-task correction.
   Measured: 50 photos, 0 disagreements, worst drift 2.59e-06.
4. **Entry *i* names the class the training split maps to index *i* — MET.**
   `export.py:70-86`, with the explicit guard at `export.py:81` that the file's
   class ids are sequential from 1 (so the `id - 1` assumption cannot silently
   break). I verified the committed JSON equals `classes.txt`'s folder names,
   stripped and in order, for all 200 entries.
5. **No numeric prefix, no underscores — MET.** `export.py:83`; the committed
   file has 200 unique entries, none containing a digit or an underscore.
6. **Runs with no GPU — MET by inspection.** `export.py:106` loads on
   `torch.device("cpu")` and `require_cuda` is never called; the conversion path
   is CPU-only. Not exercised on a GPU-less machine and not pinned by a test.
7. **Re-running on an unchanged checkpoint gives a functionally identical
   artifact — MET indirectly.** The parity gate ties every published artifact to
   the PyTorch model within 1e-3, which transitively bounds divergence between
   two runs. No test and no byte-identity claim; the spec asks only for
   functional identity, so this is adequate.

### Server

1. **Health responds and reports the model loaded — MET.**
   `server/api/controller/health.py:14-16` reads app state only;
   `server/tests/controller/test_health.py:7-11`. Cheap enough for an RTT probe.
2. **224x224 JPEG returns the full payload — MET.**
   `server/api/contracts/classification.py:21-30`;
   `server/tests/controller/test_classification.py:20-31` asserts every field.
3. **Non-224 JPEG is resized/cropped and the response says so — MET.**
   `server/api/services/classification/preprocess.py:46-51`;
   `server/tests/controller/test_classification.py:34-38`. The orchestrator's
   0.0-max-abs-diff check against the training eval transform confirms the
   geometry; I confirmed the preset sizes independently — both
   `MobileNet_V3_Large_Weights.IMAGENET1K_V2` and
   `ConvNeXt_Base_Weights.IMAGENET1K_V1` are `resize 232 / crop 224 / bilinear /
   antialias`, so the hardcoded constants at `preprocess.py:11-12` and the
   comment at `preprocess.py:16` are both accurate.
4. **Same image twice is identical — MET.**
   `server/tests/controller/test_classification.py:48-54`, plus the
   orchestrator's repeat run against the real checkpoint.
5. **Undecodable body gives a client error without crashing — MET.**
   `preprocess.py:40-43` maps `UnidentifiedImageError`/`OSError`/
   `DecompressionBombError` to `ImageDecodeError`;
   `server/api/controller/classification.py:35-38` maps that to 400. Covered at
   `test_classification.py:41-45` and `test_preprocess.py:49-57` (empty body,
   garbage, truncated JPEG). `.convert("RGB")` inside the `try`
   (`preprocess.py:41`) is what makes the truncated case land in the right
   branch, since that is where Pillow actually decodes.
6. **No CUDA means a clear startup failure — MET.**
   `server/api/services/classification/model.py:15-22`, reached from the
   lifespan at `server/api/core/app.py:31`;
   `server/tests/controller/test_health.py:21-26`.
7. **Model loaded once, never per request — MET.** Loaded only in the lifespan
   (`app.py:28-33`); the providers read `app.state`
   (`server/api/dependencies/providers.py:14-23`).
   `server/tests/core/test_app.py:23-30` proves exactly one forward pass (the
   warmup) at startup. No test posts several requests and counts loads; the
   structure makes a reload impossible.
8. **compute_ms is smaller than total_ms and excludes decode and transform —
   MET.** `server/api/services/classification/classify.py:34-42` brackets only
   `model(batch)`, with `torch.cuda.synchronize` on both sides, and the
   host-to-device copy at `classify.py:32` is deliberately outside the timer.
   `total_ms` starts at `providers.py:44-46` and ends at
   `classification.py:56`, so it strictly contains the forward pass;
   `server/tests/test_integration_classify.py:55` asserts the inequality and
   `server/tests/services/classification/test_classify.py:53-60` pins the span
   to the forward pass with a sleeping stub. See S3 on what `total_ms` does and
   does not cover.

### Logging

1. **Exactly one row with all ten fields — PARTIAL.** The schema and writer are
   right: `database/common/models.py` `ClassifyRequest` carries every field the
   spec lists, migration `0005` matches it column for column, and
   `server/tests/test_integration_classify.py:34-55` proves exactly one row with
   the posted byte count and the response's `compute_ms`. But that test, and
   every other, injects the engine; on the production path
   `server/api/core/config.py:12` never loads `DATABASE_URL`, so
   `request_log.py:25` raises inside the best-effort catch and no row is ever
   written. See B1.
2. **A logging failure still returns a successful classification and is
   surfaced — MET.** `server/api/services/classification/request_log.py:42-57`
   wraps row construction *and* the session, so a bad field value is caught as
   well as a dead connection. Covered three ways in
   `server/tests/services/classification/test_request_log.py:50-99` and
   end-to-end at `test_integration_classify.py:58-72`. The warning does reach
   stderr with no logging config, via Python's `logging.lastResort` handler.
3. **Server starts and serves with Postgres unreachable — MET.** The lifespan
   touches no database (`app.py:28-33`); the engine is built lazily on first
   insert (`request_log.py:21-26`) and a failed build is deliberately not cached.

## Scope

The server, database and export work matches the plan's Files Changed table.
Three things sit outside it:

- **The on-device spike** (`app/hooks/useBirdSpike.ts`,
  `app/screens/BirdSpikeScreen.tsx`, `app/App.tsx`, `app/README.md`,
  `app/package.json`/`package-lock.json`, 15 `app/assets/birds/*.jpg` plus
  `expected.json`, and the deletion of `app/hooks/useBenchmark.ts`,
  `app/screens/BenchmarkScreen.tsx`, `app/assets/models/yolov8n.tflite` and 15
  `app/assets/images/*.jpg`). `spec.md:102-104` puts the app, on-device
  inference and "anything the user sees" in spec 02, so this is the one real
  piece of scope drift. It is recorded as deliberate in
  `implementation-report.md`, and it produced the most valuable measurement on
  the branch, so I am not treating it as a defect — but its results have not
  landed where the write-up reads from (S1) and the deleted benchmark is only
  noted in `app/README.md` (S2).
- **`server/tests/core/test_app.py` and its package marker**, added for the
  startup warmup. Reasonable and in the right place.
- **The startup warmup itself** (`app.py:16-24`, `app.py:32`), not in the plan.
  Clearly justified: without it the first request writes a ~2224 ms figure into
  `compute_ms` that is indistinguishable after the fact from a real
  measurement. Tested.

No scope drift in `database/` or `training/`, and migration `0005` is authored
and not applied, as required.

## Blockers

#### B1 — Production request logging can never resolve `DATABASE_URL`

- **File:** `server/api/core/config.py:12`
- **Issue:** The server loads only `server/.env`, which does not exist in this
  repo; `DATABASE_URL` lives in `database/.env`, which nothing on the server
  path loads. Under `uvicorn api.core.app:app`, `request_log._lazy_engine`
  (`server/api/services/classification/request_log.py:25`) therefore raises
  "DATABASE_URL is not set" on every request, is swallowed by the best-effort
  catch at `request_log.py:56`, and `classify_requests` stays empty — so
  "Every classify request is recorded as one row in Postgres" (`spec.md:85`)
  does not hold for the deployed server. Nothing in the suite catches it: the
  one test that touches real resolution (`test_request_log.py:90-99`) deletes
  `DATABASE_URL` and asserts the warning, which codifies the broken state as
  expected behaviour. The repo already learned this exact lesson in
  `features/birds-per-photo-predictions/learnings.md` ("called `get_engine()`
  without `load_dotenv`, so `python -m ...` failed").
- **Fix:** Load `database/.env` as a fallback in `config.py` (the same way
  `database/migrations/env.py:23-33` does), or explicitly require `server/.env`
  and say so wherever the server's startup is documented. Add a test pinning the
  resolved env-file path, as `training/tests/birds/test_config.py` does for
  `training/.env`.
- **Decision:** Accepted — addressed in "Address B1, S5, S6: Make production request logging work and cover it"

## Suggestions

#### S1 — The spike's measurements are not in `findings/`, and this spec's Open Question is stale

- **File:** `findings/birds.md:252-263`
- **Issue:** `git diff main...HEAD -- findings/` is empty. The guidelines say a
  spec that produces results appends to `findings/`, "because they accumulate
  across features and are what the write-up draws on". The on-device numbers
  (4.20 ms Core ML, 7.19 ms CPU, 15/15 agreement, the 4.8e-02 decoder-induced
  probability drift, the Debug-build caveat) live only in this feature's
  `learnings.md`, while `findings/birds.md` still marks local cost at 5.877 ms
  desktop CPU with no note that a lower device figure exists. Separately,
  `spec.md:190-195` still records the Core ML operator-coverage question as
  unresolved and deferred to spec 02, when the spike answered it on this branch.
- **Fix:** Append a device-measurement section to `findings/birds.md`
  (local cost, delegate result, decoder drift against the 0.8728
  max-probability threshold, Debug-build caveat) and mark the spec's Open
  Question resolved with a pointer to it.
- **Decision:** Accepted — addressed in "Address S1, S2, S7: Record the device measurements in findings/"

#### S2 — The YOLOv8n benchmark was deleted with only an `app/README.md` note

- **File:** `app/README.md:9-13`
- **Issue:** `app/assets/models/yolov8n.tflite`, `BenchmarkScreen.tsx`,
  `useBenchmark.ts` and 15 COCO images were removed. The 17.34 ms Core ML figure
  they produced is quoted in `findings/coco-router.md:665`, which the write-up
  draws on, and that file now describes a measurement whose code and model are
  no longer in the tree. The removal is defensible, but it is recorded only in
  the app's README, not where the figure is cited.
- **Fix:** Add one line at `findings/coco-router.md:665` noting that the
  benchmark harness was removed on this branch and naming the commit, so the
  figure stays traceable.
- **Decision:** Accepted — addressed in "Address S1, S2, S7: Record the device measurements in findings/"

#### S3 — `total_ms` is the handler span, not the request span, and that is not written down

- **File:** `server/api/dependencies/providers.py:44-46`
- **Issue:** The design is sound and I verified the part that matters: with a
  200 ms delay injected into the body read, `total_ms` came back at 205 ms, so
  the stamp genuinely precedes `await request.body()`. But it excludes
  Starlette's receive and routing before dependency resolution, excludes
  response serialization and transmission, and ends before the insert
  (`classification.py:56`) — and it relies on FastAPI resolving dependencies in
  signature order, which is behaviour, not contract. Since `total_ms -
  compute_ms` is meant to be the live counterpart of the cost model's
  non-compute server term, the report could over-claim what it measures.
- **Fix:** State precisely what the two timestamps bracket in
  `ClassifyRequest`'s docstring (`database/common/models.py`) and in
  `log_classify_request`'s, and add a comment at `classification.py:28`
  recording that `started_at` must stay the first parameter. Move the stamp into
  middleware only if a true request span is wanted later.
- **Decision:** Accepted — addressed in "Address S3, S4, N4: Validate JPEG input and document the timing span"

#### S4 — Nothing checks the body is actually a JPEG, so `payload_bytes` can be polluted

- **File:** `server/api/services/classification/preprocess.py:41`
- **Issue:** `Image.open` accepts PNG, WebP, TIFF and friends, so a
  misconfigured client's 224x224 PNG would classify happily and write a
  `payload_bytes` value an order of magnitude off into the column that exists
  specifically as the live counterpart of `PAYLOAD_KB = 11.39`. The spec says
  the endpoint accepts "a single JPEG image" (`spec.md:55`), and the
  payload-size rationale is explicit at `spec.md:89-92`.
- **Fix:** After decoding, reject with 400 when `image.format != "JPEG"`, or
  record the format alongside `payload_bytes` so a polluted row is identifiable
  later.
- **Decision:** Accepted — addressed in "Address S3, S4, N4: Validate JPEG input and document the timing span"

#### S5 — The logger seam is typed `Callable[..., None]`, so mypy cannot catch a signature drift

- **File:** `server/api/dependencies/providers.py:49`
- **Issue:** `Callable[..., None]` erases the keyword-only contract, so the call
  at `classification.py:52-57` and the definition at `request_log.py:29-36` can
  diverge silently under `mypy --strict`. That is precisely the kind of break
  Task 7 had to repair when it wired the writer up, and it is invisible in tests
  because every test substitutes its own callable.
- **Fix:** Define a `Protocol` with the real signature
  (`(Classification, *, payload_bytes: int, was_transformed: bool, total_ms:
  float) -> None`) and annotate both the provider and the handler parameter with
  it.
- **Decision:** Accepted — addressed in "Address B1, S5, S6: Make production request logging work and cover it"

#### S6 — The real writer is never wired in any test

- **File:** `server/tests/conftest.py:52`
- **Issue:** The no-op default is the right call — without it the controller
  tests would have aimed inserts at the real database through `database/.env` —
  but combined with every other test overriding `get_request_logger`, the result
  is that nothing asserts the route is connected to the production writer at
  all. That is how B1 got through: the seam is tested from both sides and never
  end to end.
- **Fix:** Add two cheap assertions: `get_request_logger() is
  log_classify_request`, and the resolved-env-path pin from B1. Keep the no-op
  default exactly as it is.
- **Decision:** Accepted — addressed in "Address B1, S5, S6: Make production request logging work and cover it"

#### S7 — Settle the 12.5 ms vs 6.08-6.60 ms server-compute gap before the report quotes either

- **File:** `features/final-demo/01-inference-paths/learnings.md`
- **Issue:** My ranked view on the cause. (a) **GPU clock state is the most
  likely explanation.** A tight loop holds the SM and memory clocks at boost; a
  single isolated inference arriving at an otherwise idle GPU runs at a much
  lower clock, and a ~2x spread on a ConvNeXt-Base forward is entirely ordinary.
  (b) **Measurement asymmetry is the second candidate.** If the tight loop
  synchronized once for N iterations and divided, it pipelined kernel launches
  and systematically understates per-request latency, whereas
  `classify.py:36-42` synchronizes on both sides of every single call — the
  endpoint figure is the honest per-request number, and the loop figure is the
  optimistic one. (c) **The threadpool hypothesis in `learnings.md` I would
  effectively rule out:** `run_in_threadpool` dispatch costs microseconds, the
  GIL is released inside the forward pass, and the warmup series (2224 ms, then
  16.86 ms, then ~12.5 ms) already localises the remaining cost to the GPU, not
  to the dispatch. Practical consequence: ~12.5 ms is what a phone sending
  spaced-out single photos actually experiences, so it, not 6.731 ms, is the
  live server-compute term.
- **Fix:** Re-measure against a running server with a per-iteration-synchronized
  tight loop as the control and `nvidia-smi --query-gpu=clocks.sm,clocks.mem`
  sampled during both. Report the endpoint figure as the live term and keep
  6.731 ms labelled as the offline `birds.evaluation.evaluate` figure.
- **Decision:** Accepted — addressed in "Address S1, S2, S7: Record the device measurements in findings/"

#### S8 — Concurrent requests can overlap forward passes on one GPU

- **File:** `server/api/controller/classification.py:27`
- **Issue:** The sync `def` handler is the right choice (it keeps inference off
  the event loop), but it means Starlette's threadpool can run several forward
  passes on the same GPU at once, and each would then report an inflated
  `compute_ms` into the research log. The demo has one phone, and batching is
  explicitly out of scope (`spec.md:116`).
- **Fix:** If this ever matters, serialize inference behind a module-level
  `threading.Lock` in `classify.py` and time only the forward pass inside it.
- **Decision:** Declined — the demo has a single client, so forward passes do not overlap in practice; serialising GPU access would add contention handling for a condition that does not arise.

## Nitpicks

#### N1 — The missing-checkpoint path can surface the dataset error instead of the training command

- **File:** `training/birds/export.py:105`
- **Issue:** `species_names` runs before `load_checkpoint`, so on a clean
  machine with neither the dataset nor the checkpoint the user sees "Run `python
  -m birds.dataset.download`", not the training command `spec.md:127-128` asks
  for. Nothing is written either way, so only the message is wrong.
- **Fix:** Load the checkpoint first, then read `classes.txt`.
- **Decision:** Accepted — addressed in "Address N1, N3: Fix the checkpoint error order and relocate the startup test"

#### N2 — The response calls it `margin`; everything else calls it `top2_margin`

- **File:** `server/api/contracts/classification.py:27`
- **Issue:** The DB column is `top2_margin`, `findings/birds.md` uses
  `top2_margin`, and the spec says "top-2 margin". A bare `margin` on the wire
  is ambiguous for spec 02's client, which has not yet pinned the format.
- **Fix:** Rename the contract field to `top2_margin` now, before the app
  consumes it.
- **Decision:** Accepted — addressed in "Address N2: Rename the contract field to top2_margin"

#### N3 — A startup test lives in the controller test tree

- **File:** `server/tests/controller/test_health.py:21`
- **Issue:** `test_startup_fails_clearly_without_cuda` exercises
  `api/core/app.py` and `api/services/classification/model.py`, not the health
  controller. The guidelines require tests to mirror source, and
  `server/tests/core/test_app.py` now exists.
- **Fix:** Move it to `server/tests/core/test_app.py`.
- **Decision:** Accepted — addressed in "Address N1, N3: Fix the checkpoint error order and relocate the startup test"

#### N4 — `load_settings`'s docstring overstates what is re-read

- **File:** `server/api/core/config.py:24`
- **Issue:** It claims settings are read at call time "so tests and `.env`
  changes are honoured", but `load_dotenv` runs once at import
  (`config.py:12`), so an edited `.env` needs a process restart. Only
  `os.environ` changes are honoured per call.
- **Fix:** Reword to "reads the environment at call time so tests can
  monkeypatch it; `.env` is loaded once at import".
- **Decision:** Accepted — addressed in "Address S3, S4, N4: Validate JPEG input and document the timing span"

#### N5 — `POST /classify` publishes no request body in OpenAPI

- **File:** `server/api/controller/classification.py:26`
- **Issue:** Reading the body through a dependency means `/docs` shows the
  endpoint with no request schema, so spec 02 cannot discover the wire format
  from the server.
- **Fix:** Add `openapi_extra` declaring an `image/jpeg` binary body.
- **Decision:** Declined — the app is the only client and spec 02 pins the wire format from the contract module directly, so an OpenAPI body adds no discoverability that matters here.

## Tests

Added: 10 export tests, 27 new server tests across `core/`, `controller/`,
`services/classification/` and one integration module, and the `database/`
round-trip extended to the seventh table. The tree mirrors the source layout
correctly. The quality is better than the prototype bar in the places that
matter: `test_classify.py:10-12` uses logits of `log(w)` so the expected
probabilities, margin and top-3 ordering are exact rather than approximate and
no two probabilities tie (which would make `topk` order unspecified);
`test_export.py:87-104` proves an unusable staged file publishes nothing;
`test_request_log.py:74-87` drops the table to produce a real missing-table
failure rather than mocking one.

Gaps, in rough order of importance:

- **The production logging path is untested end to end** — the root cause of B1.
  Every test substitutes the logger (`conftest.py:52` by default,
  `test_integration_classify.py:37` and `:64` deliberately), so neither
  `get_request_logger`'s identity nor `DATABASE_URL` resolution is ever
  exercised. See S6.
- **Migration `0005` is never executed by the suite.** The ephemeral database is
  built with `Base.metadata.create_all` (`database/common/testing.py:55`), so a
  model/migration divergence is undetectable by tests. The orchestrator's manual
  `create_table`-capture check covers it for now; worth remembering that the
  check is manual and will not re-run.
- **No test posts several classify requests and asserts one model load**
  (server criterion 7), and none pins the no-GPU export path (export criterion
  6). Both are structurally sound rather than verified.
- **Fragile-ish:** `test_classify.py:50` bounds `compute_ms` below 5000 ms and
  `test_classify.py:60` requires >= 50 ms from a `time.sleep(0.05)` — both fine
  under CI jitter, but they are wall-clock assertions.
  `test_integration_classify.py:61-62` depends on 127.0.0.1:1 refusing
  connections promptly; it does here, and `connect_timeout=1` bounds the damage.
- `test_export.py:78-79` asserts the resolved output path ends in
  `adaptive-offload/app/assets/models`, which pins the repo's directory name.
  Harmless, worth knowing if the checkout is ever renamed.

One inconsistency between the workflow records, flagged for information only
rather than as a finding: `progress.md` marks Task 8 "not started — Deferred"
while `implementation-report.md` reports it completed with 331 passing tests. I
re-ran the parts I could and they agree with the report (`server/` 29,
`database/` 1, `training/tests/birds/test_export.py` 10, `ruff` clean in all
three, `mypy` clean in `database/` and `server/`); I did not re-run the full
`training/` suite.

## Recommended Decisions

- **B1** — Accept — the spec's main logging requirement does not hold for the
  deployed server, and the fix is a one-line fallback plus a path-pinning test.
- **S1** — Accept — `findings/` is what the write-up draws on, and the best
  measurement on this branch is currently invisible to it while a superseded
  figure still stands.
- **S2** — Accept — one line keeps a quoted figure traceable after its harness
  was deleted.
- **S3** — Accept — docs only, and it prevents the report over-claiming what
  `total_ms - compute_ms` measures.
- **S4** — Accept — a two-line guard protecting the column that backs the
  `PAYLOAD_KB` assumption.
- **S5** — Accept — restores the type checking the guidelines require on the one
  seam that has already drifted once.
- **S6** — Accept — cheap, and it closes the specific hole that let B1 through.
- **S7** — Accept — the number feeds the report's band boundaries; it needs
  settling, and the control experiment is small.
- **S8** — Decline — out of scope (`spec.md:116`), one client in practice, and
  adding a lock now would be speculative; revisit only if concurrent requests
  ever become real.
- **N1** — Accept — one statement reordered, and it makes the spec's stated
  error message the one users see.
- **N2** — Accept — rename now while nothing consumes the field; after spec 02
  ships it is a breaking change.
- **N3** — Accept — a file move, keeps the mirrored test tree honest.
- **N4** — Accept — one sentence; the current wording would mislead the next
  person debugging config.
- **N5** — Decline — the plan deliberately deferred OpenAPI customisation and
  spec 02 will be written against this code, not against `/docs`.
