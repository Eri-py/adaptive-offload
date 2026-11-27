
## Task 1 — export module
- `litert_torch` has no type stubs; the deferred import needs `# type: ignore[import-untyped]` for mypy.
- `mypy .` here reported 9 errors total (all pre-existing ndarray ones) with the new files clean.
- Running all of `tests/birds` in one go exceeded 120s in this sandbox; run `tests/birds/test_export.py` alone.
- Export stages to `small.tflite.partial` and `os.replace`s only after parity passes; the species JSON is written last, so a failed parity leaves neither file.
- Parity now reloads the staged file via `litert_torch.load(path)` so the shipped artifact is what is verified.

## Task 2 — real export run
- `export()` prints no parity numbers; to observe them, wrap `birds.export.require_parity` from a throwaway script (in scratchpad) and print the `ParityReport`. Real run: 50 photos, 0 disagreements, max prob drift 2.59e-06.
- Run takes ~1-2 min; litert_torch stderr noise (absl/XNNPACK "ERROR: Subgraph reshaping...") is harmless.

## Bird spike (app/hooks/useBirdSpike.ts)
- `expected.json` imports directly as typed JSON (tsc passes with expo's base tsconfig); photos are keyed by filename in a require() map so ordering can't drift.
- Layout is one function `buildInputTensor(rgba, layout)`: NCHW writes `c*PIXELS + p`, NHWC writes `p*3 + c`; alpha byte skipped (jpeg-js stride 4, hard-coded rather than derived).
- `computeStats` is not exported from useBenchmark.ts and that file is frozen, so it is duplicated in the hook.
- Static gate only (`npx tsc --noEmit`, clean); nothing was run on a device.

## Phone networking — findings from the spike run (orchestrator)

- **WSL2 is on default NAT networking** (no `/mnt/c/Users/eriol/.wslconfig` exists), so the
  WSL address `172.30.244.19` is unreachable from the phone. `npx expo start` advertises that
  address by default, which is why the dev client could not connect. `@expo/ngrok` was also
  absent, so `--tunnel` would have failed too.
- **Tailscale is now installed in WSL** and is the chosen phone-to-WSL route. systemd is PID 1
  here (`/etc/wsl.conf` sets `systemd=true`) and `/dev/net/tun` exists, so `tailscaled` runs as
  a normal service in full TUN mode — no userspace-networking workaround needed. WSL node is
  `100.97.204.46` (`eri-desktop-linux`); the phone is `eri-iphone` (`100.77.49.10`).
  `tailscale up --accept-dns=false` is deliberate: MagicDNS rewrites `/etc/resolv.conf` in WSL
  and breaks name resolution for everything else.
- **The tailnet link is relayed, not direct:** `tailscale ping eri-iphone` answers
  `via DERP(ord) in 35ms` with "direct connection not established". WSL2's NAT commonly blocks
  NAT traversal. Harmless for Metro, but **it would distort any offload latency measurement** —
  routing phone-to-desktop traffic through a Chicago relay inflates RTT and caps bandwidth.
  Resolve before taking offload numbers: mirrored WSL networking, or run the server on Windows,
  or report the relay overhead explicitly as part of the measured condition.
- **iOS App Transport Security blocks cleartext HTTP to Tailscale addresses.** The Expo dev
  client refused the connection to `http://100.97.204.46:8081`. Tailscale uses the CGNAT range
  `100.64.0.0/10`, which iOS does not treat as a local network, so the `NSAllowsLocalNetworking`
  exception Expo sets by default does not cover it even though plain LAN addresses work.
  Worked around for the spike with `npx expo start --tunnel` (ngrok serves HTTPS).
  **Spec 02 must handle this properly:** the app will call
  `http://<tailscale-ip>:<port>/classify` and will hit the same block, so it needs an ATS
  exception in `app.json`'s `ios.infoPlist` plus one dev-client rebuild, or HTTPS in front of
  the server. This is a native-config change, so it cannot be fixed by a JS reload.

## Spike results on the iPhone — the local cost is measured at last

Development (Debug) build, 15 pre-cropped CUB photos, timing only `model.run()`:

| Layout | Delegate | Match | Mean | Median | Min | Max | Std dev |
|---|---|---|---|---|---|---|---|
| NCHW | CPU | 15/15 | 7.19 ms | 6.70 | 6.61 | 9.99 | 1.02 |
| NCHW | Core ML | 15/15 | **4.20 ms** | 4.21 | 4.10 | 4.27 | 0.05 |
| NHWC | CPU | 15/15 | 8.38 ms | 6.71 | 6.61 | 26.13 | 4.83 |
| NHWC | Core ML | 15/15 | 4.18 ms | 4.19 | 4.08 | 4.25 | 0.05 |

- **The Core ML delegate accepts MobileNetV3-Large.** This was the spike's whole
  question, and the answer is yes. It is genuinely engaged rather than falling
  back: 4.20 ms against CPU's 7.19 ms, with the standard deviation collapsing
  from 1.02 ms to 0.05 ms — the tight spread is the signature of the Neural
  Engine, not of a CPU fallback.
- **Tensor layout does not matter under Core ML** (4.20 vs 4.18 ms). The NHWC
  variant was a reasonable hedge and is now proven redundant;
  `small-nhwc.tflite` can be dropped, and the export step never needs a
  channel-last mode.
- **Every combination matched PyTorch 15/15**, so the JS preprocessing is right.
  Both layouts had already been validated off-device against a reference
  interpreter (15/15, drift 1.7e-06), so this confirms the device adds nothing.
- **The local cost is ~4.2 ms, not 5.877 ms and emphatically not 74 ms.** A7
  named this the single most valuable missing measurement, because the width of
  the band where offload fits but escalation does not *is* the local cost. The
  phone runs the small model faster than the desktop CPU stand-in, which inverts
  the assumption in `findings/birds.md` that the desktop figure flatters local.
- **Caveat: this is a Debug build.** `app/README.md` says to quote latency only
  from a `preview` build, so 4.2 ms is indicative until re-measured on Release.
- **Worst probability drift is 4.8e-02, far above the 1.7e-06 seen off-device.**
  The cause is the decoder: the phone uses `jpeg-js`, the off-device mirror used
  PIL, and they differ by about ±1 per byte. Predictions still agree everywhere,
  but **0.048 is large next to the cascade's 0.8728 max-probability threshold**,
  so a confidence-threshold router would flip escalation decisions near the
  boundary depending on which decoder produced the input. Any on-device
  confidence threshold has to be treated as decoder-dependent, or tuned on
  device-decoded photos.

## Task 3: server scaffolding

- `server/pyproject.toml` needs `mypy_path = "../database"` for `common.testing` and a
  `torchvision.*` `ignore_missing_imports` override, or strict mypy fails on both.
- Ruff B008 forbids `= Depends(...)` defaults; use `Annotated[T, Depends(...)]`.
- `create_app(load_resources=False)` skips the CUDA/checkpoint lifespan so tests can
  inject stubs through `dependency_overrides`; `create_app()` fails fast without CUDA.
- Starlette warns that `httpx` TestClient is deprecated in favour of `httpx2` (harmless).

## Task 4: preprocessing and classification service

- `classify()` takes an explicit `device` rather than reading it off the model: the shared `StubModel` has no parameters, so `next(model.parameters())` would raise.
- Hand-computable softmax: use logits of `log(w)` for integer weights; probabilities are then `w / sum(w)`. Avoid tied probabilities, since `topk` tie order is unspecified.
- Decode failures surface as `UnidentifiedImageError` (garbage) or `OSError` (truncated JPEG, raised at `.convert()`/load time, not `open`). Both map to `ImageDecodeError`.

## Task 5 — Classify endpoint
- A `bytes` handler parameter is treated as a form field by FastAPI; read the raw body in an async
  dependency (`get_raw_body`) and keep the handler a sync `def` so inference runs in the threadpool.
- Device and 503 providers chain off `get_loaded_model`, so the single conftest override covers all
  of them; a parameterless `StubModel` falls back to CPU in `get_device`.

## Task 5 — what the real end-to-end run showed (orchestrator)

Driven in-process with `TestClient` and the real `large.pt` on CUDA, posting a
real 224x224 CUB crop. No server was started.

- **Correctness is exact.** The response matched an independent normalize-only
  PyTorch reference to the digit: class 20, max probability 0.804325. Repeat
  calls are deterministic. Comparing instead against the *double-transformed*
  pipeline gave 0.679482 — a 12.5-point confidence swing from re-cropping an
  already-cropped image, which is the concrete cost of the design the spec
  rejected.
- **The first request cost 2224 ms before the warmup was added**, against ~10 ms
  for the second. That is CUDA's lazy kernel initialisation, which happens on
  first *use*, not at `load_state_dict`. After adding a discarded warmup forward
  pass in the lifespan: first call 16.86 ms, then ~12.5 ms. Two reasons this
  mattered: the first scan of a live demo would appear to hang, and the research
  logs `compute_ms` as the server-compute term, so an unwarmed first request
  would write a figure ~350x the true one that is indistinguishable from a real
  measurement after the fact.
- **Server compute through the endpoint is ~12.5-12.9 ms, against 6.08-6.60 ms
  in a direct tight loop; the cause is not yet measured, but the live term is
  ~12.5 ms.** The experiment's cost model uses `server_compute_ms = 6.731`,
  which is the *offline* `birds.evaluation.evaluate` figure, so the live
  endpoint runs about 2x what the report's band boundaries assume. Ranked
  explanation, most likely first:
  1. **GPU clock state.** A tight loop holds SM and memory clocks at boost; a
     single isolated inference arriving at an otherwise idle GPU runs at a much
     lower clock. A ~2x spread on a ConvNeXt-Base forward is ordinary.
  2. **Measurement asymmetry.** If the tight loop synchronized once for N
     iterations and divided, it pipelined kernel launches and systematically
     understates per-request latency. The service synchronizes on both sides of
     every call, so the endpoint figure is the honest per-request number and the
     loop figure is the optimistic one.
  3. **The threadpool is effectively ruled out** (an earlier draft of this entry
     listed it). `run_in_threadpool` dispatch costs microseconds, the GIL is
     released inside the forward pass, and the warmup series (2224 ms, then
     16.86 ms, then ~12.5 ms) already localises the remaining cost to the GPU
     rather than the dispatch.

  Practical consequence: **~12.5 ms is what a phone sending spaced-out single
  photos actually experiences, so it, not 6.731 ms, is the live server-compute
  term.** Keep 6.731 ms labelled as the offline `birds.evaluation.evaluate`
  figure. Offload cost is network-dominated (28.6/75.8/202.3 ms for the three
  named conditions), so a ~6 ms shift moves band edges only slightly. Still
  unmeasured: the split between (1) and (2). Control experiment: re-measure
  against a running server, with a per-iteration-synchronized tight loop as the
  control, sampling `nvidia-smi --query-gpu=clocks.sm,clocks.mem` during both.
- **A ready-made demo example: `app/assets/birds/1116.jpg`.** Ground truth is
  Eastern Towhee. The small on-device model predicts American Redstart (wrong);
  the large server model predicts Eastern Towhee at 0.804 (right). One photo that
  shows why the offload path earns its keep, without appealing to aggregate
  accuracy.

## Task 6 — request log
- The ephemeral test DB already has every table (`Base.metadata.create_all`), so a "missing table" failure test must drop the table first (`ClassifyRequest.__table__.drop(engine)`). Tests never exercise migration 0005; it was matched to the model by hand.
- Instances are detached after `session.commit()` (expire_on_commit); give the row an explicit id in round-trip tests rather than reading `obj.id` afterward.
- `log_classify_request(..., engine=None)` takes an optional engine for tests; the lazy module engine is only cached on a successful build.

## Task 7 — wiring and integration test
- The classify route now takes `get_request_logger` (returns `log_classify_request`) and `get_request_start` (a `perf_counter` stamp) as dependencies. Tests bind the ephemeral engine with `dependency_overrides[get_request_logger] = lambda: partial(log_classify_request, engine=engine)`; production is untouched.
- The shared `app` fixture overrides the logger with a no-op, so existing tests never write to a real database through `DATABASE_URL` (the conftest loads `database/.env`). Tests that want the real writer override it back.
- `total_ms` starts at the `get_request_start` dependency, declared first so it resolves before the body read; it ends right after response shaping, before the log call itself.
- Unreachable-DB test: engine on 127.0.0.1:1 with `connect_timeout=1` fails fast; the response is still 200.

## Review fixes B1/S5/S6
- `config.py` loads `server/.env` then `database/.env`; `load_dotenv` never overrides, so shell > server > database. Constants `SERVER_ENV_PATH`/`DATABASE_ENV_PATH` are pinned in `tests/core/test_wiring.py`.
- `RequestLogger` Protocol lives in `request_log.py` (providers imports it); the real writer has an extra optional `engine` kwarg and still satisfies it.
