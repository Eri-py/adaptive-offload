
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
