
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
