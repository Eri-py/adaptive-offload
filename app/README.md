# adaptive-offload (app)

The iOS client, built with Expo. It currently runs the bird-classifier spike:
the exported MobileNetV3-Large (`assets/models/small.tflite`) over 15
pre-cropped CUB photos, across two tensor layouts and two delegates, reporting
both latency and agreement with PyTorch's predictions on the same JPEG bytes.

The YOLOv8n benchmark this app began as has been removed. Its measurements are
recorded in `features/mobile-inference-benchmark/learnings.md`, and the code
remains in git history.

## Build profiles

`app/eas.json` has two profiles for the iPhone, for different jobs:

| Profile | Use it for | JS comes from | Works offline |
|---|---|---|---|
| `development` | Day-to-day work on the app (hot reload) | The dev server on your machine | No |
| `preview` | Benchmark numbers you intend to report | Embedded in the binary | Yes (AC4) |

Report latency numbers from a `preview` build only. A `development` build runs
the Debug configuration with React Native's dev-mode checks on, so its timings
are slower than a real release and aren't representative.

## One-time setup

1. Log in: `npx eas-cli login`
2. Register the iPhone's UDID for internal distribution:
   `npx eas-cli device:create`

## Development build (for iterating)

1. Build and install once: `npx eas-cli build --platform ios --profile development`.
   Only rebuild when native dependencies or `app.json` plugins change; JS and
   UI changes reload over the dev server.
2. Start the dev server from `app/`, then open the dev-client app on the phone
   and connect to it.

The phone must be able to reach the dev server. WSL2 in its default NAT
networking mode is not reachable from other devices on the LAN, so either:

- run `npx expo start --tunnel` (works through any network, slightly slower
  reloads), or
- switch WSL to mirrored networking (`networkingMode=mirrored` under `[wsl2]`
  in `%UserProfile%\.wslconfig`, then `wsl --shutdown`) and allow port 8081
  through the Windows firewall, then run `npx expo start` as usual.

`--tunnel` needs `@expo/ngrok`, which is now a devDependency.

**Tailscale is not a substitute here, despite being the route the server uses.**
Pointing the dev client at the tailnet address
(`REACT_NATIVE_PACKAGER_HOSTNAME=<tailscale-ip> npx expo start`) fails: iOS App
Transport Security blocks cleartext HTTP to Tailscale's CGNAT range
(`100.64.0.0/10`), which iOS does not count as a local network, so Expo's
default `NSAllowsLocalNetworking` exception does not cover it. `--tunnel` works
because ngrok serves HTTPS. Any in-app call to the server over plain HTTP on a
tailnet address needs an ATS exception in `app.json` and a native rebuild.

## Preview build (for benchmark runs)

1. `npx eas-cli build --platform ios --profile preview`
2. Install it from the link EAS Build prints when the build finishes.

This build is a standalone Release binary with the JS bundle embedded, so it
runs with the phone in airplane mode and needs no dev server.
