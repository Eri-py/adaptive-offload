# mobile-inference-benchmark (app)

On-device TFLite inference benchmark for iOS, built with Expo. See
`features/mobile-inference-benchmark/spec.md` in the repo root for the full
feature spec.

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

## Preview build (for benchmark runs)

1. `npx eas-cli build --platform ios --profile preview`
2. Install it from the link EAS Build prints when the build finishes.

This build is a standalone Release binary with the JS bundle embedded, so it
runs with the phone in airplane mode and needs no dev server.
