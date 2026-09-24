# mobile-inference-benchmark (app)

On-device TFLite inference benchmark for iOS, built with Expo. See
`features/mobile-inference-benchmark/spec.md` in the repo root for the full
feature spec.

## Building and installing on a physical iPhone

This app must run as a real, standalone build — not a Metro-connected dev
client — because one of its acceptance criteria (AC4) requires it to launch
and run correctly in airplane mode, with no dev server reachable. Use the
`preview` EAS Build profile (`app/eas.json`), which produces an internal
distribution build with the Release configuration and the JS bundle
embedded in the binary itself.

1. Install `eas-cli` ad hoc (no need to add it as a project dependency):
   `npx eas-cli --version`
2. Log in: `npx eas-cli login`
3. Register the target iPhone's UDID for internal distribution (one-time,
   per device): `npx eas-cli device:create`
4. Build: `npx eas-cli build --platform ios --profile preview`
5. Install the resulting build onto the registered device following the
   link EAS Build prints when the build finishes.

Note: there is no `development` profile in `app/eas.json`. A dev-client
(`developmentClient: true`) build fetches its JS bundle from a running
`expo start` over the network, so it cannot launch in airplane mode and
would fail AC4 outright; it also runs JS in `__DEV__` mode, which isn't a
representative binary for benchmarking. If a dev-client build is ever
genuinely needed for local iteration, add `expo-dev-client` as a real
dependency first and add the profile back deliberately — don't rely on
EAS's interactive prompt to install it implicitly.
