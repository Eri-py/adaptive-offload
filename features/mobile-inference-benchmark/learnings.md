# Learnings — mobile-inference-benchmark

## Task 1 — TFLite export blocked by a broken shared venv (torch/torchvision/litert-torch conflict)

`training/export_tflite_model.py` calling `model.export(format="tflite")` fails in the
current shared `.venv/` with:

```
AttributeError: '_OpNamespace' 'aten' object has no attribute 'cholesky'
```

raised deep inside `torch/distributed/tensor/_ops/_math_ops.py` while importing
`litert_torch` (this ultralytics version silently redirects the deprecated
`format="tflite"` to `format="litert"`, which now converts via the `litert_torch`
package instead of the old ONNX→TensorFlow→tflite path — there is no more
TensorFlow-based tflite export in this ultralytics version to fall back to).

Root cause: the venv has an unresolvable version conflict, not a script bug —
- `torchvision==0.29.0` (installed) declares `Requires-Dist: torch (==2.14.0)`.
- `litert-torch==0.9.4` (installed, needed for the litert/tflite export path)
  declares `Requires-Dist: torch<2.14.0,>=2.4.0`.
- Installed `torch==2.13.0` satisfies neither package's actual expectation
  cleanly, and the resulting internal inconsistency is what surfaces as the
  `aten.cholesky` AttributeError (a torch-internal op-registry lookup that only
  makes sense for a torch build torchvision/litert_torch didn't actually get).
- These two pins are also mutually exclusive as of current PyPI releases
  (torchvision 0.29.0 wants exactly 2.14.0; litert-torch 0.9.4 forbids 2.14.0),
  so this cannot be fixed by adjusting torch's version alone without also
  moving one of the other two packages — a change to the shared venv's core ML
  stack, out of scope for a single asset-export task. Also worth noting:
  there is currently no `training/pyproject.toml` in the working tree (only a
  stale `training.egg-info/`), so there's no manifest to pin against without
  first reconstructing it.
- Did not attempt to fix by installing/upgrading packages in the shared venv —
  that's a cross-cutting environment change beyond this task's scope, flagged
  to the orchestrator instead.

Image bundling (`bundle_sample_images()`) is independent of this and completed
successfully — 15 real `app/assets/images/*.jpg` files, verified with PIL
`Image.verify()` plus a full re-decode.

## Task 1 follow-up — ultralytics pin fixed the litert/torch conflict, but export now blocked by a second, unrelated conflict (tensorflow vs. scipy/numpy)

Applied the chosen fix: pinned `training/pyproject.toml`'s dependency to
`ultralytics>=8.0,<8.4.83` (below the 8.4.83 litert cutover). Reinstalling
(`pip install -e ./training` in the shared venv) resolved to
`ultralytics==8.4.82`. As a side effect of pip's resolver satisfying
`torchvision==0.29.0`'s exact `torch==2.14.0` requirement, `torch` was bumped
2.13.0 → 2.14.0 (not touched directly — this is what actually fixed the
original torch/torchvision self-inconsistency noted in the entry above,
incidentally). `litert-torch` is still installed but is no longer imported by
this export path at all, so its now-unsatisfied `torch<2.14.0` requirement
(visible in `pip check`) is inert for this script.

This confirmed the original diagnosis: with `format="tflite"` no longer
silently redirected to the litert path, ultralytics 8.4.82 uses its legacy
ONNX→TensorFlow SavedModel→TFLite exporter instead, and no longer touches
`litert_torch`/torch-version-sensitive code at all.

However, the export still does not succeed, for a second and completely
unrelated reason discovered only now: the legacy exporter auto-installs
`tensorflow` on first use (ultralytics' own `check_requirements` mechanism)
since it wasn't already present in the venv. It installed `tensorflow==2.19.0`
(taking ~9-10 minutes). Importing it then fails:

```
AttributeError: module 'numpy' has no attribute '_no_nep50_warning'
```

raised from inside numpy's own bundled `numpy/testing/_private/utils.py`
(installed numpy is 2.1.3), reached via TensorFlow's legacy Keras
`feature_column` compat layer importing `scipy.sparse` (installed scipy is
1.18.1), which pulls in `scipy._external.array_api_compat`, which clones
`numpy.testing` and trips over this. Confirmed `tensorflow==2.19.0`'s own
declared numpy constraint (`numpy<2.2.0,>=1.26.0`) is *not* violated by
2.1.3 — this isn't a version-bound mismatch pip could have caught; it looks
like a real incompatibility between this specific scipy release's
`array_api_compat` shim and this specific numpy release's `numpy.testing`
internals, surfaced only through TensorFlow's legacy-Keras import chain.
Neither `numpy` nor `scipy` was touched or changed by this task; both are
exactly the same versions the rest of `training/` (datagen, router) already
depends on and passes 86/86 tests against — this failure is isolated to the
`tensorflow` import path, not a defect in the shared numpy/scipy install.

Did not attempt a fix (e.g. pinning `tensorflow`/`numpy`/`scipy`) — that's a
second cross-cutting shared-venv change, out of scope for the "smallest,
isolated fix" this corrective task was scoped to, and it isn't yet clear
which package's pin would need to move without risking the rest of
`training/`'s numpy/scipy-dependent code. Flagged to the orchestrator instead.
`app/assets/models/yolov8n.tflite` was not created; `app/assets/models/`
doesn't exist yet.

## Task 1 follow-up 2 — isolation (throwaway venv) was the fix; shared venv left untouched

Per orchestrator decision, stopped trying to make the shared venv's stack
work for this one-off conversion and isolated the export into a throwaway
virtualenv outside `.venv/` instead. `training/pyproject.toml`'s ultralytics
pin was reverted to unpinned `ultralytics>=8.0` by the orchestrator before
this task, since it was only ever needed for the (abandoned) shared-venv
approach.

Environment: `python3 -m venv` failed outright (`ensurepip` not available,
`python3.12-venv` apt package missing, no passwordless sudo available to
install it). Used `uv venv /tmp/tflite-export-venv` instead (uv was already
on PATH) — bundles its own resolver/installer so it never needed
`ensurepip`. `uv pip install --python /tmp/tflite-export-venv/bin/python
<pkgs>` stood in for `pip install` throughout.

Even in this completely clean venv, plain `pip install ultralytics` (latest,
8.4.161) reproduced the *original* attempt-1 conflict, not a fresh
resolution: ultralytics' own `check_requirements` AutoUpdate mechanism,
triggered at export time (not at install time), pulled in `litert-torch`
which demands `torch<2.14`, and downgraded the venv's torch from 2.14.0 to
2.13.0 to satisfy it — leaving the already-installed `torchvision==0.29.0`
(which demands exactly `torch==2.14.0`) inconsistent, reproducing the same
`AttributeError: '_OpNamespace' 'aten' object has no attribute 'cholesky'`
as the very first attempt. So a clean pip resolve at *install* time doesn't
help when the conflict is introduced by an *export-time* auto-install
ultralytics does internally — isolation alone wasn't sufficient here, the
version pin from attempt 2 was still necessary.

Applying that pin (`ultralytics<8.4.83`) in the isolated venv resolved to
`ultralytics==8.4.82` with a clean, mutually consistent `torch==2.13.0` /
`torchvision==0.28.0` pair (no conflict this time, since nothing else in
this venv forced `torch==2.14.0`). This unlocked the legacy TF exporter path
as expected, but three *more* issues surfaced in sequence, all fixed by
pinning within the isolated venv (never touching the shared one):

1. Same numpy/scipy incompatibility as before
   (`AttributeError: module 'numpy' has no attribute '_no_nep50_warning'`,
   surfaced via TensorFlow's legacy Keras → `scipy.sparse` → vendored
   `array_api_compat` import chain) — this reproduced even in a from-scratch
   venv once `tensorflow==2.19.0` auto-installed and forced numpy down to
   2.1.3, confirming this is a genuine numpy 2.1.3 / scipy 1.18.1
   incompatibility (numpy's own `numpy/testing/_private/utils.py` references
   `np._no_nep50_warning`, which numpy 2.1.3's `__init__.py` no longer
   exposes), not a symptom of the shared venv's other pins. Fixed by pinning
   `numpy==2.0.2` and `scipy==1.13.1` explicitly (still inside tensorflow's
   own declared `numpy<2.2,>=1.26` range) — this pair does not hit the bug.
2. A protobuf gencode/runtime mismatch
   (`google.protobuf.runtime_version.VersionError`): the freshly-installed
   `onnx` (latest, 1.23.0) was built against protobuf ≥6.31.1, but
   `tensorflow==2.19.0`'s own auto-install had already forced the venv's
   `protobuf` down to 5.29.6 (tensorflow 2.19 caps protobuf below 6). Fixed
   by pinning `onnx<1.18` (resolved to 1.17.0, built against protobuf 5.x)
   rather than touching protobuf/tensorflow.
3. `onnx2tf` (ultralytics' ONNX→TensorFlow bridge, also auto-installed) needs
   `tf_keras`, `sng4onnx`, and `onnx_graphsurgeon`, and ultralytics'
   auto-install for these specifically queried an NVIDIA package mirror
   (`https://pypi.ngc.nvidia.com`) that returned a DNS failure in this
   environment. Worked around by installing the same three packages
   directly (`pip install "tf_keras<=2.19.0" "sng4onnx>=1.0.1"
   "onnx_graphsurgeon>=0.3.26"`) from the default index instead, which
   resolved fine and didn't disturb numpy/scipy/protobuf.

With all of the above pinned, `export_tflite_model()` completed cleanly:
`app/assets/models/yolov8n.tflite` (12,865,855 bytes, ~12.3 MB) was created.
Verified it's a real, loadable TFLite model (loaded via
`tf.lite.Interpreter` from the isolated venv's own `tensorflow==2.19.0`):
input `images` is `[1, 640, 640, 3]` float32 (NHWC image), output
`Identity` is `[1, 84, 8400]` float32 — exactly matching the PyTorch model's
own reported output shape from the export log (4 bbox coords + 80 COCO
classes × 8400 anchors). Also cleaned up one stray side-effect file
(`calibration_image_sample_data_20x128x128x3_float32.npy`, downloaded by
ultralytics into the cwd during export, unrelated to any of the three
issues above and not a deliverable) from the repo root after verifying the
export.

Final versions that worked together in the isolated venv (`/tmp/tflite-export-venv`,
Python 3.12.14): `torch==2.13.0`, `torchvision==0.28.0`, `ultralytics==8.4.82`,
`onnx==1.17.0`, `onnxruntime==1.30.0`, `onnxslim==0.1.96`, `onnx2tf==1.28.8`,
`tf-keras==2.19.0`, `tensorflow==2.19.0`, `numpy==2.0.2`, `scipy==1.13.1`,
`protobuf==5.29.6`, `sng4onnx==2.0.1`, `onnx-graphsurgeon==0.6.1`.

## Task 2 — Windows-side node/npx quirks scaffolding into `app/`

- **`npx create-expo-app` (and any npx-resolved `.cmd` shim, e.g. bare `npx
  tsc`) cannot run with cwd inside the WSL-native repo path.** The Windows
  `node.exe` process (reached through `/mnt/c/Program Files/nodejs/npx`) gets
  its cwd via WSL interop as a UNC path
  (`\\wsl.localhost\Ubuntu-24.04\home\...`). Any step in the npx/`.cmd` chain
  that shells out through `cmd.exe` inherits that UNC cwd, and `cmd.exe`
  flatly refuses UNC as a working directory — it prints "UNC paths are not
  supported. Defaulting to Windows directory." and silently resets cwd to
  `C:\Windows`. For `create-expo-app` this surfaced as `EPERM: operation not
  permitted, mkdir 'C:\Windows\app'`; for `npx tsc` it silently ran with the
  wrong cwd and dumped the generic help text instead of type-checking
  anything (no tsconfig found at `C:\Windows`).
  - **Fix used:** do the actual scaffold in a directory under `/mnt/c/...`
    (a real Windows drive letter, not a UNC path — e.g.
    `/mnt/c/Users/<user>/AppData/Local/Temp/...`), where `cmd.exe`'s cwd
    resolves fine, then `cp` the generated files over into the repo's
    `app/` on the native Linux path. Confirmed no filename collisions with
    the `app/assets/images/`, `app/assets/models/` already bundled by Task
    1 (the template's own `assets/` only added icon/splash pngs).
  - For **plain JS-entry CLIs already installed in `app/node_modules`**
    (`tsc`, `expo`), the fix is simpler: skip the `.cmd`/npx shim entirely
    and invoke the Windows `node.exe` directly against the package's JS
    entry point, e.g.
    `"/mnt/c/Program Files/nodejs/node.exe" node_modules/typescript/bin/tsc --noEmit`
    or `.../node.exe node_modules/expo/bin/cli start`. This never spawns an
    intermediate `cmd.exe`, so the UNC cwd is preserved correctly and the
    command runs against the real project files. `npm install` itself is
    unaffected by this (it's invoked as `node.exe npm-cli.js` directly by
    the `npm` bash wrapper, no `cmd.exe` hop) — matches the task's note
    that `npm install` had already been validated on the native path.
- **`create-expo-app@latest` (SDK 57, React 19.2.3 / RN 0.86.3) defaults**:
  pass `--no-agents-md` to skip generating `AGENTS.md`/`CLAUDE.md`/
  `.claude/settings.json` inside `app/` (would otherwise shadow/duplicate
  the repo-root `CLAUDE.md` and plugin config). The template also drops its
  own MIT `LICENSE` file and a per-project `.git/` — both discarded when
  copying into `app/` (a nested git repo under this repo's `app/` would be
  wrong, and the Expo-authored LICENSE doesn't apply to this project).
- **`App.tsx` return type**: with React 19 types + `"jsx": "react-jsx"`
  (from `expo/tsconfig.base`), the bare global `JSX` namespace is no longer
  auto-available — `(): JSX.Element` fails with `TS2503: Cannot find
  namespace 'JSX'`. Used `import type { ReactElement } from 'react'` and
  `(): ReactElement` instead.
- **Real environment blocker (unresolved) — Metro's file watcher cannot
  watch a UNC directory.** Even invoking `node.exe` directly (bypassing the
  `cmd.exe`/UNC-cwd problem above), `expo start` crashes shortly after
  "Starting Metro Bundler" with:
  ```
  Error: EISDIR: illegal operation on a directory, watch
  '\\wsl.localhost\Ubuntu-24.04\home\eriol\projects\adaptive-offload\app'
  ```
  from `@expo/metro-file-map`'s `FallbackWatcher` calling Node's `fs.watch`
  on the project root. This is a real crash, not a shim/cwd artifact — the
  Windows-side Node.exe's `fs.watch` implementation does not support
  recursive-watching a directory reached via a `\\wsl.localhost\...` UNC
  path. `npx tsc --noEmit`'s one-shot type-check doesn't hit this (it reads
  files once, doesn't watch), so Task 2's first success criterion is
  cleanly met, but the second (`expo start` reaching a ready
  state/crash-free) is **not** met as of this task, given the current
  Windows-node-against-WSL-native-path setup. Flagging to the orchestrator
  rather than attempting a fix — candidates for a real fix (untried, all
  out of scope for this task): install Watchman (uses a different watch
  backend that may handle UNC), run the dev server from a Linux-native
  Node.js instead of the Windows-side one (the environment note says none
  is set up), or keep `app/` on `/mnt/c/...` for `expo start`/`eas build`
  purposes while the rest of the repo stays on the native Linux path.

Confirmed the shared venv was never touched: `training`'s
`ruff check . && mypy . && pytest -q` re-run from the shared `.venv/` after
this task is still clean, same 86/86 passing as before.

## Task 2 follow-up — fixed the Metro/UNC crash: installed a native Linux Node

Took the orchestrator's suggested fix from the list above: installed a
Linux-native Node.js (v24.21.0 LTS, official prebuilt tarball, no root
needed) to `~/.local/node/`, rather than continuing to rely on the
Windows-side `node.exe` reached through `/mnt/c/`. This removes the WSL/UNC
boundary entirely — Node now executes on the same side of the filesystem as
the project files, so `fs.watch` never crosses into `\\wsl.localhost\...`
territory in the first place.

Getting this to actually take effect for future shell/tool invocations
needed some care: this session's shell doesn't source `~/.bashrc` for
non-interactive invocations (Ubuntu's default `.bashrc` `return`s early for
non-interactive shells, before reaching an appended `PATH` line), and a
`~/.profile` edit worked for an explicit login shell (`bash -lc`) but *not*
for how each Bash-tool call is actually invoked in this environment — some
third mechanism, not a profile-sourcing login/interactive shell. The
reliable fix: symlink `node`/`npm`/`npx` directly into `~/.local/bin/`,
which was already confirmed on `PATH` from the very start of this session
regardless of shell-invocation mode. `app/node_modules` was reinstalled
clean under the native Node (467 packages, no errors) to avoid keeping any
native-module artifacts built against the old Windows Node.

Verified fixed: `npx expo start` now reaches `Starting Metro Bundler` →
`Waiting on http://localhost:8081` and stays up (killed manually via
`timeout` after confirming the ready state) — no `EISDIR`/UNC crash. This
unblocks Tasks 3, 4, 5, and 7, all of which need a working Metro/dev-server
run to verify.

## Task 3 — TFLite native module: what could and couldn't be verified without a device/Mac

`react-native-fast-tflite@3.0.1` installed cleanly (`npm install
react-native-fast-tflite`), and its `enableCoreMLDelegate` config-plugin key
(checked against the installed version's own README, since the task noted
the exact key name might differ by version) matched exactly — no adjustment
needed to the plan's assumed `["react-native-fast-tflite", {
"enableCoreMLDelegate": true }]` shape.

- **`react-native-nitro-modules` is a required peer dependency, not bundled.**
  The library is built on Nitro Modules; npm auto-installs the peer into
  `node_modules` but does *not* add it to `package.json`'s own
  `dependencies` unless installed explicitly. Per the library's own README
  install step 1 (`yarn add react-native-fast-tflite
  react-native-nitro-modules`), installed it explicitly
  (`npm install react-native-nitro-modules`, resolved to `0.37.1`) rather
  than relying on the transitive/peer resolution — otherwise a clean
  `npm ci` elsewhere (e.g. an EAS Build container) could resolve a different
  compatible-but-untested version, or fail to resolve it at all depending on
  npm's peer-install behavior in that environment.
- **`metro.config.js` (new file, not in this task's original Files list) is
  required for `require('*.tflite')` to bundle at all.** The library's
  README installation step 2 says Metro needs `tflite` added to
  `resolver.assetExts` for `.tflite` files to be treated as a bundleable
  asset (same mechanism RN uses for images/fonts). Without it, any
  `require('assets/models/yolov8n.tflite')` call — including the one Task
  4's real hook will make — fails at Metro's bundling stage. Treated this as
  in-scope for "wire the library into the project" (it's the library's own
  required install step, immediately following the `npm install` this task
  already does) rather than out-of-scope scope creep; flagging it here since
  it wasn't in the plan's original file list for this task.
- **No ambient `.d.ts` needed for `require('*.tflite')` to type-check.**
  Worried initially that TS would need a `declare module '*.tflite'`
  (there's precedent for this pattern — `node_modules/expo/types/global.d.ts`
  does it for `*.css`/`*.sass`/`*.scss`). Tested directly: `require(...)` of
  an arbitrary/even nonexistent file extension already type-checks with no
  error under this project's tsconfig (`expo/tsconfig.base` + `strict:
  true`), confirming RN/Metro's global `require` is typed as accepting/
  returning `any` in this project as-is (not narrowed per extension) — TS's
  static module-existence checking only applies to `import` statements, not
  bare `require(...)` calls. `loadTensorflowModel`'s real type signature
  (checked directly in
  `node_modules/react-native-fast-tflite/lib/typescript/loadTensorflowModel.d.ts`)
  is `(source: number | { url: string }, delegates: TensorflowModelDelegate[]) =>
  Promise<TfliteModel>` — a `require(...)` result (`any`) is assignable to
  the `number` arm without a cast.
- **No CocoaPods/Xcode/simulator available in this Linux environment** (no
  `pod`/`ruby`/`gem` on PATH, confirmed) — matches the task's own expected
  limitation. What this made unverifiable: whether `loadTensorflowModel`
  actually loads `yolov8n.tflite` and `model.run(...)` actually returns real
  output tensors on-device. That requires a real iOS build (simulator or
  physical device), which is Task 6's/the user's territory (EAS Build).
- **What was verified instead, as real evidence the wiring is correct (not
  just "looks plausible")**:
  1. `npx expo config --json` resolves the plugin correctly —
     `_internal.pluginHistory` lists `{"react-native-fast-tflite": {"version":
     "3.0.1"}}`, confirming Expo's config-plugin system actually ran the
     plugin against `app.json`, not just that the JSON was well-formed.
  2. `npx expo prebuild --platform ios --no-install` (the `--no-install`
     flag skips the `pod install` step CocoaPods would need) succeeded and
     generated a real `ios/` project. Its `Podfile` has
     `$EnableCoreMLDelegate=true` at the very top (exactly the flag the
     plugin is documented to set per the README's "Bare React Native"
     section) and uses standard `use_native_modules!`/`use_expo_modules!`
     autolinking — `react-native-fast-tflite`'s and
     `react-native-nitro-modules`'s own `.podspec` files are present in
     `node_modules`, so `pod install` (unavailable here) would discover them
     automatically at that point; nothing further to configure. `expo
     prebuild` also auto-filled `ios.bundleIdentifier` in `app.json`
     (`com.anonymous.mobile-inference-benchmark`, generated since none
     existed) as a required side effect of generating a real iOS project —
     left in place since it's needed for any future prebuild/EAS build
     anyway (Task 6), harmless, and squarely inside `app.json`.
  3. A throwaway `app/verify-tflite.ts` (removed before finishing, per the
     task) called `loadTensorflowModel(require('./assets/models/
     yolov8n.tflite'), [])` then `model.run([input.buffer])` with a
     correctly-shaped `Float32Array(1*640*640*3)` input (matching the
     `[1,640,640,3]` float32 NHWC input tensor confirmed in Task 1's
     `tf.lite.Interpreter` check) — this type-checked cleanly against the
     library's real `.d.ts` files (`npx tsc --noEmit`, zero errors),
     confirming the exact call shape Task 4's hook will use is valid against
     the installed version's real API, not just written from README memory.
     Actual execution/output was not observed (no device/simulator) — this
     file proves the call is well-typed and structurally correct, not that
     it succeeds at runtime.
  4. `npx expo start` still reaches `Waiting on http://localhost:8081`
     cleanly after all of the above (new deps + `metro.config.js` + config
     plugin) — no regression from Task 2's baseline.
  5. Removed the generated `ios/` directory again after verification (it's
     gitignored per `app/.gitignore`'s `/ios` entry, and dropping it
     restores the managed-workflow state Tasks 4/5/7 expect — a native
     project auto-persisting from this task could otherwise silently change
     how `expo start`/`tsc` behave for later tasks).
- **No lint step exists yet for `app/`** — `package.json` has no `lint`
  script and there's no ESLint config in `app/` as of Task 2's scaffold, so
  the quality gate for this task was `npx tsc --noEmit` only (clean, zero
  errors/warnings). Flagging in case a later task is expected to add
  ESLint — it isn't there yet.

## Task 4 — Benchmark hook: image-to-tensor preprocessing needed real new dependencies, and `@types/jpeg-js` turned out to be for the wrong API version

Nothing suitable for decoding a bundled JPEG into raw pixel data (and
resizing it) was already present in `app/` — confirmed by checking
`node_modules` top-level and `expo`'s own declared dependencies. Added three
real, justified new dependencies rather than over-engineering around their
absence:

- `expo-asset` and `expo-file-system` — both already ship *inside* `expo`'s
  own `node_modules` (nested, e.g.
  `node_modules/expo/node_modules/expo-asset`) since `expo` itself depends
  on them, but a nested `node_modules/expo/node_modules/expo-asset` is not
  resolvable via `require('expo-asset')` from app-level code (Node/Metro
  resolution only walks up through node_modules directories visible to the
  requiring file, not sideways into another package's private nested
  deps) — relying on it would also be fragile even if it happened to hoist,
  since it isn't declared in `app/package.json`. Installed both explicitly
  via `npx expo install expo-asset expo-file-system` (not plain `npm
  install`) so their versions are the ones Expo SDK 57 actually declares as
  compatible (`expo-asset@~57.0.18`, `expo-file-system@~57.0.7`) — this
  matched Task 3's own install method for `react-native-fast-tflite`'s peer.
  `expo-file-system@57` ships a new `File`/`Directory` API (the SDK 54+
  redesign) whose `File.arrayBuffer()` returns a raw `ArrayBuffer` directly
  from a `file://` URI — this avoided an entire unnecessary
  read-as-base64-then-decode round trip that the older/legacy
  `expo-file-system` API would have required.
  - **Side effect worth flagging**: `expo install expo-asset` auto-added
    `"expo-asset"` to `app.json`'s Expo config-plugins list (a one-line
    diff) — not in this task's original `Files` list, same situation Task 3
    flagged for `metro.config.js`: a necessary, automatic install side
    effect of wiring in a real dependency, not scope creep. `app/app.json`
    and `app/package.json`'s file *mode* also flipped 755→644 as an
    incidental `npm`/`expo` rewrite side effect (content changes are the
    only ones that matter; not fixed, harmless).
- `jpeg-js` (pure-JS JPEG decoder, no native module — works on the Hermes JS
  thread with no platform-specific code, so nothing here depends on a
  device/simulator build to *type-check* correctly) — decodes the raw JPEG
  bytes from `File.arrayBuffer()` into an RGBA `Uint8Array` pixel buffer.
  - **`@types/jpeg-js` (0.3.x on npm) documents the wrong API and was
    deliberately *not* installed.** It types `decode`'s second argument as a
    plain `boolean` (the old jpeg-js 0.3 signature). The actual installed
    runtime (`jpeg-js@0.4.4`, confirmed against its own
    `node_modules/jpeg-js/README.md`) takes an *options object*
    (`{ useTArray, formatAsRGBA, ... }`) instead — installing
    `@types/jpeg-js` would have type-checked cleanly against an API this
    version doesn't actually have, and silently miscompiled at the call
    site. Installed it once to inspect, confirmed the mismatch, then
    uninstalled it and wrote a small ambient declaration
    (`app/types/jpeg-js.d.ts`) matching the *real* 0.4.4 `decode(data,
    options?)` shape instead, with a comment explaining why the
    DefinitelyTyped package was skipped (so a future contributor doesn't
    "helpfully" reintroduce it).
- **Preprocessing approach chosen**: nearest-neighbor resize (hand-written,
  no resize library needed) from the decoded image's native resolution
  down/up to the model's required 640×640, reading only the R/G/B channels
  of jpeg-js's default RGBA output (alpha dropped) and normalizing each to
  `[0, 1]` by dividing by 255, written directly into a NHWC-ordered
  `Float32Array`. This is a real, correct-shape preprocessing pass, not a
  stub — but per the task's own scope note, it makes no attempt at
  pixel-exact parity with `training/datagen/simulate/yolo_inference.py`'s
  preprocessing, since this benchmark only measures `model.run()` latency
  and never compares detection output.
- **`Float32Array.prototype.buffer` is typed `ArrayBufferLike`, not
  `ArrayBuffer`** (TS's lib.es2017 typed-array types allow a
  `SharedArrayBuffer`-backed view), which doesn't structurally satisfy
  `TfliteModel.run(input: ArrayBuffer[])`. Fixed with a narrow, commented
  `as ArrayBuffer` cast at the one call site — safe here because the array
  is always freshly allocated by this hook's own `resizeAndNormalize`
  (never a view onto a `SharedArrayBuffer`), not a blanket type-safety
  hole.

**What was verified**: `npx tsc --noEmit` is clean (zero errors) against the
hook's real code, the real (non-stubbed) `.d.ts` types for
`react-native-fast-tflite` (reused from Task 3's already-verified API
shape — `loadTensorflowModel(source, delegates)` returning `Promise<TfliteModel>`,
`TfliteModel.run(input: ArrayBuffer[]): Promise<ArrayBuffer[]>`), the real
`expo-asset`/`expo-file-system` v57 `.d.ts` types, and the hand-written
`jpeg-js` ambient declaration. Traced both the success path (idle → running
→ done, `results` populated) and the failure path (any thrown error at any
stage — asset download, JPEG decode, model load, or `model.run()` itself —
is caught by the single `try/catch` wrapping the whole `runBenchmark` body,
surfaced as a string via `error` state, `status` set to `'error'`, nothing
swallowed silently) by manual code review; this satisfies the task's two
success criteria structurally.

**What necessarily still awaits a real device test (Task 6/7 territory, no
Mac/device available in this environment, same limitation Task 3
documented)**: whether `loadTensorflowModel` actually loads the bundled
`.tflite` file at runtime, whether `Asset.downloadAsync()` +
`File.arrayBuffer()` actually produce readable bytes for a bundled JPEG on a
real iOS build, whether `jpeg-js`'s pure-JS decode is fast enough on-device
to not dominate the measured wall-clock loop in an unacceptable way (it
isn't timed — only `model.run()` is timed — but it does still gate how long
a full 15-image benchmark run takes end to end), and whether `model.run()`
actually returns without throwing for this specific model/input shape.

## Task 6 — EAS build configuration, and how to validate `eas.json` without live credentials

- **`eas-cli` was not installed anywhere** (no local devDependency, no
  global install). Deliberately did *not* add it as an `app/`
  devDependency — the task's `Files` list is `app/eas.json` only, and
  `eas-cli` is a developer-machine tool invoked ad hoc (`eas build ...`),
  not a runtime/build dependency of the app bundle itself; Expo's own docs
  treat `npx eas-cli` (or a global install) as the normal path rather than
  pinning it in `package.json`. Used `npx eas-cli` throughout, which
  auto-installed v24.7.0 into npm's `_npx` cache on first invocation — no
  permanent change to `app/` beyond `eas.json`.
- **`eas build:configure` and `eas config` both require an authenticated
  Expo session** (confirmed by actually running `npx eas-cli config
  --platform ios --profile development --json --non-interactive`, which
  failed cleanly with "An Expo user account is required to proceed" —
  not a schema/config error, i.e. it got past parsing `eas.json` before
  hitting the auth wall). So neither can be used as this task's local
  validation step, as the task brief anticipated.
- **Found a real, unauthenticated validation path anyway**: `eas-cli`
  depends on the separate `@expo/eas-json` package to parse and resolve
  `eas.json`, and that package is plain Node with no network/auth calls
  in its read path. Located it inside the npx cache
  (`~/.npm/_npx/<hash>/node_modules/@expo/eas-json`) after `npx eas-cli`
  had already pulled it in, and drove it directly from a throwaway Node
  script: `EasJsonAccessor.fromProjectPath(<app dir>)` +
  `EasJsonUtils.getBuildProfileAsync(accessor, Platform.IOS,
  'development')`. This runs the actual schema validation and profile
  merge logic the real CLI uses, not a hand-rolled guess at the schema —
  it resolved the `development`/iOS profile to
  `{ credentialsSource: "remote", distribution: "internal",
  developmentClient: true, simulator: false }` and enumerated all three
  profile names (`development`, `preview`, `production`) cleanly, which
  is about as strong a local validation as is possible without live
  credentials. Worth reusing this trick (`@expo/eas-json`'s
  `EasJsonAccessor`/`EasJsonUtils`) for any future `eas.json` change in
  this project rather than re-deriving the schema from memory or docs.

## Review finding B2 (removed the `development` EAS profile, added app/README.md)

- Removed the `development` block from `app/eas.json` entirely (orchestrator
  decision: no dev-client build wanted). `app/package.json` was confirmed to
  have no `expo-dev-client` dependency, matching the finding's observation
  that the old profile would have hit EAS's interactive
  install-it-or-fail-at-runtime prompt.
- Re-validated with the same unauthenticated `@expo/eas-json` trick Task 6's
  entry documented (`EasJsonAccessor.fromProjectPath` +
  `EasJsonUtils.getBuildProfileAsync`), reusing the already-cached package
  at `~/.npm/_npx/e25a38a8cc65d08e/node_modules/@expo/eas-json` rather than
  re-triggering an `npx eas-cli` pull. `preview` (ios) resolves to
  `{ credentialsSource: "remote", distribution: "internal", simulator: false
  }` and `production` (ios) to `{ credentialsSource: "remote", distribution:
  "store", autoIncrement: true }`, both clean. Also confirmed
  `getBuildProfileAsync(..., "development")` now throws "Missing build
  profile in eas.json" with the profile list explicitly excluding
  `development` — i.e. the removal actually took effect in the schema the
  real CLI would resolve, not just visually in the JSON.
- Added `app/README.md` (didn't exist before) with the build/install
  sequence: `eas device:create` (one-time UDID registration required before
  an internal-distribution build can install on a device) → `eas build
  --platform ios --profile preview`, plus a short note on why there's no
  `development` profile (dev-client can't launch in airplane mode, breaking
  AC4) and what to do if one is ever genuinely needed later (add
  `expo-dev-client` as a real dependency deliberately, don't rely on EAS's
  interactive prompt).
- Corrected `features/mobile-inference-benchmark/progress.md`'s Task 6 entry,
  which previously told the user to run `eas build --platform ios --profile
  development` — updated to point at `preview` and mention
  `eas device:create` and `app/README.md`.
- No code/type-checked surface changed (`eas.json` is data, not TS), so
  `npx tsc --noEmit` was re-run only as a no-regression check (clean, zero
  output/errors). `app/package.json` still has no `lint` script (same gap
  Task 3 already flagged) — nothing to run there.

## Review finding B1 (app-side warm-up)

- `react-native-fast-tflite`'s `TfliteModel.run(input: ArrayBuffer[]):
  Promise<ArrayBuffer[]>` accepts any correctly-shaped `ArrayBuffer` — a
  zero-filled `Float32Array(640*640*3).buffer` is fine as a throwaway
  warm-up input, no need for a real decoded image.
- Placed the warm-up call inside the `model == null` branch in
  `useBenchmark.ts`, right after `loadTensorflowModel` and before caching
  into `modelRef`. This runs the discarded inference exactly once per
  model load (not once per benchmark run), which is the right cadence
  since `modelRef` persists the loaded model across repeated
  `runBenchmark()` calls — only the very first call after a fresh load
  pays the one-off packing/compilation cost the finding describes.
