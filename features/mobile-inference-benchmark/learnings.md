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
