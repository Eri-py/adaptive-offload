
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
