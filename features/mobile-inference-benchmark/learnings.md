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
