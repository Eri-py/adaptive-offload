# Learnings — flower-model-finetuning

## Task 2 — mypy on `model.classifier[i]` assignment

Indexing/assigning into a torchvision model's `.classifier` (a
`nn.Sequential`) fails strict mypy even though `torchvision.*` has
`ignore_missing_imports = true`: the attribute type resolves through
`nn.Module.__getattr__` (`Tensor | Module`, not `Any`), so `classifier[i]`
and reassigning it hit `union-attr`/`operator`/`assignment` errors, not just
a simple `no-any-return`. Fix: `cast(nn.Sequential, model.classifier)` once,
then index/assign normally (`nn.Sequential` has proper `__getitem__`/
`__setitem__` stubs in torch, so no further casts are needed) — cleaner than
scattering `# type: ignore` comments per line, and it stays correct if the
classifier's exact submodule types change later.

## Task 3 — AMP choice, seeding, and patching a private module function

- **AMP choice: bf16, no `GradScaler`.** `fit`'s `torch.autocast` uses
  `dtype=torch.bfloat16` when `device.type == "cuda"` (a no-op autocast on
  CPU, harmless for tests). bf16's exponent range matches fp32's, so unlike
  fp16 it can't underflow to zero in the backward pass and needs no loss
  scaling — one less moving part than the fp16 + `GradScaler` combo, and the
  guidelines' "keep it simple" steer points the same way for a single-semester
  prototype.
- **`fit`'s epoch numbering is 1-based** (`FitResult.best_epoch`,
  `EpochResult.epoch`), matching the printed "epoch N/epochs" progress line
  and human-readable references like "the checkpoint from epoch 2".
- **Extended two files outside Task 3's own list, both flagged in the task
  brief as acceptable minimal extensions:**
  - `config.py`: added `TrainSettings` (a frozen dataclass) and
    `TRAIN_SETTINGS: dict[ModelName, TrainSettings]` with the small/large
    epochs/lr/weight_decay/batch_size table from the spec.
  - `data.py`: `make_data_loader` gained an optional `generator:
    torch.Generator | None = None` parameter, passed straight through to
    `DataLoader`, so `train.main()` can seed the shuffle order
    reproducibly. `shuffle=False` callers are unaffected.
- **Patching a private module function for a scripted-accuracy test, without
  importing `pytest`:** `fit` calls `_evaluate_accuracy(...)` as a bare
  module-level name inside its loop body, so a test can do
  `train_module._evaluate_accuracy = fake_fn` (save/restore the original in
  `try/finally`) and `fit` picks up the replacement on its next call, since
  Python resolves that name against the module's `__dict__` at call time, not
  at `fit`'s definition time. This avoids needing the `monkeypatch` fixture
  (which would pull in a `pytest.MonkeyPatch` type annotation and thus a
  `pytest` import) for a case this simple — matches the task's "no pytest
  import" steer for this file, while `test_data.py`/`test_models.py` (Tasks
  1–2) do import `pytest`, for `monkeypatch.setattr`/`pytest.raises`.

## Task 4 — Evaluation: naming, precision, and a mypy generic mismatch

- **`test_accuracy` lives happily in `flowers/evaluate.py` under its literal
  spec name.** pytest only collects from files matching `test_*.py`/
  `*_test.py`, and `evaluate.py` doesn't match that pattern, so the name
  itself is never a problem there. The actual hazard is one level removed:
  `test_evaluate.py` doing `from flowers.evaluate import test_accuracy`
  would pull a module-level callable named `test_accuracy` into a
  `test_*.py` file's namespace, and pytest *would* collect and try to run
  that (with no arguments, since it can't know to pass `model`/`loader`/
  `device`) — an immediate `TypeError` failure. Fix: `from flowers import
  evaluate as ev` and call `ev.test_accuracy(...)`, matching the plan's own
  suggested workaround. No renaming needed.
- **Accuracy uses plain fp32, no autocast** — the opposite tradeoff from
  training's per-epoch bf16 autocast (Task 3's learning). This runs once per
  model over the full test split, so there's no repeated-cost to amortize by
  trading precision for speed; fp32 keeps the reported number the simplest
  to reason about.
- **Eval batch size reuses `TRAIN_SETTINGS[name].batch_size`** rather than
  introducing a separate eval-batch-size constant — one less knob, and batch
  size only affects accuracy-computation throughput, not the result (which
  is summed unweighted across all items regardless of batch grouping).
- **`mean_latency_ms`'s warm-up cycles through the dataset with `i %
  len(dataset)`** so a `warmup` larger than the dataset (plausible in tests
  with a handful of synthetic items) doesn't `IndexError`. Timed items still
  iterate the full dataset once, un-cycled.
- **mypy: don't explicitly annotate a `DataLoader[...]` variable built from
  `TensorDataset`.** `TensorDataset(logits, labels)` is a
  `Dataset[tuple[Tensor, ...]]` (variadic), not `Dataset[tuple[Tensor,
  Tensor]]`; annotating the `DataLoader` variable with the two-tuple form
  produces a mypy `arg-type` error at the `DataLoader(...)` call. Left
  uninferred (no explicit annotation), mypy infers the correct variadic type
  on its own with no error — simpler than fighting the stub.
