# Review — Flower Model Fine-Tuning

## Verdict

The implementation meets the spec, and there are no blockers. The training and evaluation method holds up. `fit` has no parameter that could receive the test split. The checkpoint is chosen only by validation accuracy, computed in fp32 with each model's own eval transform. The test split is loaded only in `evaluate.py`, and timing measures only the batch-1 forward pass, with CUDA sync on the GPU. I re-ran `python -m flowers.evaluate` on the saved checkpoints:

- **Accuracy:** reproduced exactly (0.9221 small, 0.9467 large).
- **Latency:** within about 2% (5.658 vs 5.560 ms on the CPU, 6.698 vs 6.630 ms on the GPU).

The training logs in the implementer's scratchpad match the reported best validation epochs: small 94.41% at epoch 25, large 96.47% at epoch 13.

I also ran:

- the new tests: 15 passed
- `ruff check .`: clean
- `mypy .`: exactly the 9 known errors

I did not re-run the full suite, because it may need Postgres.

There are two main gaps:

- A fresh machine without the dataset would download ImageNet weights before the missing-dataset error fires (S1).
- A few pieces that could be tested without a GPU have no tests (S3).

The rest is polish.

## Acceptance Criteria

1. **Missing dataset: training stops with a message naming the data and the command, and downloads nothing.** PARTIAL.
   - The error itself is right. `training/flowers/data.py:89-97` catches torchvision's `RuntimeError` and raises `FileNotFoundError` naming `python -m flowers.download`. `tests/flowers/test_data.py:627-633` tests it.
   - The ordering is wrong. `training/flowers/train.py:521` calls `build_model(model_name, pretrained=True)` before building the data loaders (`:523`). On a machine with no torchvision weight cache, it downloads 22–350 MB of ImageNet weights before failing.
   - The plan counts ImageNet weights as "model setup, not dataset acquisition", so this is not a spec violation of the dataset rule. But the AC literally says "downloads nothing", and the fix is to reorder two lines (see S1).
2. **Download step puts the official splits in `training/data/`, and nothing there is tracked.** MET.
   - `training/flowers/download.py:169-172` downloads the three splits.
   - Progress records 1,020 / 1,020 / 6,149 images.
   - `git check-ignore` confirms `training/data/` is ignored (`.gitignore:15`).
3. **No GPU: training stops with a clear message.** MET.
   - `training/flowers/models.py:372-379` raises the error, and `train.py:518` calls it before any model or data work.
   - `tests/flowers/test_models.py:776-780` tests it.
4. **The kept checkpoint is chosen on validation only, and test is used only in the final evaluation.** MET.
   - `train.py:448-457` has no test parameter, and a signature test enforces this (`test_train.py:891-903`).
   - The checkpoint is saved on strict improvement in validation accuracy (`train.py:498-502`).
   - A scripted-accuracy test proves the best epoch, not the last one, is kept (`test_train.py:847-888`).
   - The test split is only loaded in `evaluate.py:296-313`.
5. **Both models' weights are saved under `training/models/`, and git shows them untracked or ignored.** MET.
   - `train.py:534` saves to `WEIGHTS_DIR / "<name>.pt"`.
   - Both files exist, and `*.pt` is ignored (`.gitignore:18`).
   - `git status` shows `training/models/` as untracked only because of the directory itself; the `.pt` files inside are ignored.
6. **Evaluation reports top-1 accuracy on all 6,149 test photos, and time per photo on CPU (small) and GPU (large).** MET.
   - The code is at `evaluate.py:287-320`.
   - I verified it by re-running (numbers above). The device label also records the CPU thread count.
7. **Tests cover everything that doesn't need a GPU or the real dataset; the suite passes, ruff is clean, and mypy shows only the 9 known errors.** PARTIAL.
   - Ruff and mypy are verified, and the 15 new tests pass.
   - A few paths that need no GPU are untested:
     - the missing-checkpoint error in `evaluate._load_checkpoint`
     - the `eval_transform` / `train_transform` wiring, which the plan's `test_data.py` row lists as "split and transform wiring"
     - the order in `train.main` (see S3)

## Scope

The diff matches the plan's Files Changed table. Everything outside the plan's file list was either pre-authorised or planned:

- `config.py` gained `TrainSettings` / `TRAIN_SETTINGS`.
- `data.py:make_data_loader` gained an optional `generator` parameter.

Both were pre-authorised small extensions in Task 3 and are recorded in `learnings.md`.

- The `torchvision` mypy override in `pyproject.toml:84-86` was a conditional step in the plan.

Nothing is out of scope. No `shared/` folder was created, and no new `[project.scripts]` were added.

## Blockers

None.

## Suggestions

#### S1 — Pretrained weights download before the missing-dataset check

- **File:** `training/flowers/train.py:521`
- **Issue:** `build_model(..., pretrained=True)` runs before `make_data_loader`. On a fresh machine with no dataset, training downloads ImageNet weights and only then fails. This contradicts AC1's "downloads nothing".
- **Fix:** Build `train_loader` / `val_loader` before calling `build_model` in `main()`. Loading the dataset is what raises the `FileNotFoundError`, so this surfaces the error before any network access.
- **Decision:** Accepted — carried into the birds revamp (features/birds-revamp/spec.md), which rewrites this code for CUB-200 and fixes this there

#### S2 — Duplicate top-1 accuracy function

- **File:** `training/flowers/train.py:432-445`, `training/flowers/evaluate.py:206-223`
- **Issue:** `train._evaluate_accuracy` and `evaluate.test_accuracy` are the same loop; the only difference is that the second calls `model.to(device)`. If one is later changed (for example, to add autocast), validation and test accuracy would silently be measured differently.
- **Fix:** Keep one implementation and import it in the other module. For example, `train.fit` could call `evaluate.test_accuracy`. `train.py` would then import from `evaluate.py`, which is fine because `evaluate.py` doesn't import `train`. Alternatively, move the function into a small `flowers/metrics.py`. Update the monkeypatch target in `test_train.py` to match.
- **Decision:** Accepted — carried into the birds revamp (features/birds-revamp/spec.md), which rewrites this code for CUB-200 and fixes this there

#### S3 — Paths that don't need a GPU are untested

- **File:** `training/flowers/evaluate.py:269-276`, `training/flowers/data.py:119-135`
- **Issue:** Three things are untested:
  - The missing-checkpoint `FileNotFoundError` in `_load_checkpoint`.
  - The eval transform actually being each weights' own `transforms()`. This matters for the next spec: a wrong resize or crop would silently skew both accuracy and confidence.
  - The train-before-model order from S1.

  The spec asks for tests on everything that doesn't need a GPU or the dataset.
- **Fix:**
  - Add a `test_evaluate.py` test that points `WEIGHTS_DIR` at `tmp_path` and asserts the error message names `python -m flowers.train --model small`.
  - Add a `test_data.py` test that `eval_transform("small")` / `("large")` have crop size 224 and resize size 232, matching `MODEL_WEIGHTS[name].transforms()`.
  - If S1 is accepted, a `train.main` test with `Flowers102` patched to raise, and `build_model` patched to fail if it is called.
- **Decision:** Accepted — carried into the birds revamp (features/birds-revamp/spec.md), which rewrites this code for CUB-200 and fixes this there

#### S4 — Multi-line comments and docstrings break the comment guideline

- **File:** `training/flowers/config.py:1-8`, `training/flowers/config.py:47-49`, `training/flowers/evaluate.py:207-212`, `training/flowers/evaluate.py:232-240`, `training/flowers/evaluate.py:277-279`, `training/flowers/train.py:458-463`, `training/flowers/data.py:82-84`
- **Issue:** `.claude/coding-guidelines.md` ("Comments") says "One line, under ~100 chars. Two lines only when truly needed", with no multi-line blocks. Several comments and docstrings run 3–9 lines, and some restate the code. For example, the `config.py` docstring describes what the module plainly holds.
- **Fix:** Cut each one to a single "why" line. Examples:
  - `evaluate.py:232`: `"""Mean batch-1 forward-pass ms over `dataset`; warm-up cycles items, CUDA synced."""`
  - `evaluate.py:277-279`: `# Weights are overwritten by the checkpoint, so skip the ImageNet download.`
- **Decision:** Accepted — carried into the birds revamp (features/birds-revamp/spec.md), which rewrites this code for CUB-200 and fixes this there

## Nitpicks

#### N1 — Hand-rolled patching instead of `monkeypatch`

- **File:** `training/tests/flowers/test_train.py:866-880`
- **Issue:** The test swaps `train_module._evaluate_accuracy` by hand, with `try/finally`. The other flower test files use `pytest.MonkeyPatch`, so this one is inconsistent.
- **Fix:** Take a `monkeypatch` fixture and call `monkeypatch.setattr(train_module, "_evaluate_accuracy", fake_evaluate_accuracy)`.
- **Decision:** Accepted — carried into the birds revamp (features/birds-revamp/spec.md), which rewrites this code for CUB-200 and fixes this there

#### N2 — `torch.load` without `weights_only`

- **File:** `training/flowers/evaluate.py:281`, `training/tests/flowers/test_train.py:885`
- **Issue:** `pyproject.toml` allows `torch>=2.4`. On 2.4 and 2.5, `torch.load` defaults to `weights_only=False` and prints a FutureWarning. The installed 2.14 defaults to `True`, so behaviour depends on the version.
- **Fix:** Pass `weights_only=True` explicitly.
- **Decision:** Declined — The installed torch already defaults to `weights_only=True`, and the checkpoints are self-produced.

## Tests

**Added (15 tests, all pass):**

- `test_data.py` (5): missing-data error, split name per loader, loader dispatch.
- `test_models.py` (4): 102-way output for both architectures (random init), GPU check both ways.
- `test_train.py` (3): a checkpoint is written; the best epoch is kept, not the last (the saved state is compared tensor by tensor with the epoch-2 snapshot); `fit` has no test parameter.
- `test_evaluate.py` (3): accuracy matches a hand-computed case; latency is positive and finite, including when warm-up is longer than the dataset.

The checkpoint-selection test is the most important one for research correctness, and it is solid.

**Gaps:** see S3 (checkpoint-missing error, eval-transform wiring, `train.main` order). `download.main` is untested; that is reasonable, since it needs the network.

**Fragility:**

- The `test_models.py` tests build full ConvNeXt-Base on the CPU. They are slow-ish but deterministic.
- The latency tests only assert that the value is positive and finite, which is appropriate.

## Observations for the routing-viability spec (not defects of this spec)

1. **The accuracy gap is small.** Small is 92.21% and large is 94.67%, a 2.46 pp difference, or about 151 of 6,149 photos.
   - The small model is wrong on about 479 photos and the large model on about 328.
   - Routing can only gain accuracy on photos where the small model is wrong and the large one is right. How many there are depends on how much the two models' errors overlap, which this spec didn't measure (correctly).
   - If the errors overlap heavily, the accuracy an ideal router could reach is only a few points above the small model alone. Measure the disagreement matrix first, before building a router.
   - With 6,149 test photos, the 2.46 pp gap is well outside sampling noise (roughly ±0.3–0.35 pp per model), so it is real, just small.
2. **The latency numbers are not a local-vs-server comparison.** The two measurements run on different hardware:
   - The small model's 5.56 ms is on a 6-core desktop Ryzen 5 9600X, which is far faster than a phone CPU.
   - The large model's 6.63 ms is on an RTX 5070 at fp32, with no network or serialisation cost.

   Their near-equality is an artefact of that setup, not evidence that the phone path is as fast as the server path. The router's latency model needs:
   - a real or scaled phone latency for the small model (for example, a measured TFLite or Core ML number, or a documented slowdown factor)
   - network cost added to the server path

   The CPU thread count (`torch.get_num_threads()`, 6 here) is not pinned, so the CPU number also depends on the machine.
3. **Validation accuracy had saturated at the end of both runs.**
   - Large: epochs 13–15 are all 96.47%. The first one is kept, because the check is strict `>`.
   - Small: epochs 23–30 all fall between 94.31% and 94.41%.

   With 1,020 validation photos (one photo is about 0.1 pp), the choice of epoch among these is within noise. Both test scores sit about 2 pp below validation, which is the expected optimism from choosing on validation. It is not a sign of leakage.
4. **Confidence calibration was not examined** (correctly, since it is out of scope). Both models were trained with label smoothing 0.1, which lowers and compresses softmax confidence. The next spec's "does small-model confidence predict correctness" analysis should expect that, and should not read raw softmax maxima as calibrated probabilities.

## Recommended Decisions

- **S1** — Accept — Swapping two lines makes AC1 hold literally on a fresh machine, with no downside.
- **S2** — Accept — One accuracy function keeps validation and test measured the same way; a small change.
- **S3** — Accept — Cheap tests, and the eval-transform check guards a silent-skew risk that carries into the next spec.
- **S4** — Accept — A direct guideline violation; trimming comments is risk-free.
- **N1** — Accept — A trivial consistency fix, and `monkeypatch` restores the original more reliably.
- **N2** — Decline — The installed torch already defaults to `weights_only=True`, and the checkpoints are self-produced; this is only worth doing while touching those lines for S2 or S3.
