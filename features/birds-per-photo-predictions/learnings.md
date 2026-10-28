# Learnings — birds-per-photo-predictions

- Task 1: `python`/`ruff`/`mypy` are not on PATH; use the repo-root `.venv/bin` (`export PATH=/home/eriol/projects/adaptive-offload/.venv/bin:$PATH`). `ruff format --check` flags two pre-existing files (`common/models.py` SimulationRun line, `migrations/env.py`) that predate this task; `ruff check` and `mypy` are clean.
- Task 2: `split_ids()` returns ids only, so `predict_photos` takes `expected_labels` from the caller; there is no public accessor for a split's labels (CubDataset._records is private), so Task 3 must supply them (e.g. add one to splits.py) or the alignment check has nothing to compare against.
- Task 3: BirdPrediction's PK excludes split, so write_predictions deletes by (dataset, split, model_name) while reads filter the same triple; photo-id ordering is done in pandas (int key) since ids are stringified ints and SQL string order would put "10" before "2".
- Task 4: `store.py` imports `PhotoPrediction` from `predictions.py`, and the plan puts `main()` in `predictions.py`, so `record_split` imports `write_predictions` locally to avoid a cycle (moving `PhotoPrediction` would fix it but touches files outside the task). `main()` reuses the private `_load_checkpoint`/`_MODEL_NAMES` from `evaluate.py`, which is outside the Files list. Dataset check is `split_labels()` called first; the new public accessor is `splits.split_labels()`.
- Task 4 (orchestrator note): the subagent's hand-back report came back as the
  literal string "placeholder", so it was verified from the tree and the diffs
  instead. Gate was green (236 tests, mypy 9, ruff clean) and all four success
  criteria are covered by tests, so the task was accepted on that evidence.
- Task 4 (orchestrator note): two design smells left for the reviewer, neither
  a correctness problem. (1) `predictions.py` imports the private
  `_MODEL_NAMES` and `_load_checkpoint` from `evaluation/evaluate.py` — the
  same reach-into-privates pattern that `split_labels()` was added to avoid,
  and it couples the new module to one the plan froze. (2) `predictions.py`
  and `store.py` now import each other, worked around with a function-local
  import in `record_split`; this is a consequence of the plan putting `main()`
  in `predictions.py` rather than in its own entry-point module. Also minor:
  `split_labels()` re-reads the index files once per (model, split), so seven
  reads per full run.
- Task 5: `summary.py` splits pure computation (`summarize(small, large)` over two DataFrames) from DB loading, so the hand-computed tests need no Postgres. Escalated count is integer `n * percent // 100` (floor), AUC is NaN when a model has only one outcome (skipped from MLflow metrics), and gap-recovered is NaN on a zero gap and unclamped above 1.

- Task 6 defect: `main()` in `birds/evaluation/predictions.py` and `summary.py` called `get_engine()` without `load_dotenv`, so `python -m ...` failed with "DATABASE_URL is not set". Unit tests monkeypatch `get_engine`, so they never exercised the real resolution. Fix: `ENV_PATH` constant in `predictions.py` (3 `.parent` hops from `birds/evaluation/`, resolves to `training/.env`), `load_dotenv(ENV_PATH)` at the top of both mains; `summary.py` imports the constant. A test pins `ENV_PATH == training/.env`. Lesson: when tests stub an environment-dependent seam, pin the resolved path/config separately.

- Task 7: `findings/birds.md` now separates unreproducible numbers (flowers, and the pilot's birds checkpoints 0.7727/0.8676) from reproducible ones (committed checkpoints, 0.7798/0.8699). The flowers-vs-birds comparison was made on the pilot's numbers and stays labelled as such; this feature does not re-settle it, since flowers cannot be re-run. Also unsettled: the file's "same recipe unchanged" training settings describe the pilot run, and nothing here verifies the committed checkpoints used identical settings.

- B1 fix: `main()` loaded checkpoints inside the model loop, so a missing `large.pt` failed after `small`'s rows were committed. The test only had both missing. Now `evaluate.require_checkpoint(name)` (path check, shared with `_load_checkpoint` so the message is identical) runs for every model before any write. Asymmetric test: `small.pt` present, `large.pt` absent, and `_load_checkpoint` patched to fail, so only the up-front check can raise.
- S1: summary.py errors now name the recording command via _RECORD_COMMAND; tests match the command string (empty frames hit the 'no stored predictions' branch).
- S2: ENV_PATH now lives in birds/config.py (TRAINING_DIR / .env); both main()s import it. Hop-count test moved to tests/birds/test_config.py.
