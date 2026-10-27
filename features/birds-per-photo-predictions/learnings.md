# Learnings — birds-per-photo-predictions

- Task 1: `python`/`ruff`/`mypy` are not on PATH; use the repo-root `.venv/bin` (`export PATH=/home/eriol/projects/adaptive-offload/.venv/bin:$PATH`). `ruff format --check` flags two pre-existing files (`common/models.py` SimulationRun line, `migrations/env.py`) that predate this task; `ruff check` and `mypy` are clean.
