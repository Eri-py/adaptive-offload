"""MLflow experiment tracking for the birds pipeline; the only module that calls MLflow.

Tracking never breaks the caller: set BIRDS_TRACKING=off to skip it entirely, and any
MLflow failure during run setup or teardown is printed as a warning and ignored.
"""

import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlflow

from birds import config

MLRUNS_DIR = config.MLRUNS_DIR
EXPERIMENT_NAME = config.EXPERIMENT_NAME


@dataclass(frozen=True)
class RunHandle:
    """Handle to the active run; id is None when tracking is off or failed to start."""

    id: str | None


NOOP_HANDLE = RunHandle(id=None)


def _tracking_off() -> bool:
    # Read per call so tests can flip it with monkeypatch.setenv.
    return os.environ.get("BIRDS_TRACKING", "").strip().lower() == "off"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=config.MLRUNS_DIR.parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _warn(what: str, exc: Exception) -> None:
    print(f"WARNING: MLflow tracking {what} failed ({type(exc).__name__}: {exc}); continuing.")


@contextmanager
def run(
    run_name: str, params: Mapping[str, Any], *, run_id: str | None = None
) -> Iterator[RunHandle]:
    """Open an MLflow run (resuming run_id if given), log params, yield a handle."""
    if _tracking_off():
        yield NOOP_HANDLE
        return

    handle = NOOP_HANDLE
    active = False
    try:
        # MLflow 3.x refuses the plain-directory file store unless explicitly allowed.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(EXPERIMENT_NAME)
        active_run = mlflow.start_run(run_id=run_id, run_name=run_name)
        active = True
        handle = RunHandle(id=active_run.info.run_id)
        mlflow.log_params(dict(params))
        commit = _git_commit()
        if commit is not None:
            mlflow.set_tag("git_commit", commit)
    except Exception as exc:  # noqa: BLE001 - tracking must never break the caller
        _warn("setup", exc)
        if not active:
            handle = NOOP_HANDLE

    status = "FINISHED"
    try:
        yield handle
    except BaseException:
        status = "FAILED"
        raise
    finally:
        if active:
            try:
                mlflow.end_run(status=status)
            except Exception as exc:  # noqa: BLE001
                _warn("teardown", exc)


def log_metrics(
    handle: RunHandle, metrics: Mapping[str, float], *, step: int | None = None
) -> None:
    """Log metrics to the handle's run; a no-op for the no-op handle. Never raises."""
    if handle.id is None:
        return
    try:
        mlflow.log_metrics(dict(metrics), step=step)
    except Exception as exc:  # noqa: BLE001
        _warn("metric logging", exc)
