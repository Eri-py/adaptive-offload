"""MLflow experiment tracking for every training attempt under `training/`.

The one module that calls MLflow, so the backend can move without touching callers.
Tracking never breaks its caller: set `TRAINING_TRACKING=off` to skip it entirely, and
any MLflow failure is printed as a warning and ignored.

Run history lives in its own Postgres database (`MLFLOW_TRACKING_URI` in `training/.env`),
deliberately separate from `adaptive_offload` — MLflow keeps its own Alembic version in the
default `alembic_version` table, so one database would have the two schemas clobber
each other.
"""

import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

TRAINING_DIR = Path(__file__).resolve().parents[1]
# Artifacts are never logged (checkpoints are hundreds of MB and already on disk); this
# only keeps MLflow from defaulting its artifact root into the working directory.
ARTIFACTS_DIR = TRAINING_DIR / "mlartifacts"


@dataclass(frozen=True)
class RunHandle:
    """Handle to the active run; id is None when tracking is off or failed to start."""

    id: str | None


NOOP_HANDLE = RunHandle(id=None)


def _tracking_off() -> bool:
    # Read per call so tests can flip it with monkeypatch.setenv.
    return os.environ.get("TRAINING_TRACKING", "").strip().lower() == "off"


def _tracking_uri() -> str:
    """Resolve the tracking database URL, mirroring `common.db`'s no-silent-default rule."""
    load_dotenv(TRAINING_DIR / ".env")
    url = os.environ.get("MLFLOW_TRACKING_URI")
    if not url:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Add it to training/.env (the Postgres "
            "connection string for the 'mlflow' database, e.g. "
            "postgresql+psycopg://user:pass@host:5432/mlflow)."
        )
    return url


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=TRAINING_DIR,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _warn(what: str, exc: Exception) -> None:
    print(f"WARNING: MLflow tracking {what} failed ({type(exc).__name__}: {exc}); continuing.")


# --- The run-id sidecar ------------------------------------------------------
# Tracking metadata, not training logic, so both entry points read it from here rather
# than evaluation importing the training module for a filename convention.


def run_sidecar_path(checkpoint_path: Path) -> Path:
    """Where the MLflow run id for `checkpoint_path` is recorded (`<checkpoint>.run.json`)."""
    return checkpoint_path.with_name(checkpoint_path.name + ".run.json")


def record_run_id(checkpoint_path: Path, run_id: str | None) -> None:
    """Write the sidecar for `run_id`; with no id, remove any stale one so it can't mislead.

    Callers write this *before* training, not after: the checkpoint is overwritten as soon
    as an epoch improves, so a sidecar written afterwards leaves an interrupted run's new
    model sitting beside the previous run's id.
    """
    sidecar = run_sidecar_path(checkpoint_path)
    if run_id is None:
        sidecar.unlink(missing_ok=True)
        return
    sidecar.write_text(json.dumps({"run_id": run_id}))


def read_run_id(checkpoint_path: Path) -> str | None:
    """The run id recorded beside `checkpoint_path`; None if absent or unreadable."""
    try:
        run_id = json.loads(run_sidecar_path(checkpoint_path).read_text())["run_id"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return run_id if isinstance(run_id, str) and run_id else None


# --- Runs --------------------------------------------------------------------


@contextmanager
def run(
    run_name: str, params: Mapping[str, Any], *, experiment: str, run_id: str | None = None
) -> Iterator[RunHandle]:
    """Open an MLflow run in `experiment` (resuming run_id if given), log params, yield a handle."""
    if _tracking_off():
        yield NOOP_HANDLE
        return

    handle = NOOP_HANDLE
    active = False
    try:
        mlflow.set_tracking_uri(_tracking_uri())
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=ARTIFACTS_DIR.as_uri())
        mlflow.set_experiment(experiment)
        active_run = mlflow.start_run(run_id=run_id, run_name=run_name)
        active = True
        handle = RunHandle(id=active_run.info.run_id)
    except Exception as exc:  # noqa: BLE001 - tracking must never break the caller
        _warn("setup", exc)
        if not active:
            handle = NOOP_HANDLE

    if active:
        # Separate from setup so a rejected param (MLflow refuses to change one on a
        # resumed run) doesn't get reported as the run having failed to start.
        try:
            mlflow.log_params(dict(params))
            # Only when creating the run: on a resume this would overwrite the commit that
            # produced the model with whatever is checked out now.
            if run_id is None:
                commit = _git_commit()
                if commit is not None:
                    mlflow.set_tag("git_commit", commit)
        except Exception as exc:  # noqa: BLE001
            _warn("parameter logging", exc)

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
                # Leaving the run on the fluent stack would make every later run in this
                # process fail to start and silently record nothing.
                try:
                    mlflow.end_run()
                except Exception:  # noqa: BLE001
                    from mlflow.tracking import fluent

                    fluent._active_run_stack.get().clear()  # type: ignore[no-untyped-call]


def log_metrics(
    handle: RunHandle, metrics: Mapping[str, float], *, step: int | None = None
) -> None:
    """Log metrics to the handle's run; a no-op for the no-op handle. Never raises."""
    if handle.id is None:
        return
    log_to_run(handle.id, metrics=metrics, step=step)


def log_to_run(
    run_id: str,
    *,
    metrics: Mapping[str, float] | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    step: int | None = None,
) -> None:
    """Write to `run_id` through the client, without resuming or re-ending it. Never raises.

    Targeting the run id directly is what makes the handle meaningful — the fluent
    `mlflow.log_metrics` writes to whichever run is globally active instead. It also
    leaves a finished run's `end_time` and status alone, so recorded training durations
    stay the duration of the training.
    """
    try:
        client = MlflowClient(tracking_uri=_tracking_uri())
        for key, value in (params or {}).items():
            client.log_param(run_id, key, value)
        for key, tag_value in (tags or {}).items():
            client.set_tag(run_id, key, tag_value)
        for key, metric_value in (metrics or {}).items():
            client.log_metric(run_id, key, metric_value, step=step or 0)
    except Exception as exc:  # noqa: BLE001
        _warn("metric logging", exc)
