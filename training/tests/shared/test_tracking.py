import os
from collections.abc import Iterator
from typing import Any

import mlflow
import pytest
from common.testing import ephemeral_postgres_database
from mlflow.tracking import MlflowClient

from shared import tracking

_EXPERIMENT = "test-experiment"


@pytest.fixture(scope="module")
def tracking_db() -> Iterator[str]:
    """A disposable Postgres database standing in for the real `mlflow` one.

    Module-scoped because MLflow runs its own Alembic migrations the first time it
    touches an empty database — once per module rather than once per test.
    `training/tests/conftest.py` has already loaded `training/.env`.
    """
    admin_url = os.environ.get("POSTGRES_ADMIN_URL")
    if not admin_url:
        raise RuntimeError(
            "POSTGRES_ADMIN_URL is not set. Add it to training/.env "
            "(a Postgres admin connection string pointed at the "
            "'postgres' maintenance database)."
        )
    with ephemeral_postgres_database(admin_url) as engine:
        yield engine.url.render_as_string(hide_password=False)


def _point_store_at(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", url)


def _run_count(url: str) -> int:
    client = MlflowClient(tracking_uri=url)
    experiment = client.get_experiment_by_name(_EXPERIMENT)
    if experiment is None:
        return 0
    return len(client.search_runs([experiment.experiment_id]))


def test_off_switch_yields_none_id_writes_nothing_and_runs_body(
    monkeypatch: pytest.MonkeyPatch, tracking_db: str
) -> None:
    monkeypatch.setenv("TRAINING_TRACKING", "off")
    _point_store_at(monkeypatch, tracking_db)
    before = _run_count(tracking_db)
    ran = False

    with tracking.run("r", {"lr": 0.1}, experiment=_EXPERIMENT) as handle:
        tracking.log_metrics(handle, {"acc": 1.0})
        ran = True

    assert handle.id is None
    assert ran
    assert _run_count(tracking_db) == before


def test_missing_tracking_uri_warns_and_still_runs_body(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TRAINING_TRACKING", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    # Point .env loading at a directory with no .env, so nothing restores the variable.
    monkeypatch.setattr(tracking, "TRAINING_DIR", tracking.TRAINING_DIR / "nope")
    ran = False

    with tracking.run("r", {"lr": 0.1}, experiment=_EXPERIMENT) as handle:
        ran = True

    assert ran
    assert handle.id is None
    assert "MLFLOW_TRACKING_URI is not set" in capsys.readouterr().out


def test_run_records_params_and_metrics(monkeypatch: pytest.MonkeyPatch, tracking_db: str) -> None:
    monkeypatch.delenv("TRAINING_TRACKING", raising=False)
    _point_store_at(monkeypatch, tracking_db)

    with tracking.run("my-run", {"lr": 0.1, "model": "small"}, experiment=_EXPERIMENT) as handle:
        tracking.log_metrics(handle, {"acc": 0.5}, step=1)
        tracking.log_metrics(handle, {"acc": 0.75}, step=2)

    assert handle.id is not None
    client = MlflowClient(tracking_uri=tracking_db)
    data = client.get_run(handle.id).data
    assert data.params == {"lr": "0.1", "model": "small"}
    assert data.metrics["acc"] == 0.75
    assert [m.value for m in client.get_metric_history(handle.id, "acc")] == [0.5, 0.75]
    assert client.get_run(handle.id).info.status == "FINISHED"


def test_run_can_resume_an_existing_run_id(
    monkeypatch: pytest.MonkeyPatch, tracking_db: str
) -> None:
    monkeypatch.delenv("TRAINING_TRACKING", raising=False)
    _point_store_at(monkeypatch, tracking_db)

    with tracking.run("first", {"a": 1}, experiment=_EXPERIMENT) as first:
        pass
    with tracking.run("again", {}, experiment=_EXPERIMENT, run_id=first.id) as second:
        tracking.log_metrics(second, {"m": 2.0})

    assert second.id == first.id
    client = MlflowClient(tracking_uri=tracking_db)
    assert client.get_run(str(first.id)).data.metrics["m"] == 2.0


def test_mlflow_failure_on_start_warns_and_still_runs_body(
    monkeypatch: pytest.MonkeyPatch, tracking_db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TRAINING_TRACKING", raising=False)
    _point_store_at(monkeypatch, tracking_db)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("store unreachable")

    monkeypatch.setattr("shared.tracking.mlflow.start_run", boom)
    ran = False

    with tracking.run("r", {"lr": 0.1}, experiment=_EXPERIMENT) as handle:
        tracking.log_metrics(handle, {"acc": 1.0})
        ran = True

    assert ran
    assert handle.id is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "store unreachable" in out


def test_body_exception_propagates(monkeypatch: pytest.MonkeyPatch, tracking_db: str) -> None:
    monkeypatch.delenv("TRAINING_TRACKING", raising=False)
    _point_store_at(monkeypatch, tracking_db)

    with (
        pytest.raises(ValueError, match="from body"),
        tracking.run("r", {}, experiment=_EXPERIMENT),
    ):
        raise ValueError("from body")

    assert mlflow.active_run() is None
