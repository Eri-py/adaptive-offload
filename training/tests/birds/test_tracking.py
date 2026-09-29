from pathlib import Path
from typing import Any

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from birds import tracking


def _point_store_at(monkeypatch: pytest.MonkeyPatch, store: Path) -> None:
    monkeypatch.setattr(tracking, "MLRUNS_DIR", store)


def test_off_switch_yields_none_id_writes_nothing_and_runs_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIRDS_TRACKING", "off")
    store = tmp_path / "mlruns"
    _point_store_at(monkeypatch, store)
    ran = False

    with tracking.run("r", {"lr": 0.1}) as handle:
        tracking.log_metrics(handle, {"acc": 1.0})
        ran = True

    assert handle.id is None
    assert ran
    assert not store.exists()


def test_run_records_params_and_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BIRDS_TRACKING", raising=False)
    store = tmp_path / "mlruns"
    _point_store_at(monkeypatch, store)

    with tracking.run("my-run", {"lr": 0.1, "model": "small"}) as handle:
        tracking.log_metrics(handle, {"acc": 0.5}, step=1)
        tracking.log_metrics(handle, {"acc": 0.75}, step=2)

    assert handle.id is not None
    client = MlflowClient(tracking_uri=store.as_uri())
    data = client.get_run(handle.id).data
    assert data.params == {"lr": "0.1", "model": "small"}
    assert data.metrics["acc"] == 0.75
    assert [m.value for m in client.get_metric_history(handle.id, "acc")] == [0.5, 0.75]
    assert client.get_run(handle.id).info.status == "FINISHED"


def test_run_can_resume_an_existing_run_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BIRDS_TRACKING", raising=False)
    store = tmp_path / "mlruns"
    _point_store_at(monkeypatch, store)

    with tracking.run("first", {"a": 1}) as first:
        pass
    with tracking.run("again", {}, run_id=first.id) as second:
        tracking.log_metrics(second, {"m": 2.0})

    assert second.id == first.id
    client = MlflowClient(tracking_uri=store.as_uri())
    assert client.get_run(str(first.id)).data.metrics["m"] == 2.0


def test_mlflow_failure_on_start_warns_and_still_runs_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("BIRDS_TRACKING", raising=False)
    _point_store_at(monkeypatch, tmp_path / "mlruns")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("store unreachable")

    monkeypatch.setattr("birds.tracking.mlflow.start_run", boom)
    ran = False

    with tracking.run("r", {"lr": 0.1}) as handle:
        tracking.log_metrics(handle, {"acc": 1.0})
        ran = True

    assert ran
    assert handle.id is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "store unreachable" in out


def test_body_exception_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BIRDS_TRACKING", raising=False)
    _point_store_at(monkeypatch, tmp_path / "mlruns")

    with pytest.raises(ValueError, match="from body"), tracking.run("r", {}):
        raise ValueError("from body")

    assert mlflow.active_run() is None
