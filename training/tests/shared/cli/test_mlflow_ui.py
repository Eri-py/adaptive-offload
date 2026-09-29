import os
import sys
from typing import Any

import pytest

from shared.cli import mlflow_ui

_URI = "postgresql+psycopg://u:p@localhost:5432/mlflow"


def _patch(monkeypatch: pytest.MonkeyPatch, *, which: str | None = "/venv/bin/mlflow") -> list[Any]:
    execs: list[Any] = []
    monkeypatch.setattr(mlflow_ui, "tracking_uri", lambda: _URI)
    monkeypatch.setattr(mlflow_ui, "_mlflow_executable", lambda: which)
    monkeypatch.setattr(os, "execv", lambda path, args: execs.append((path, args)))
    return execs


def test_passes_the_store_uri_and_extra_arguments_through(monkeypatch: pytest.MonkeyPatch) -> None:
    execs = _patch(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["mlflow-ui", "--port", "5001"])

    mlflow_ui.main()

    assert execs == [
        (
            "/venv/bin/mlflow",
            ["mlflow", "ui", "--backend-store-uri", _URI, "--port", "5001"],
        )
    ]


def test_unset_tracking_uri_exits_with_the_resolver_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)

    def boom() -> str:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set. Add it to training/.env")

    monkeypatch.setattr(mlflow_ui, "tracking_uri", boom)

    with pytest.raises(SystemExit) as excinfo:
        mlflow_ui.main()

    assert "MLFLOW_TRACKING_URI is not set" in str(excinfo.value)


def test_missing_mlflow_executable_exits_rather_than_failing_in_execv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execs = _patch(monkeypatch, which=None)

    with pytest.raises(SystemExit) as excinfo:
        mlflow_ui.main()

    assert "was not found" in str(excinfo.value)
    assert execs == []
