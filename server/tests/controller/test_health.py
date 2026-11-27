import pytest
from fastapi.testclient import TestClient

from api.core.app import create_app


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_model_not_loaded_without_state() -> None:
    with TestClient(create_app(load_resources=False)) as bare_client:
        body = bare_client.get("/health").json()

    assert body["model_loaded"] is False


def test_startup_fails_clearly_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        with TestClient(create_app()):
            pass
