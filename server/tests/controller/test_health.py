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
