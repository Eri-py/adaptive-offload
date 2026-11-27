"""Tests for POST /classify against the stubbed app."""

import io

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.dependencies.providers import get_loaded_model

JPEG_HEADERS = {"Content-Type": "image/jpeg"}


def _jpeg(size: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (120, 80, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_classify_returns_full_payload(client: TestClient) -> None:
    response = client.post("/classify", content=_jpeg(224), headers=JPEG_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["species"] == "Species 7"
    assert payload["class_index"] == 7
    assert 0.0 < payload["top2_margin"] <= payload["max_probability"] <= 1.0
    assert [item["index"] for item in payload["top3"]] == [7, 3, 11]
    assert payload["top3"][0]["name"] == "Species 7"
    assert payload["compute_ms"] >= 0.0
    assert payload["geometric_transform_applied"] is False


def test_non_224_image_reports_transform(client: TestClient) -> None:
    response = client.post("/classify", content=_jpeg(300), headers=JPEG_HEADERS)

    assert response.status_code == 200
    assert response.json()["geometric_transform_applied"] is True


def test_non_image_body_returns_400(client: TestClient) -> None:
    response = client.post("/classify", content=b"not an image", headers=JPEG_HEADERS)

    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_same_image_twice_is_deterministic(client: TestClient) -> None:
    image = _jpeg(224)
    first = client.post("/classify", content=image, headers=JPEG_HEADERS).json()
    second = client.post("/classify", content=image, headers=JPEG_HEADERS).json()

    for key in ("class_index", "max_probability", "top2_margin"):
        assert first[key] == second[key]


def test_missing_model_returns_503(app: FastAPI) -> None:
    app.dependency_overrides[get_loaded_model] = lambda: None
    with TestClient(app) as client:
        response = client.post("/classify", content=_jpeg(224), headers=JPEG_HEADERS)

    assert response.status_code == 503
