"""End-to-end classify tests: real JPEG in, response out, one logged row.

The writer is bound to the ephemeral Postgres engine via a dependency override, so
production still resolves `DATABASE_URL` lazily and no test-only setting exists.
"""

import io
from functools import partial

import pytest
from common.models import ClassifyRequest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from api.dependencies.providers import get_request_logger
from api.services.classification.request_log import log_classify_request


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), (90, 140, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _rows(engine: Engine) -> list[ClassifyRequest]:
    with Session(engine) as session:
        return list(session.scalars(select(ClassifyRequest)))


def test_classify_logs_exactly_one_row(
    app: FastAPI, postgres_engine: Engine, jpeg_bytes: bytes
) -> None:
    app.dependency_overrides[get_request_logger] = lambda: partial(
        log_classify_request, engine=postgres_engine
    )
    with TestClient(app) as client:
        response = client.post("/classify", content=jpeg_bytes)

    assert response.status_code == 200
    body = response.json()
    assert body["species"] == "Species 7"
    assert body["class_index"] == 7
    assert len(body["top3"]) == 3

    rows = _rows(postgres_engine)
    assert len(rows) == 1
    row = rows[0]
    assert row.payload_bytes == len(jpeg_bytes)
    assert row.predicted_label == 7
    assert row.compute_ms == pytest.approx(body["compute_ms"])
    assert row.compute_ms < row.total_ms


def test_classify_succeeds_when_database_unreachable(
    app: FastAPI, postgres_engine: Engine, jpeg_bytes: bytes
) -> None:
    dead_engine = create_engine(
        "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    app.dependency_overrides[get_request_logger] = lambda: partial(
        log_classify_request, engine=dead_engine
    )
    with TestClient(app) as client:
        response = client.post("/classify", content=jpeg_bytes)

    assert response.status_code == 200
    assert response.json()["species"] == "Species 7"
    assert _rows(postgres_engine) == []
