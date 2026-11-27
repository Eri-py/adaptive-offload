import logging

import pytest
from common.models import ClassifyRequest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from api.services.classification import request_log
from api.services.classification.classify import Classification, Prediction
from api.services.classification.request_log import log_classify_request

_CLASSIFICATION = Classification(
    top1=Prediction(index=7, name="Species 7", probability=0.75),
    max_probability=0.75,
    margin=0.5,
    top3=[
        Prediction(index=7, name="Species 7", probability=0.75),
        Prediction(index=3, name="Species 3", probability=0.25),
        Prediction(index=11, name="Species 11", probability=0.0),
    ],
    compute_ms=12.5,
)


def test_writes_one_row_with_expected_fields(postgres_engine: Engine) -> None:
    log_classify_request(
        _CLASSIFICATION,
        payload_bytes=2048,
        was_transformed=True,
        total_ms=30.25,
        engine=postgres_engine,
    )

    with Session(postgres_engine) as session:
        rows = session.scalars(select(ClassifyRequest)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.payload_bytes == 2048
    assert row.was_transformed is True
    assert row.predicted_label == 7
    assert row.predicted_name == "Species 7"
    assert row.max_probability == 0.75
    assert row.top2_margin == 0.5
    assert row.compute_ms == 12.5
    assert row.total_ms == 30.25
    assert row.requested_at is not None
    assert row.id


def test_raising_session_is_swallowed_and_warns(
    postgres_engine: Engine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _ExplodingSession:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("database exploded")

    monkeypatch.setattr(request_log, "Session", _ExplodingSession)

    with caplog.at_level(logging.WARNING, logger=request_log.__name__):
        log_classify_request(
            _CLASSIFICATION,
            payload_bytes=1,
            was_transformed=False,
            total_ms=1.0,
            engine=postgres_engine,
        )

    assert any(
        record.levelno == logging.WARNING and "database exploded" in record.getMessage()
        for record in caplog.records
    )


def test_missing_table_is_swallowed_and_warns(
    postgres_engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    ClassifyRequest.__table__.drop(postgres_engine)  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING, logger=request_log.__name__):
        log_classify_request(
            _CLASSIFICATION,
            payload_bytes=1,
            was_transformed=False,
            total_ms=1.0,
            engine=postgres_engine,
        )

    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_unset_database_url_is_swallowed_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(request_log, "_engine", None)

    with caplog.at_level(logging.WARNING, logger=request_log.__name__):
        log_classify_request(_CLASSIFICATION, payload_bytes=1, was_transformed=False, total_ms=1.0)

    assert any("DATABASE_URL" in record.getMessage() for record in caplog.records)
