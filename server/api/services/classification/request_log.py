"""Best-effort logging of each classify request to the `classify_requests` table.

Logging must never break a response: any failure (unreachable database, missing
table, bad or unset URL) is logged as a warning and swallowed.
"""

import logging

from common.db import get_engine
from common.models import ClassifyRequest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from api.services.classification.classify import Classification

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def _lazy_engine() -> Engine:
    """Builds the engine on first use; a failed build is not cached, so a later call retries."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def log_classify_request(
    classification: Classification,
    *,
    payload_bytes: int,
    was_transformed: bool,
    total_ms: float,
    engine: Engine | None = None,
) -> None:
    """Inserts one `classify_requests` row; never raises.

    `total_ms` is the whole-request time; `classification.compute_ms` is the forward pass only.
    `engine` overrides the lazily built `DATABASE_URL` engine (used by tests).
    """
    try:
        row = ClassifyRequest(
            payload_bytes=payload_bytes,
            was_transformed=was_transformed,
            predicted_label=classification.top1.index,
            predicted_name=classification.top1.name,
            max_probability=classification.max_probability,
            top2_margin=classification.margin,
            compute_ms=classification.compute_ms,
            total_ms=total_ms,
        )
        with Session(engine or _lazy_engine()) as session:
            session.add(row)
            session.commit()
    except Exception as error:  # noqa: BLE001 - best-effort by design: never fail the response
        logger.warning("Failed to log classify request: %s", error)
