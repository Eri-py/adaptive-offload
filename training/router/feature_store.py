"""Read/write access to the `frame_features` table, via `common.models`.

Mirrors `datagen.persistence`'s style: every function takes a SQLAlchemy
`Engine` explicitly (never a module-level connection) so tests can substitute
the ephemeral-Postgres fixture engine and `router.extract_features` can pass
whatever engine `common.db.get_engine()` builds.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.models import FrameFeatures, SceneComplexity
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from router.frame_features.confidence import ConfidenceFeatures
from router.frame_features.image_stats import ImageStats


@dataclass(frozen=True)
class FeatureRow:
    """One frame's combined image-statistics and confidence features, ready to store."""

    file_name: str
    image_stats: ImageStats
    confidence: ConfidenceFeatures


def list_dataset_frames(engine: Engine, dataset: str) -> list[str]:
    """List `dataset`'s frame file names from `scene_complexity`, ordered by `file_name`.

    Mirrors `datagen.persistence.get_known_complexity`'s ordering rationale —
    deterministic output regardless of Postgres's unordered `SELECT` row
    order.
    """
    with Session(engine) as session:
        return list(
            session.execute(
                select(SceneComplexity.file_name)
                .where(SceneComplexity.dataset == dataset)
                .order_by(SceneComplexity.file_name)
            ).scalars()
        )


def get_known_feature_file_names(engine: Engine, dataset: str) -> set[str]:
    """Return the file names already stored in `frame_features` for `dataset`."""
    with Session(engine) as session:
        return set(
            session.execute(
                select(FrameFeatures.file_name).where(FrameFeatures.dataset == dataset)
            ).scalars()
        )


def store_features(engine: Engine, dataset: str, rows: list[FeatureRow]) -> None:
    """Insert new `frame_features` rows, skipping file names already present.

    Mirrors `datagen.persistence.store_complexity_scores`'s already-known-check
    shape.
    """
    with Session(engine) as session:
        known_file_names = set(
            session.execute(
                select(FrameFeatures.file_name).where(FrameFeatures.dataset == dataset)
            ).scalars()
        )
        new_rows = [
            FrameFeatures(
                dataset=dataset,
                file_name=row.file_name,
                sharpness=row.image_stats.sharpness,
                brightness=row.image_stats.brightness,
                contrast=row.image_stats.contrast,
                colorfulness=row.image_stats.colorfulness,
                entropy=row.image_stats.entropy,
                detection_count=row.confidence.detection_count,
                max_confidence=row.confidence.max_confidence,
                mean_confidence=row.confidence.mean_confidence,
                min_confidence=row.confidence.min_confidence,
                mean_box_area=row.confidence.mean_box_area,
                min_box_area=row.confidence.min_box_area,
            )
            for row in rows
            if row.file_name not in known_file_names
        ]
        session.add_all(new_rows)
        session.commit()
