"""Choosing the benchmark frame set, and resolving a frame's image on disk.

A frame is eligible only if Postgres has both its scene-complexity score and
its desktop YOLOv8n (local-path) timing, and its image is actually on disk —
the phone has to decode the same JPEG the desktop measurement did, and a
missing one would otherwise produce a frame the app can't fetch.
"""

from pathlib import Path

from common.models import Label, ModelInference, SceneComplexity
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from api.contracts.frames import BenchmarkFrame


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    """`count` indices spanning 0..total-1 inclusive, so both ends are included.

    Distinct and ascending whenever `count <= total`, which the caller ensures.
    """
    if count <= 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def eligible_frames(
    session: Session, *, dataset: str, images_dir: Path
) -> list[BenchmarkFrame]:
    """Every eligible frame of `dataset`, ordered least to most complex.

    Ordering breaks complexity ties on file name so the same database contents
    always produce the same order, keeping separate benchmark runs comparable.
    """
    statement = (
        select(
            SceneComplexity.file_name,
            SceneComplexity.scene_complexity,
            ModelInference.latency_ms,
        )
        .join(
            ModelInference,
            and_(
                ModelInference.dataset == SceneComplexity.dataset,
                ModelInference.file_name == SceneComplexity.file_name,
                ModelInference.model_path == Label.LOCAL,
            ),
        )
        .where(SceneComplexity.dataset == dataset)
        .order_by(SceneComplexity.scene_complexity, SceneComplexity.file_name)
    )
    return [
        BenchmarkFrame(
            frame_id=file_name,
            scene_complexity=scene_complexity,
            desktop_local_latency_ms=latency_ms,
        )
        for file_name, scene_complexity, latency_ms in session.execute(statement).all()
        if (images_dir / file_name).is_file()
    ]


def select_frames(
    session: Session, *, dataset: str, images_dir: Path, count: int
) -> list[BenchmarkFrame]:
    """`count` eligible frames spread evenly across the complexity range.

    Returns every eligible frame when fewer than `count` exist.
    """
    frames = eligible_frames(session, dataset=dataset, images_dir=images_dir)
    if count >= len(frames):
        return frames
    return [frames[index] for index in _evenly_spaced_indices(len(frames), count)]


def resolve_frame_image(
    session: Session, *, dataset: str, images_dir: Path, frame_id: str
) -> Path | None:
    """The on-disk image path for a frame Postgres knows about, else `None`.

    The database lookup is what authorizes the read, so an id naming anything
    else — including a path reaching outside `images_dir` — finds nothing. The
    containment check below is a second guard, not the primary one.
    """
    if session.get(SceneComplexity, (dataset, frame_id)) is None:
        return None
    try:
        resolved = (images_dir / frame_id).resolve(strict=True)
    except OSError:
        return None
    if resolved.parent != images_dir.resolve():
        return None
    return resolved
