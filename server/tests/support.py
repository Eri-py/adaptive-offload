"""Seeding helpers shared by the frame-endpoint tests."""

from dataclasses import dataclass
from pathlib import Path

from common.models import Label, ModelInference, SceneComplexity
from sqlalchemy.orm import Session

DATASET = "coco_val2017"


@dataclass(frozen=True)
class FrameFixture:
    """One seeded frame; eligible unless its local timing or image file is absent."""

    file_name: str
    scene_complexity: float
    local_latency_ms: float | None = 12.0
    offload_latency_ms: float | None = None
    on_disk: bool = True


def jpeg_bytes(file_name: str) -> bytes:
    """Deterministic per-name stand-in image bytes, behind a real JPEG magic number."""
    seed = sum(file_name.encode())
    return b"\xff\xd8\xff\xe0" + bytes((seed + index * 7) % 256 for index in range(64))


def frame_series(count: int, *, prefix: str = "frame") -> list[FrameFixture]:
    """`count` eligible frames with strictly increasing complexity and latency."""
    return [
        FrameFixture(
            file_name=f"{prefix}_{index:03d}.jpg",
            scene_complexity=round(0.01 * (index + 1), 4),
            local_latency_ms=10.0 + index,
        )
        for index in range(count)
    ]


def seed_frames(session: Session, images_dir: Path, frames: list[FrameFixture]) -> None:
    """Write each frame's rows to Postgres and its image file to `images_dir`."""
    for frame in frames:
        session.add(
            SceneComplexity(
                dataset=DATASET,
                file_name=frame.file_name,
                scene_complexity=frame.scene_complexity,
            )
        )
        if frame.local_latency_ms is not None:
            session.add(
                ModelInference(
                    dataset=DATASET,
                    file_name=frame.file_name,
                    model_path=Label.LOCAL,
                    latency_ms=frame.local_latency_ms,
                    accuracy=0.8,
                )
            )
        if frame.offload_latency_ms is not None:
            session.add(
                ModelInference(
                    dataset=DATASET,
                    file_name=frame.file_name,
                    model_path=Label.OFFLOAD,
                    latency_ms=frame.offload_latency_ms,
                    accuracy=0.9,
                )
            )
        if frame.on_disk:
            (images_dir / frame.file_name).write_bytes(jpeg_bytes(frame.file_name))
    session.commit()
