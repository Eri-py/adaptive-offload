"""Unit tests for frame eligibility, spread and image-path resolution."""

from pathlib import Path

from sqlalchemy.orm import Session

from api.services.benchmark.frames import (
    _evenly_spaced_indices,
    eligible_frames,
    resolve_frame_image,
    select_frames,
)
from tests.support import DATASET, FrameFixture, frame_series, seed_frames


def test_evenly_spaced_indices_include_both_ends() -> None:
    assert _evenly_spaced_indices(20, 5) == [0, 5, 10, 14, 19]
    assert _evenly_spaced_indices(10, 10) == list(range(10))
    assert _evenly_spaced_indices(10, 1) == [0]


def test_eligible_frames_order_by_complexity_then_file_name(
    postgres_session: Session, images_dir: Path
) -> None:
    seed_frames(
        postgres_session,
        images_dir,
        [
            FrameFixture(file_name="b.jpg", scene_complexity=0.5),
            FrameFixture(file_name="a.jpg", scene_complexity=0.5),
            FrameFixture(file_name="c.jpg", scene_complexity=0.1),
        ],
    )

    frames = eligible_frames(postgres_session, dataset=DATASET, images_dir=images_dir)

    assert [frame.frame_id for frame in frames] == ["c.jpg", "a.jpg", "b.jpg"]


def test_select_frames_reports_the_stored_desktop_timing(
    postgres_session: Session, images_dir: Path
) -> None:
    seed_frames(postgres_session, images_dir, frame_series(4))

    frames = select_frames(postgres_session, dataset=DATASET, images_dir=images_dir, count=2)

    assert [frame.frame_id for frame in frames] == ["frame_000.jpg", "frame_003.jpg"]
    assert [frame.desktop_local_latency_ms for frame in frames] == [10.0, 13.0]


def test_resolve_frame_image_rejects_unknown_and_escaping_ids(
    postgres_session: Session, images_dir: Path
) -> None:
    seed_frames(postgres_session, images_dir, frame_series(1))
    (images_dir.parent / "secret.jpg").write_bytes(b"do not serve me")

    resolved = resolve_frame_image(
        postgres_session, dataset=DATASET, images_dir=images_dir, frame_id="frame_000.jpg"
    )
    assert resolved == (images_dir / "frame_000.jpg").resolve()

    for frame_id in ["not_a_frame.jpg", "../secret.jpg", "/etc/passwd"]:
        assert (
            resolve_frame_image(
                postgres_session, dataset=DATASET, images_dir=images_dir, frame_id=frame_id
            )
            is None
        )
