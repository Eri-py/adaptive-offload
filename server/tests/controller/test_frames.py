"""Endpoint tests for the benchmark frame list and frame images.

One test per acceptance criterion in
`features/server-served-benchmark/01-server-frame-endpoints/spec.md`.
"""

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.core.app import create_app
from api.core.config import Settings, get_settings
from api.dependencies.database import get_session
from tests.conftest import ClientFactory
from tests.support import DATASET, FrameFixture, frame_series, jpeg_bytes

FRAMES_URL = "/benchmark/frames"


def test_defaults_to_fifteen_frames(build_client: ClientFactory) -> None:
    client = build_client(frame_series(20))

    body = client.get(FRAMES_URL).json()

    assert body["dataset"] == DATASET
    assert len(body["frames"]) == 15
    for frame in body["frames"]:
        assert frame["frame_id"]
        assert isinstance(frame["scene_complexity"], float)
        assert isinstance(frame["desktop_local_latency_ms"], float)


def test_returns_the_requested_count(build_client: ClientFactory) -> None:
    client = build_client(frame_series(20))

    assert len(client.get(FRAMES_URL, params={"count": 5}).json()["frames"]) == 5


def test_returns_all_eligible_when_fewer_than_requested(build_client: ClientFactory) -> None:
    client = build_client(frame_series(3))

    assert len(client.get(FRAMES_URL, params={"count": 10}).json()["frames"]) == 3


def test_spans_the_complexity_range(build_client: ClientFactory) -> None:
    seeded = frame_series(20)
    client = build_client(seeded)

    body = client.get(FRAMES_URL, params={"count": 5}).json()
    scores = [frame["scene_complexity"] for frame in body["frames"]]

    assert scores == sorted(scores)
    assert scores[0] == min(frame.scene_complexity for frame in seeded)
    assert scores[-1] == max(frame.scene_complexity for frame in seeded)


def test_is_deterministic_across_identical_requests(build_client: ClientFactory) -> None:
    client = build_client(frame_series(20))

    first = client.get(FRAMES_URL, params={"count": 7}).json()
    second = client.get(FRAMES_URL, params={"count": 7}).json()

    assert first == second


def test_excludes_frames_without_a_local_timing_or_an_image(build_client: ClientFactory) -> None:
    eligible = frame_series(3)
    client = build_client(
        [
            *eligible,
            FrameFixture(file_name="untimed.jpg", scene_complexity=0.5, local_latency_ms=None),
            FrameFixture(
                file_name="offload_only.jpg",
                scene_complexity=0.6,
                local_latency_ms=None,
                offload_latency_ms=99.0,
            ),
            FrameFixture(file_name="missing_file.jpg", scene_complexity=0.7, on_disk=False),
        ]
    )

    returned = {f["frame_id"] for f in client.get(FRAMES_URL).json()["frames"]}

    assert returned == {frame.file_name for frame in eligible}


def test_serves_the_image_byte_for_byte_as_jpeg(build_client: ClientFactory) -> None:
    client = build_client(frame_series(1))

    response = client.get(f"{FRAMES_URL}/frame_000.jpg/image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == jpeg_bytes("frame_000.jpg")


def test_unknown_frame_id_is_not_found(build_client: ClientFactory) -> None:
    client = build_client(frame_series(1))

    assert client.get(f"{FRAMES_URL}/not_a_frame.jpg/image").status_code == 404


def test_an_id_reaching_outside_the_image_folder_is_not_found(
    build_client: ClientFactory, images_dir: Path
) -> None:
    secret = images_dir.parent / "secret.jpg"
    secret.write_bytes(b"do not serve me")
    client = build_client(frame_series(1))

    response = client.get(f"{FRAMES_URL}/..%2Fsecret.jpg/image")

    assert response.status_code == 404
    assert secret.read_bytes() not in response.content


def test_says_so_when_the_database_is_unavailable(settings: Settings) -> None:
    # Port 1 refuses immediately, which is what Postgres being down looks like.
    dead_engine = create_engine("postgresql+psycopg://nobody:nobody@127.0.0.1:1/nothing")

    def dead_session() -> Iterator[Session]:
        with Session(dead_engine) as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = dead_session
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as client:
        response = client.get(FRAMES_URL)

    assert response.status_code == 503
    assert "database is unavailable" in response.json()["detail"].lower()
