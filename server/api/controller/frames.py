"""Routes for the benchmark frame set and the frames' image files."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.contracts.frames import BenchmarkFrameSet
from api.core.config import Settings, get_settings
from api.dependencies.database import get_session
from api.services.benchmark import frames as frame_service

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/frames", response_model=BenchmarkFrameSet)
def list_frames(
    session: SessionDep,
    settings: SettingsDep,
    count: Annotated[int | None, Query(ge=1)] = None,
) -> BenchmarkFrameSet:
    """The frame set for one benchmark run, spread across the complexity range."""
    selected = frame_service.select_frames(
        session,
        dataset=settings.dataset,
        images_dir=settings.images_dir,
        count=count or settings.default_frame_count,
    )
    return BenchmarkFrameSet(dataset=settings.dataset, frames=selected)


@router.get("/frames/{frame_id}/image")
def get_frame_image(
    frame_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> FileResponse:
    """A listed frame's original JPEG, byte-for-byte as the desktop measured it."""
    path: Path | None = frame_service.resolve_frame_image(
        session,
        dataset=settings.dataset,
        images_dir=settings.images_dir,
        frame_id=frame_id,
    )
    if path is None:
        raise HTTPException(status_code=404, detail=f"No such benchmark frame: {frame_id!r}")
    return FileResponse(path, media_type="image/jpeg")
