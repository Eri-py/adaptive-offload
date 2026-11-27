"""Health endpoint; reads state only so it is cheap enough to use as an RTT probe."""

from typing import Annotated

from fastapi import APIRouter, Depends
from torch import nn

from api.contracts.classification import HealthResponse
from api.dependencies.providers import get_loaded_model

router = APIRouter()


@router.get("/health")
def health(model: Annotated[nn.Module | None, Depends(get_loaded_model)]) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None)
