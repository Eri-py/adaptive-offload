"""Request/response models for the classification endpoints."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness plus whether the model finished loading."""

    status: str
    model_loaded: bool
