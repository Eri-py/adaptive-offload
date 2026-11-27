"""Request/response models for the classification endpoints."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness plus whether the model finished loading."""

    status: str
    model_loaded: bool


class SpeciesProbability(BaseModel):
    """One ranked species guess."""

    index: int
    name: str
    probability: float


class ClassifyResponse(BaseModel):
    """Top-1 result, confidence measures and timing for one classified image."""

    species: str
    class_index: int
    max_probability: float
    top2_margin: float
    top3: list[SpeciesProbability]
    compute_ms: float
    geometric_transform_applied: bool
