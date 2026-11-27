"""FastAPI dependency providers; tests override these to inject a stub model."""

from fastapi import Request
from torch import nn


def get_loaded_model(request: Request) -> nn.Module | None:
    """The startup-loaded model, or None if startup has not populated it."""
    model: nn.Module | None = getattr(request.app.state, "model", None)
    return model


def get_species(request: Request) -> list[str]:
    """The startup-loaded species list (empty if startup has not populated it)."""
    species: list[str] = getattr(request.app.state, "species", [])
    return species
