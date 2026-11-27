"""FastAPI dependency providers; tests override these to inject a stub model."""

from typing import Annotated

import torch
from fastapi import Depends, HTTPException, Request
from torch import nn


def get_loaded_model(request: Request) -> nn.Module | None:
    """The startup-loaded model, or None if startup has not populated it."""
    model: nn.Module | None = getattr(request.app.state, "model", None)
    return model


def get_species(request: Request) -> list[str]:
    """The startup-loaded species list (empty if startup has not populated it)."""
    species: list[str] = getattr(request.app.state, "species", [])
    return species


def get_required_model(model: Annotated[nn.Module | None, Depends(get_loaded_model)]) -> nn.Module:
    """The loaded model, or HTTP 503 so callers never hit an attribute error on None."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return model


def get_device(model: Annotated[nn.Module, Depends(get_required_model)]) -> torch.device:
    """The model's device; parameterless stub models fall back to CPU."""
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


async def get_raw_body(request: Request) -> bytes:
    """Raw request bytes; avoids a multipart envelope so payload size matches the image."""
    return await request.body()
