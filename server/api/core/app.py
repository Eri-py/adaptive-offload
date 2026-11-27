"""FastAPI app factory and startup wiring."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from torch import nn

from api.controller import classification, health
from api.core.config import load_settings
from api.services.classification.model import load_model, load_species
from api.services.classification.preprocess import INPUT_SIZE


def _warm_up(model: nn.Module) -> None:
    """One discarded forward pass so CUDA lazy init is not billed to the first request."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    with torch.inference_mode():
        model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device))
        # Kernel launches are async; wait so warmup finishes before serving starts.
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@asynccontextmanager
async def _load_resources(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    app.state.species = load_species(settings.species_labels)
    app.state.model = load_model(settings.large_checkpoint)
    _warm_up(app.state.model)
    yield


@asynccontextmanager
async def _no_resources(app: FastAPI) -> AsyncIterator[None]:
    yield


def create_app(load_resources: bool = True) -> FastAPI:
    """Builds the app; tests pass load_resources=False and inject stubs instead."""
    app = FastAPI(lifespan=_load_resources if load_resources else _no_resources)
    app.include_router(health.router)
    app.include_router(classification.router)
    return app


app = create_app()
