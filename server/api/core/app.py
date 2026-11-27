"""FastAPI app factory and startup wiring."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.controller import health
from api.core.config import load_settings
from api.services.classification.model import load_model, load_species


@asynccontextmanager
async def _load_resources(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    app.state.species = load_species(settings.species_labels)
    app.state.model = load_model(settings.large_checkpoint)
    yield


@asynccontextmanager
async def _no_resources(app: FastAPI) -> AsyncIterator[None]:
    yield


def create_app(load_resources: bool = True) -> FastAPI:
    """Builds the app; tests pass load_resources=False and inject stubs instead."""
    app = FastAPI(lifespan=_load_resources if load_resources else _no_resources)
    app.include_router(health.router)
    return app


app = create_app()
