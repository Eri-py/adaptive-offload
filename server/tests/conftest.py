"""Shared fixtures for `server/tests/`.

The `client` fixture swaps the model/species providers for a stub so no test
needs a GPU or the 335 MB checkpoint. `postgres_engine` mirrors
`database/tests/conftest.py` (scoped CLAUDE.md exception: ephemeral test DBs only).
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from common.testing import ephemeral_postgres_database
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from torch import nn

from api.core.app import create_app
from api.dependencies.providers import get_loaded_model, get_species

load_dotenv(Path(__file__).resolve().parent.parent.parent / "database" / ".env")

NUM_CLASSES = 200


class StubModel(nn.Module):
    """Returns fixed logits peaked at class 7, regardless of input."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(batch.shape[0], NUM_CLASSES)
        logits[:, 7] = 5.0
        logits[:, 3] = 3.0
        logits[:, 11] = 1.0
        return logits


@pytest.fixture
def stub_species() -> list[str]:
    return [f"Species {index}" for index in range(NUM_CLASSES)]


@pytest.fixture
def app(stub_species: list[str]) -> FastAPI:
    application = create_app(load_resources=False)
    stub = StubModel().eval()
    application.dependency_overrides[get_loaded_model] = lambda: stub
    application.dependency_overrides[get_species] = lambda: stub_species
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    """Function-scoped engine bound to a fresh, uniquely-named ephemeral test DB."""
    admin_url = os.environ.get("POSTGRES_ADMIN_URL")
    if not admin_url:
        raise RuntimeError(
            "POSTGRES_ADMIN_URL is not set. Add it to database/.env "
            "(a Postgres admin connection string pointed at the 'postgres' database)."
        )
    with ephemeral_postgres_database(admin_url) as engine:
        yield engine
