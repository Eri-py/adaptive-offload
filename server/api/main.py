"""Uvicorn entry point: `uvicorn api.main:app --reload` from `server/`."""

from api.core.app import create_app

app = create_app()
