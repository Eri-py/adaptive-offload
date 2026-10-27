"""FastAPI app construction and wiring."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import InterfaceError, OperationalError

from api.controller import frames

# Connection-level SQLAlchemy failures: Postgres down, refusing connections, or
# dropping them mid-request. Anything narrower leaks a 500 when the DB is off.
_DATABASE_UNAVAILABLE_ERRORS = (OperationalError, InterfaceError)


def create_app() -> FastAPI:
    """Build the app with its routes and database-unavailable handling."""
    app = FastAPI(title="adaptive-offload server", version="0.1.0")
    app.include_router(frames.router)

    async def database_unavailable(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "The database is unavailable."},
        )

    for error in _DATABASE_UNAVAILABLE_ERRORS:
        app.add_exception_handler(error, database_unavailable)

    return app
