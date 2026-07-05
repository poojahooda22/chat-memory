from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine
from sqlmodel import Session, create_engine

from app.config import Settings


def build_engine(settings: Settings) -> Engine:
    """Create the one engine (connection pool). Owned by the app lifespan."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def get_session(request: Request) -> Iterator[Session]:
    """Per-request DB session, drawn from the lifespan-owned engine on app.state."""
    with Session(request.app.state.engine) as session:
        yield session