from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import chat, health, memories, uploads
from app.config import get_settings
from app.db import build_engine
from app.llm import build_llm_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # shared resources: created once at startup, closed once at shutdown
    settings = get_settings()
    app.state.engine = build_engine(settings)
    app.state.llm = build_llm_client(settings)
    yield
    app.state.engine.dispose()


app = FastAPI(
    title="chat-memory",
    version="0.1.0",
    description="Memory layer for AI assistants: episodic + semantic memory with provenance.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(memories.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(uploads.router, prefix="/api/v1")


def _problem(status: int, title: str, detail: str, instance: str) -> JSONResponse:
    """RFC 9457 problem+json envelope — the one error shape this API returns."""
    return JSONResponse(
        status_code=status,
        content={"type": "about:blank", "title": title, "status": status,
                 "detail": detail, "instance": instance},
        media_type="application/problem+json",
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _problem(exc.status_code, exc.detail or "HTTP error", str(exc.detail), str(request.url))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _problem(422, "Validation failed", str(exc.errors()), str(request.url))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # never leak a stack trace to the client; the server log has the real error
    return _problem(500, "Internal server error", "An unexpected error occurred.",
                    str(request.url))