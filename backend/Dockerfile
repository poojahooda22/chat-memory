# build stage: resolve dependencies with uv against the committed lockfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# runtime stage: slim image, venv on PATH, no build tooling
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# Bind to the platform-provided $PORT (Render/most PaaS inject it; default 8000 locally).
# Apply migrations to head on boot, then exec uvicorn so it becomes PID 1 and receives SIGTERM
# for graceful shutdown. Migrations are idempotent — a no-op once the DB is already at head.
CMD alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}