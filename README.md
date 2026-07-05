# chat-memory

A memory layer for AI assistants, built from two research papers:

- **Mem0** ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)) — the memory engine: extract
  facts from conversation, then decide per fact whether to ADD, UPDATE, DELETE, or skip,
  against the most similar existing memories.
- **Episodic Memory is the Missing Piece for Long-Term LLM Agents**
  ([arXiv:2502.06975](https://arxiv.org/abs/2502.06975)) — the memory spec: memories should be
  episodes first — timestamped events bound to their context (when, where, why) — with
  semantic facts distilled from them.

The result is a chat assistant that remembers you across sessions and can show exactly
where every memory came from: each fact links back to the episode that created it, and
every change to a memory is recorded in an audit trail.

## Architecture

```
frontend (React, later)  →  FastAPI backend  →  Postgres + pgvector
                                 │
                                 └→ LLM + embeddings via one OpenAI-compatible client
```

Four tables carry the whole design:

| Table | Role |
|---|---|
| `episodes` | Timestamped events with context — the raw diary |
| `memories` | Semantic facts distilled from episodes, embedded for similarity search |
| `memory_history` | Every ADD / UPDATE / DELETE — the audit trail |
| `conversation_summaries` | Rolling summary per conversation, used by the extraction step |

## Run it locally

Requirements: Docker Desktop, [uv](https://docs.astral.sh/uv/), Python 3.12+.

```sh
# 1. start Postgres (with pgvector) on port 5434
docker compose up -d

# 2. install backend dependencies
cd backend
uv sync

# 3. configure secrets
cp .env.example .env   # then fill in AI_GATEWAY_API_KEY

# 4. create the database schema
uv run alembic upgrade head

# 5. start the API (port 8000 by default; use another if it's taken)
uv run uvicorn app.main:app --reload --port 8001
```

Open http://localhost:8001/docs for the interactive API explorer,
or http://localhost:8001/health for the heartbeat.

Run the tests:

```sh
cd backend
uv run pytest
```

## Status

Phase 0 — service skeleton: app boots, schema exists, health check green.
The extraction/update pipeline, chat, dashboard, and evaluation land in later phases.
