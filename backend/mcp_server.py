"""Local stdio MCP server — exposes chat-memory's memory as tools for any MCP host.

Point a host (Claude Desktop, Cursor) at this, or run `uv run python mcp_server.py`. It is the
SINGLE-USER local wrapper: it acts for one user, `MCP_USER_ID`. Multi-tenant remote (per-user
OAuth token propagation over Streamable HTTP) is the Tier-2 upgrade. It reuses the exact memory
engine the REST app uses — the REST API stays the source of truth; this is a second surface.

stdio rule: stdout IS the JSON-RPC stream — never print to it. All logging goes to stderr.
"""

import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from sqlmodel import Session, func, select

from app.config import get_settings
from app.db import build_engine
from app.llm import build_llm_client
from app.memory.graph import neighbors as graph_neighbors_query
from app.memory.pipeline import distil_text
from app.memory.retrieval import recall as recall_bundle
from app.models import Entity

logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # NEVER stdout on a stdio server
log = logging.getLogger("chat-memory-mcp")

_settings = get_settings()
_engine = build_engine(_settings)  # create_engine is lazy — no connection until a tool runs
_client = build_llm_client(_settings)


def _user_id() -> str:
    """The one user this local server acts for. Resolved per-call so importing needs no env."""
    uid = os.environ.get("MCP_USER_ID")
    if not uid:
        raise RuntimeError("MCP_USER_ID is not set — the server needs the user id it acts for")
    return uid


mcp = FastMCP("chat-memory")


# ── structured outputs, so a host gets typed results instead of opaque text ──
class Fact(BaseModel):
    content: str = Field(description="the remembered fact (the CURRENT belief)")
    source: str = Field(description="where it came from: chat | photo | quiz | import | inferred | mcp")
    confidence: float = Field(description="0-1 trust; low = a seeded guess, high = directly observed")
    source_episode_ids: list[str] = Field(description="the episodes this fact was distilled from")
    revised: bool = Field(default=False, description="true if this fact corrected an earlier belief")
    previously: str | None = Field(
        default=None, description="if revised, the prior belief this replaced — do NOT re-assert it"
    )
    ingested_at: str | None = Field(
        default=None,
        description="if revised, ISO time the ENGINE recorded the correction — NOT when the user "
        "changed their mind; never state it as a life-event date",
    )
    has_older: bool = Field(
        default=False,
        description="if true, there were still-earlier revisions too — the change was not direct",
    )


class Recall(BaseModel):
    facts: list[Fact]
    photos: list[str] = Field(description="photo memories, each with its capture date + place")
    confidence: float = Field(description="overall 0-1 trust in this recall (0 = nothing found)")


class Neighbor(BaseModel):
    name: str
    weight: float = Field(description="0-1 bond strength = co-occurrence × recency × confidence")
    shared_photos: int


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def recall(query: str) -> Recall:
    """Recall what is known about the user, relevant to `query`, BEFORE answering anything personal.

    Returns distilled facts — each with its origin and a 0-1 confidence you can threshold on (treat
    a low-confidence fact as a guess, not fact) — plus photo memories, and one overall confidence.
    An empty result with confidence 0 means nothing is known yet; do not invent details.

    Belief revision: when a fact has `revised: true`, the user PREVIOUSLY believed `previously` and
    now believes `content` — answer from `content` and never re-assert `previously`. `ingested_at`
    is when the memory engine RECORDED the correction, not when the user changed their mind, so do
    not narrate it as a life-event date. When `has_older` is true, there were earlier revisions too,
    so do not claim the change was direct."""
    with Session(_engine) as session:
        bundle = recall_bundle(session, _client, _settings, user_id=_user_id(), message=query)
    return Recall(
        facts=[
            Fact(
                content=f.content, source=f.source, confidence=f.confidence,
                source_episode_ids=f.source_episode_ids,
                revised=f.revised, previously=f.previously,
                ingested_at=f.ingested_at.isoformat() if f.ingested_at else None,
                has_older=f.has_older,
            )
            for f in bundle.facts
        ],
        photos=bundle.photo_lines,
        confidence=bundle.confidence,
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    ),
)
def remember(text: str) -> str:
    """Save a durable fact the user shared into their long-term memory. Pass a natural sentence
    ('I'm a backend developer', 'my dog is named Monty'); it is distilled into facts and
    deduplicated against what is already known. Use only for durable facts about the user, not
    passing chit-chat."""
    with Session(_engine) as session:
        ops = distil_text(
            session, _client, _settings,
            user_id=_user_id(), text=text, source_ids=[], source="mcp", confidence=1.0,
        )
        session.commit()
    if not ops:
        return "Nothing durable to remember from that."
    return "Recorded: " + "; ".join(f"{op.event} {op.text}" for op in ops)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def graph_neighbors(entity: str) -> list[Neighbor]:
    """Explore the user's relationship graph: given a person/pet/thing they have named (e.g. 'Monty'),
    return the entities that co-occur with it in their photos — strongest bond first — with the edge
    weight and how many photos they share. Empty if the entity is unknown or has no connections."""
    with Session(_engine) as session:
        entity_row = session.exec(
            select(Entity).where(
                Entity.user_id == _user_id(),
                func.lower(Entity.name) == entity.strip().lower(),
            )
        ).first()
        if entity_row is None:
            return []
        pairs = graph_neighbors_query(session, user_id=_user_id(), entity_id=entity_row.id)
        return [
            Neighbor(name=other.name, weight=edge.weight, shared_photos=edge.cooccur_count)
            for other, edge in pairs
        ]


if __name__ == "__main__":
    log.info("chat-memory MCP server starting (stdio) for user %s", os.environ.get("MCP_USER_ID"))
    mcp.run()  # stdio transport by default