"""The Mem0 two-phase memory pipeline.

record_exchange() is the write path: store the raw messages as episodes (episodic paper),
extract durable facts (phase 1), then for each fact decide ADD / UPDATE / DELETE / NOOP
against the most similar existing memories (phase 2), recording every change in the audit
trail. search_memories() is the read path Phase 2's chat will reuse.

The LLM client and Settings are passed in, never imported globally, so tests inject fakes.
"""

from dataclasses import dataclass, field

from sqlmodel import Session, col, select

from app.config import Settings
from app.memory.embeddings import embed_text
from app.memory.prompts import (
    build_decision_messages,
    build_extraction_messages,
    build_summary_messages,
    parse_json,
)
from app.models import ConversationSummary, Episode, Memory, MemoryHistory

RECENT_WINDOW = 10  # paper's m: recent messages fed to extraction
SIMILAR_TOP_K = 10  # paper's s: similar memories fed to the decision


@dataclass
class MemoryOperation:
    event: str  # ADD | UPDATE | DELETE | NOOP
    memory_id: str | None
    text: str


@dataclass
class RecordResult:
    episode_ids: list[str] = field(default_factory=list)
    operations: list[MemoryOperation] = field(default_factory=list)


def _chat_json(client, model: str, messages: list[dict]) -> dict:
    """One chat completion parsed to a dict.

    We do NOT pass response_format: the prompts already demand strict JSON, parse_json is
    robust to stray prose/fences, and this keeps the call portable across the gateway and
    Ollama (the gateway rejects response_format for some models).
    """
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return parse_json(response.choices[0].message.content)


def _store_episodes(
    session: Session, client, settings: Settings, *, user_id, conversation_id, messages
) -> list[Episode]:
    """Write each message of the exchange as an episode — single-shot, embedded, never rewritten."""
    episodes: list[Episode] = []
    for msg in messages:
        content = msg["content"]
        episode = Episode(
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
            context={"role": msg["role"], "source": "chat"},
            embedding=embed_text(client, settings.embedding_model, content),
        )
        session.add(episode)
        episodes.append(episode)
    session.flush()  # assign ids, make visible to later queries in this transaction
    return episodes


def _recent_episode_texts(session: Session, *, user_id, conversation_id) -> list[str]:
    if conversation_id is None:
        return []
    stmt = (
        select(Episode)
        .where(Episode.user_id == user_id, Episode.conversation_id == conversation_id)
        .order_by(col(Episode.occurred_at).desc())
        .limit(RECENT_WINDOW)
    )
    rows = list(session.exec(stmt).all())
    rows.reverse()  # chronological
    return [f'{e.context.get("role", "user")}: {e.content}' for e in rows]


def _load_summary(session: Session, conversation_id) -> str:
    if conversation_id is None:
        return ""
    row = session.exec(
        select(ConversationSummary).where(
            ConversationSummary.conversation_id == conversation_id
        )
    ).first()
    return row.summary if row else ""


def _search_similar(session: Session, *, user_id, embedding, limit) -> list[Memory]:
    """Top-k live memories by cosine distance on the pgvector column."""
    stmt = (
        select(Memory)
        .where(Memory.user_id == user_id, Memory.is_deleted == False)  # noqa: E712
        # pgvector's cosine_distance comparator isn't in SQLAlchemy's type stubs
        .order_by(col(Memory.embedding).cosine_distance(embedding))  # pyrefly: ignore[missing-attribute]
        .limit(limit)
    )
    return list(session.exec(stmt).all())


def record_exchange(
    session: Session,
    client,
    settings: Settings,
    *,
    user_id: str,
    conversation_id: str | None,
    messages: list[dict],
) -> RecordResult:
    result = RecordResult()

    # ── episodic layer: raw events, written once ──
    recent = _recent_episode_texts(session, user_id=user_id, conversation_id=conversation_id)
    episodes = _store_episodes(
        session, client, settings,
        user_id=user_id, conversation_id=conversation_id, messages=messages,
    )
    result.episode_ids = [str(e.id) for e in episodes]
    source_ids = [str(e.id) for e in episodes]

    # ── phase 1: extraction ──
    summary = _load_summary(session, conversation_id)
    extraction = _chat_json(
        client, settings.llm_model, build_extraction_messages(summary, recent, messages)
    )
    facts = [f.strip() for f in extraction.get("facts", []) if isinstance(f, str) and f.strip()]

    # ── phase 2: per-fact ADD / UPDATE / DELETE / NOOP ──
    for fact in facts:
        fact_embedding = embed_text(client, settings.embedding_model, fact)
        similar = _search_similar(
            session, user_id=user_id, embedding=fact_embedding, limit=SIMILAR_TOP_K
        )
        decision = _chat_json(
            client, settings.llm_model,
            build_decision_messages(fact, [m.content for m in similar]),
        )
        op = _apply_decision(
            session, client, settings,
            fact=fact, fact_embedding=fact_embedding, similar=similar,
            decision=decision, user_id=user_id, source_ids=source_ids,
        )
        result.operations.append(op)

    return result


def _apply_decision(
    session, client, settings, *, fact, fact_embedding, similar, decision, user_id, source_ids
) -> MemoryOperation:
    event = str(decision.get("event", "NOOP")).upper()
    idx = decision.get("target_index")
    target = similar[idx] if isinstance(idx, int) and 0 <= idx < len(similar) else None

    if event == "ADD":
        memory = Memory(
            user_id=user_id,
            content=decision.get("text") or fact,
            embedding=fact_embedding,
            source_episode_ids=source_ids,
        )
        session.add(memory)
        session.flush()
        session.add(MemoryHistory(memory_id=memory.id, event="ADD", new_content=memory.content))
        return MemoryOperation("ADD", str(memory.id), memory.content)

    if event == "UPDATE" and target is not None:
        old = target.content
        new_text = decision.get("text") or fact
        target.content = new_text
        target.embedding = embed_text(client, settings.embedding_model, new_text)
        target.source_episode_ids = list(dict.fromkeys(target.source_episode_ids + source_ids))
        session.add(target)
        session.add(
            MemoryHistory(memory_id=target.id, event="UPDATE", old_content=old, new_content=new_text)
        )
        return MemoryOperation("UPDATE", str(target.id), new_text)

    if event == "DELETE" and target is not None:
        target.is_deleted = True
        session.add(target)
        session.add(
            MemoryHistory(memory_id=target.id, event="DELETE", old_content=target.content)
        )
        return MemoryOperation("DELETE", str(target.id), target.content)

    return MemoryOperation("NOOP", None, fact)


def search_memories(
    session: Session, client, settings: Settings, *, user_id: str, query: str, limit: int = 5
) -> list[Memory]:
    """Read path: the live memories most relevant to a query. Phase 2's chat reuses this."""
    embedding = embed_text(client, settings.embedding_model, query)
    return _search_similar(session, user_id=user_id, embedding=embedding, limit=limit)


def run_summary_refresh(engine, client, settings: Settings, conversation_id: str) -> None:
    """Background entry point: refresh a conversation's summary in its own session.

    Shared by the /memories and /chat routes — a lost run is harmless (the next exchange just
    uses a slightly older summary), so it runs off the request path via BackgroundTasks.
    """
    with Session(engine) as session:
        refresh_summary(session, client, settings, conversation_id)
        session.commit()


def refresh_summary(session: Session, client, settings: Settings, conversation_id: str) -> None:
    """Regenerate the rolling summary for a conversation (runs off the request path)."""
    episodes = list(
        session.exec(
            select(Episode)
            .where(Episode.conversation_id == conversation_id)
            .order_by(col(Episode.occurred_at))
        ).all()
    )
    if not episodes:
        return
    texts = [f'{e.context.get("role", "user")}: {e.content}' for e in episodes]
    reply = _chat_json(client, settings.llm_model, build_summary_messages(texts))
    summary_text = reply.get("summary", "")

    row = session.exec(
        select(ConversationSummary).where(
            ConversationSummary.conversation_id == conversation_id
        )
    ).first()
    if row is None:
        session.add(ConversationSummary(conversation_id=conversation_id, summary=summary_text))
    else:
        row.summary = summary_text
        session.add(row)