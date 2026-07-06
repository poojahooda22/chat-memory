"""Phase 2: the remembering chat.

answer() closes the loop — it READS memory (retrieve relevant facts to answer well) and WRITES
memory (record the exchange so the conversation keeps teaching the system). Both halves reuse
the Phase-1 pipeline: search_memories() to retrieve, record_exchange() to store.
"""

from dataclasses import dataclass

from sqlmodel import Session, col, select

from app.config import Settings
from app.memory.pipeline import (
    MemoryOperation,
    record_exchange,
    search_image_episodes,
    search_memories,
)
from app.memory.prompts import build_chat_messages
from app.models import Episode

RETRIEVE_TOP_K = 5   # memories injected into the chat prompt
PHOTO_TOP_K = 3      # image-derived episodes injected alongside them
HISTORY_TURNS = 6    # recent turns kept for short-term continuity


@dataclass
class ChatResult:
    reply: str
    memories_used: list[str]
    photos_used: list[str]
    operations: list[MemoryOperation]


def _recent_turns(session: Session, *, user_id, conversation_id) -> list[dict]:
    """The last few messages of this conversation, chronological, as chat turns."""
    if conversation_id is None:
        return []
    stmt = (
        select(Episode)
        .where(Episode.user_id == user_id, Episode.conversation_id == conversation_id)
        .order_by(col(Episode.occurred_at).desc())
        .limit(HISTORY_TURNS)
    )
    rows = list(session.exec(stmt).all())
    rows.reverse()
    return [{"role": e.context.get("role", "user"), "content": e.content} for e in rows]


def answer(
    session: Session,
    client,
    settings: Settings,
    *,
    user_id: str,
    conversation_id: str | None,
    message: str,
) -> ChatResult:
    # 1. RETRIEVE — both stores: distilled facts AND the photo episodes themselves
    memories = search_memories(
        session, client, settings, user_id=user_id, query=message, limit=RETRIEVE_TOP_K
    )
    memory_texts = [m.content for m in memories]
    photo_episodes = search_image_episodes(
        session, client, settings, user_id=user_id, query=message, limit=PHOTO_TOP_K
    )
    photo_texts = [
        f"[captured {e.occurred_at.date().isoformat()}] {e.content}" for e in photo_episodes
    ]

    # 2. AUGMENT — inject memories + photo memories + recent turns + the new message
    history = _recent_turns(session, user_id=user_id, conversation_id=conversation_id)
    messages = build_chat_messages(memory_texts, photo_texts, history, message)

    # 3. GENERATE — a natural reply (higher temperature than the deterministic pipeline decisions)
    completion = client.chat.completions.create(
        model=settings.llm_model, messages=messages, temperature=0.7
    )
    reply = completion.choices[0].message.content.strip()

    # 4. REMEMBER — feed the exchange back into memory (write path); closes the loop
    recorded = record_exchange(
        session, client, settings,
        user_id=user_id, conversation_id=conversation_id,
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
    )

    return ChatResult(
        reply=reply,
        memories_used=memory_texts,
        photos_used=photo_texts,
        operations=recorded.operations,
    )