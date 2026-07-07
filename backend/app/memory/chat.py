"""Phase 2: the remembering chat.

answer() closes the loop — it READS memory (hybrid retrieval: decompose the question, then
filter photos by entity/time/place or fall back to similarity) and WRITES memory (record the
exchange so the conversation keeps teaching the system). Retrieval lives in retrieval.py; the
write path is record_exchange() from the Phase-1 pipeline.
"""

from dataclasses import dataclass

from sqlmodel import Session, col, select

from app.config import Settings
from app.memory.pipeline import MemoryOperation, record_exchange
from app.memory.prompts import build_chat_messages
from app.memory.retrieval import retrieve
from app.models import Episode

HISTORY_TURNS = 6  # recent turns kept for short-term continuity

@dataclass
class ChatResult:
    reply: str
    memories_used: list[str]
    photos_used: list[str]
    operations: list[MemoryOperation]


def _photo_line(episode: Episode) -> str:
    """A photo memory for the prompt: capture date + place (when known) + its caption."""
    place = ((episode.context or {}).get("place") or {}).get("name")
    where = f", in {place}" if place else ""
    return f"[captured {episode.occurred_at.date().isoformat()}{where}] {episode.content}"


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
    # 1. RETRIEVE — hybrid: decompose the question, filter photos by entity/time/place when it
    #    names them (exact, all matches), else fall back to similarity; facts by similarity
    result = retrieve(session, client, settings, user_id=user_id, message=message)
    memory_texts = [m.content for m in result.facts]
    photo_texts = [_photo_line(e) for e in result.photos]

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