"""Phase 2: the remembering chat.

answer() closes the loop — it READS memory (hybrid retrieval: decompose the question, then
filter photos by entity/time/place or fall back to similarity) and WRITES memory (record the
exchange so the conversation keeps teaching the system). Retrieval lives in retrieval.py; the
write path is record_exchange() from the Phase-1 pipeline.
"""

from dataclasses import dataclass

from sqlmodel import Session, col, select

from app.config import Settings
from app.memory.prompts import build_chat_messages
from app.memory.retrieval import recall
from app.models import Episode

HISTORY_TURNS = 6  # recent turns kept for short-term continuity

@dataclass
class ChatResult:
    reply: str
    memories_used: list[str]
    photos_used: list[str]


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


@dataclass
class PreparedReply:
    messages: list[dict]
    memories_used: list[str]
    photos_used: list[str]


def prepare_reply(
    session: Session,
    client,
    settings: Settings,
    *,
    user_id: str,
    conversation_id: str | None,
    message: str,
) -> PreparedReply:
    """RETRIEVE + AUGMENT: hybrid retrieval, then build the prompt messages. GENERATE is the
    caller's step — the chat route STREAMS it token by token; answer() runs it in one shot. Both
    share this so retrieval + prompt-building stay identical."""
    # RETRIEVE — hybrid recall: decompose the question, filter/rank facts + photos, return them as
    # a receipts-carrying bundle. The prompt uses only the text lines; the provenance + confidence
    # on the bundle are for the MCP layer + proactive gate (P2).
    bundle = recall(session, client, settings, user_id=user_id, message=message)
    memory_texts = bundle.fact_lines
    photo_texts = bundle.photo_lines

    # AUGMENT — inject memories + photo memories + recent turns + the new message
    history = _recent_turns(session, user_id=user_id, conversation_id=conversation_id)
    messages = build_chat_messages(memory_texts, photo_texts, history, message)
    return PreparedReply(messages=messages, memories_used=memory_texts, photos_used=photo_texts)


def answer(
    session: Session,
    client,
    settings: Settings,
    *,
    user_id: str,
    conversation_id: str | None,
    message: str,
) -> ChatResult:
    """Non-streaming reply — used by tests and any programmatic caller. The chat route streams
    instead (see routes/chat.py). The REMEMBER step is off the request path (learn_from_exchange)."""
    prepared = prepare_reply(
        session, client, settings,
        user_id=user_id, conversation_id=conversation_id, message=message,
    )
    completion = client.chat.completions.create(
        model=settings.llm_model, messages=prepared.messages, temperature=0.7
    )
    reply = completion.choices[0].message.content.strip()
    return ChatResult(
        reply=reply, memories_used=prepared.memories_used, photos_used=prepared.photos_used
    )