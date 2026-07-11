"""Phase 2: the remembering chat.

answer() closes the loop — it READS memory (hybrid retrieval: decompose the question, then
filter photos by entity/time/place or fall back to similarity) and WRITES memory (record the
exchange so the conversation keeps teaching the system). Retrieval lives in retrieval.py; the
write path is record_exchange() from the Phase-1 pipeline.
"""

from dataclasses import dataclass, field

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
    dialogue_used: list[str] = field(default_factory=list)


def recent_turn_episodes(session: Session, *, user_id, conversation_id) -> list[Episode]:
    """The last few episodes of this conversation, newest first — the SINGLE source both the recent-
    turns prompt block and the dialogue-search exclusion read from, so the set injected verbatim is
    exactly the set excluded from cross-conversation recall (no double-injection, no dropped turn).
    Deterministic id tiebreaker so the boundary row is stable across readers."""
    if conversation_id is None:
        return []
    stmt = (
        select(Episode)
        .where(Episode.user_id == user_id, Episode.conversation_id == conversation_id)
        .order_by(col(Episode.occurred_at).desc(), col(Episode.id).desc())
        .limit(HISTORY_TURNS)
    )
    return list(session.exec(stmt).all())


@dataclass
class PreparedReply:
    messages: list[dict]
    memories_used: list[str]
    photos_used: list[str]
    dialogue_used: list[str] = field(default_factory=list)


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
    # The current conversation's recent turns, read ONCE: they become the short-term history block
    # AND the exclusion set for cross-conversation dialogue search (so a turn is never both injected
    # verbatim and recalled as an excerpt, and older turns of THIS conversation stay searchable).
    recent = recent_turn_episodes(session, user_id=user_id, conversation_id=conversation_id)
    exclude_ids = [e.id for e in recent]
    history = [
        {"role": e.context.get("role", "user"), "content": e.content} for e in reversed(recent)
    ]

    # RETRIEVE — hybrid recall: facts + photos + PAST chat excerpts, as a receipts-carrying bundle.
    bundle = recall(
        session, client, settings, user_id=user_id, message=message,
        exclude_episode_ids=exclude_ids, user_tz=settings.user_tz,
    )
    memory_texts = bundle.fact_lines
    photo_texts = bundle.photo_lines
    dialogue_texts = bundle.dialogue_lines

    # AUGMENT — inject memories + photo memories + past-conversation excerpts + recent turns
    messages = build_chat_messages(memory_texts, photo_texts, dialogue_texts, history, message)
    return PreparedReply(
        messages=messages, memories_used=memory_texts, photos_used=photo_texts,
        dialogue_used=dialogue_texts,
    )


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
        reply=reply, memories_used=prepared.memories_used, photos_used=prepared.photos_used,
        dialogue_used=prepared.dialogue_used,
    )