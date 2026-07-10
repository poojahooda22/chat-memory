"""Communication-style inference — the identity trait nobody states.

People rarely SAY "I write terse and lead with the answer"; they demonstrate it across many
messages. This module reads the user's own recent chat messages (never the assistant's), asks
the LLM to describe HOW they write as a few durable traits, and lands each trait through the
SAME phase-2 decision pipeline every fact uses — `source="inferred"`, mid confidence, and
provenance pointing at the exact messages the trait was read from. Re-running goes through the
UPDATE decision, which retires the previous inference and writes a successor — so the style
profile tracks the user instead of freezing at the first guess.

Trust ordering, by design: quiz-stated style (0.6) < behavior-inferred (0.7) < directly
stated in chat (1.0). A user saying "give me short answers" still outranks what we infer.
"""

from sqlmodel import Session, col, func, select

from app.config import Settings
from app.memory.pipeline import MemoryOperation, _chat_json, record_fact
from app.memory.prompts import build_style_messages
from app.models import Episode

INFERRED_CONFIDENCE = 0.7  # read off real behavior: above a quiz guess, below a direct statement
MIN_MESSAGES = 8  # below this the "style" is noise — skip rather than guess
SAMPLE_SIZE = 30  # most recent user-authored chat messages fed to the inference
REFRESH_EVERY = 10  # auto re-infer once every N new user messages
MAX_TRAITS = 4  # the prompt asks for 2-4; enforce the ceiling even if the LLM overreturns


def _user_message_filter(user_id: str):
    """The style sample is the user's own chat messages ONLY — never the assistant's replies
    (that would be OUR style), never photo/quiz episodes (no writing behavior in them)."""
    return (
        Episode.user_id == user_id,
        col(Episode.context)["source"].astext == "chat",
        col(Episode.context)["role"].astext == "user",
    )


def _recent_user_messages(session: Session, *, user_id: str, limit: int) -> list[Episode]:
    stmt = (
        select(Episode)
        .where(*_user_message_filter(user_id))
        .order_by(col(Episode.occurred_at).desc())
        .limit(limit)
    )
    rows = list(session.exec(stmt).all())
    rows.reverse()  # chronological, oldest first — matches the prompt's framing
    return rows


def infer_style(
    session: Session, client, settings: Settings, *, user_id: str
) -> list[MemoryOperation]:
    """One inference pass: sample -> LLM trait read -> each trait through the decision pipeline.

    Each trait fact's provenance (`source_episode_ids`) is the sampled messages themselves —
    the receipts are the real behavior the trait was read from, not a synthetic episode."""
    episodes = _recent_user_messages(session, user_id=user_id, limit=SAMPLE_SIZE)
    if len(episodes) < MIN_MESSAGES:
        return []
    reply = _chat_json(
        client, settings.llm_model, build_style_messages([e.content for e in episodes])
    )
    traits = [t.strip() for t in reply.get("traits", []) if isinstance(t, str) and t.strip()]
    source_ids = [str(e.id) for e in episodes]
    return [
        record_fact(
            session, client, settings,
            user_id=user_id, fact=trait, source_ids=source_ids,
            source="inferred", confidence=INFERRED_CONFIDENCE,
        )
        for trait in traits[:MAX_TRAITS]
    ]


def run_style_refresh(engine, client, settings: Settings, *, user_id: str) -> None:
    """Background entry point (manual trigger): infer now — own session, own commit."""
    with Session(engine) as session:
        infer_style(session, client, settings, user_id=user_id)
        session.commit()


def maybe_refresh_style(engine, client, settings: Settings, *, user_id: str) -> None:
    """Background entry point (auto trigger): after a chat exchange, re-infer roughly once every
    REFRESH_EVERY user messages.

    This is a best-effort trigger, not an exact schedule: it counts messages AFTER the exchange
    commits, so interleaved concurrent writes can step the count past a multiple and skip a
    refresh — it self-heals at the next multiple, and the manual POST /style/refresh always
    forces one. A per-user watermark (fire when count - last >= REFRESH_EVERY, guarded) is the
    exact-and-idempotent upgrade (see the Tier-2 notes). Correctness never depends on this firing:
    the trust guard means a stale style profile is only ever slightly behind, never wrong."""
    with Session(engine) as session:
        count = session.exec(
            select(func.count()).select_from(Episode).where(*_user_message_filter(user_id))
        ).one()
    if count < MIN_MESSAGES or count % REFRESH_EVERY != 0:
        return
    run_style_refresh(engine, client, settings, user_id=user_id)