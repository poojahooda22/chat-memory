"""The 'ours' config — the system under test: our Mem0 memory engine.

INGEST replays a LOCOMO conversation through the real write path (record_exchange), one call
per session — so each session is one extraction pass plus its ADD/UPDATE/DELETE/NOOP decisions,
exactly the flow a real multi-turn exchange takes. ANSWER retrieves the most relevant distilled
memories (search_memories — the product's read path) and generates a short answer from just
those.

This is the paper's whole bet: a small set of curated, deduplicated facts answers about as well
as dumping the entire history — at a fraction of the tokens and latency.
"""

from sqlmodel import Session, col, select

from app.config import Settings
from app.memory.embeddings import embed_text
from app.memory.pipeline import record_exchange, search_memories
from app.models import Episode
from eval.common import AnswerResult, timed_answer
from eval.loader import Sample

# Our memory has TWO stores (plan §0b, "retrieval searches both"): distilled semantic facts and
# the episodic layer of raw events. A faithful answer retrieves from both — the facts give the
# gist, the episodes carry the specific detail (an exact date, "painted a sunrise") that
# distillation drops. Retrieving facts only under-measures the architecture.
FACT_K = 10  # distilled facts (the semantic store)
EPISODE_K = 15  # raw dialogue turns (the episodic store)


def _role(speaker: str, speaker_a: str) -> str:
    """LOCOMO has two named speakers; map them onto user/assistant so the episodic layer gets
    alternating roles (the role only labels the episode — the speaker name rides in the text)."""
    return "user" if speaker == speaker_a else "assistant"


def ingest(
    session: Session,
    client,
    settings: Settings,
    sample: Sample,
    *,
    user_id: str,
    conversation_id: str,
) -> None:
    """Replay the conversation into memory — one record_exchange call per session."""
    for dialogue in sample.sessions:
        messages = [
            {
                "role": _role(turn.speaker, sample.speaker_a),
                # keep the speaker name AND the session date inside the text: our episodes'
                # occurred_at is the real ingest time, not LOCOMO's simulated date, so a temporal
                # fact can only be dated if the date travels in the content itself.
                "content": f"[{dialogue.date_time}] {turn.speaker}: {turn.text}",
            }
            for turn in dialogue.turns
        ]
        if not messages:
            continue
        record_exchange(
            session,
            client,
            settings,
            user_id=user_id,
            conversation_id=conversation_id,
            messages=messages,
        )
        session.commit()  # persist this session's episodes + memories before the next one


def _search_episodes(session, client, settings, *, user_id: str, query: str, limit: int):
    """Top-k raw episodes by cosine — the episodic store the product's chat path doesn't yet
    search for text (it only retrieves image episodes). LOCOMO shows why it should."""
    embedding = embed_text(client, settings.embedding_model, query)
    stmt = (
        select(Episode)
        .where(col(Episode.user_id) == user_id, col(Episode.embedding).is_not(None))
        # pyrefly: ignore[missing-attribute]  — pgvector comparator missing from stubs
        .order_by(col(Episode.embedding).cosine_distance(embedding))
        .limit(limit)
    )
    return list(session.exec(stmt).all())


def answer(
    session: Session, client, settings: Settings, question: str, *, user_id: str
) -> AnswerResult:
    def retrieve_context() -> str:
        facts = search_memories(
            session, client, settings, user_id=user_id, query=question, limit=FACT_K
        )
        episodes = _search_episodes(
            session, client, settings, user_id=user_id, query=question, limit=EPISODE_K
        )
        blocks = []
        if facts:
            blocks.append("Facts I remember:\n" + "\n".join(f"- {m.content}" for m in facts))
        if episodes:
            blocks.append(
                "Relevant conversation moments:\n" + "\n".join(f"- {e.content}" for e in episodes)
            )
        return "\n\n".join(blocks) or "(no memories)"

    return timed_answer(retrieve_context, client, settings.llm_model, question)