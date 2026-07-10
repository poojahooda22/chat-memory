"""The taste-quiz cold-start seeder — give a brand-new user's memory something to know at turn one.

A first-ever session is empty, so the assistant knows nothing about you. This asks a few
funnel-ordered questions (general -> specific, all skippable) and turns each answer into memory
through the SAME extraction pipeline a chat message uses — but seeded at LOW confidence with
`source="quiz"`, because a quiz answer is a stated *guess*, not something observed in use. That
is exactly what the P1 trust model is for: a later, directly-observed chat fact can supersede a
quiz seed, and retrieval trusts it less until then.
"""

from sqlmodel import Session

from app.config import Settings
from app.memory.embeddings import embed_text
from app.memory.pipeline import MemoryOperation, distil_text
from app.models import Episode

# a quiz answer is a low-confidence asserted preference, not an observed fact — trust it less
SEED_CONFIDENCE = 0.6

# funnel: general -> specific, skippable. Each yields rich, durable facts when answered.
QUESTIONS: list[dict[str, str]] = [
    {"id": "work", "prompt": "What do you do — work, studies, or how you spend your days?"},
    {"id": "interests", "prompt": "What are you into? Hobbies, interests, things you love."},
    {"id": "people", "prompt": "Who matters most to you — the people or pets in your life?"},
    {"id": "goals", "prompt": "What are you working toward right now — a goal or a project?"},
    {"id": "style", "prompt": "How do you like an assistant to talk to you — casual, concise, detailed?"},
]


def _seed_answer(
    session: Session, client, settings: Settings, *, user_id: str, question: str, answer: str
) -> list[MemoryOperation]:
    """One answer -> a `source="quiz"` episode (provenance) + distilled facts at seed confidence."""
    episode = Episode(
        user_id=user_id,
        conversation_id=None,
        content=answer,
        context={"source": "quiz", "question": question},
        embedding=embed_text(client, settings.embedding_model, answer),
    )
    session.add(episode)
    session.flush()  # assign the id so the facts can point back to it
    return distil_text(
        session, client, settings,
        user_id=user_id, text=answer, source_ids=[str(episode.id)],
        source="quiz", confidence=SEED_CONFIDENCE,
    )


def seed_from_quiz(engine, client, settings: Settings, *, user_id: str, answers: list[dict]) -> None:
    """Background entry point: turn onboarding answers into seeded memories, off the request path
    (the submit returns immediately; the Memory page fills in). Own session, own commit."""
    with Session(engine) as session:
        for item in answers:
            answer = str(item.get("answer") or "").strip()
            if answer:
                _seed_answer(
                    session, client, settings,
                    user_id=user_id, question=str(item.get("question") or ""), answer=answer,
                )
        session.commit()