"""The taste-quiz seeder: an onboarding answer becomes a LOW-confidence memory (source='quiz'),
distilled through the same pipeline as chat, with a provenance episode. Zero tokens (fake LLM)."""

import json
import uuid

from sqlmodel import select

from app.config import get_settings
from app.memory.onboarding import SEED_CONFIDENCE, _seed_answer
from app.models import Episode, Memory
from tests.conftest import FakeLLM

SETTINGS = get_settings()


def _live_memories(session, user_id):
    return list(
        session.exec(
            select(Memory).where(Memory.user_id == user_id, Memory.is_deleted == False)  # noqa: E712
        ).all()
    )


def test_quiz_answer_seeds_low_confidence_fact_with_provenance(db_session):
    user = f"q-{uuid.uuid4()}"
    # extraction finds one durable fact from the answer, the decision ADDs it
    llm = FakeLLM([
        json.dumps({"facts": ["Works as a backend developer"]}),
        json.dumps({"event": "ADD", "target_index": None, "text": "Works as a backend developer"}),
    ])
    _seed_answer(
        db_session, llm, SETTINGS,
        user_id=user, question="What do you do?", answer="I'm a backend developer who loves Go",
    )
    db_session.flush()

    seeded = next(m for m in _live_memories(db_session, user) if "backend" in m.content)
    assert seeded.source == "quiz"  # tagged as a quiz seed (the P1 origin)
    assert seeded.confidence == SEED_CONFIDENCE  # trusted less than an observed chat fact
    # a provenance episode was written, so the fact traces back to the onboarding answer
    episodes = list(db_session.exec(select(Episode).where(Episode.user_id == user)).all())
    assert any((e.context or {}).get("source") == "quiz" for e in episodes)