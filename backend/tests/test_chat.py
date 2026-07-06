"""Phase 2: the remembering chat reads memory to answer, and writes memory as it talks.

Zero tokens — the fake LLM scripts the reply and the follow-up extraction/decision.
"""

import json
import uuid

from sqlmodel import select

from app.config import get_settings
from app.memory.chat import answer
from app.models import Episode, Memory
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()


def _user() -> str:
    return f"test-{uuid.uuid4()}"


def _live_memories(session, user_id):
    return list(
        session.exec(
            select(Memory).where(Memory.user_id == user_id, Memory.is_deleted == False)  # noqa: E712
        ).all()
    )


def test_chat_retrieves_memory_and_answers_from_it(db_session):
    user = _user()
    # seed an existing memory with a vector that matches a name query
    seeded = Memory(
        user_id=user, content="Name is Pooja",
        embedding=fake_embedding("Name is Pooja"), source_episode_ids=[],
    )
    db_session.add(seeded)
    db_session.flush()

    # reply, then extraction returns no new fact
    llm = FakeLLM(["Your name is Pooja!", json.dumps({"facts": []})])
    result = answer(
        db_session, llm, SETTINGS,
        user_id=user, conversation_id="c1", message="what is my name?",
    )

    assert "Pooja" in result.reply
    assert "Name is Pooja" in result.memories_used  # it actually retrieved the memory
    # the exchange was recorded as episodes (user question + assistant reply)
    episodes = list(db_session.exec(select(Episode).where(Episode.user_id == user)).all())
    assert len(episodes) == 2


def test_chat_records_a_new_fact_stated_mid_conversation(db_session):
    user = _user()
    # reply, extraction finds a fact, decision ADDs it
    llm = FakeLLM([
        "Nice to meet you, Alex!",
        json.dumps({"facts": ["Name is Alex"]}),
        json.dumps({"event": "ADD", "target_index": None, "text": "Name is Alex"}),
    ])
    result = answer(
        db_session, llm, SETTINGS,
        user_id=user, conversation_id="c1", message="Hi, I'm Alex",
    )
    db_session.flush()

    assert result.operations[0].event == "ADD"
    memories = _live_memories(db_session, user)
    assert any("Alex" in m.content for m in memories)  # the chat taught the memory