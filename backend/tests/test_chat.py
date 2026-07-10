"""Phase 2: the remembering chat READS memory to answer.

The memory WRITE now runs off the request path (learn_from_exchange -> record_exchange), so
answer() only retrieves + generates. Zero tokens — the fake LLM scripts each reply.
"""

import json
import uuid

from sqlmodel import select

from app.config import get_settings
from app.memory.chat import answer
from app.memory.pipeline import record_exchange
from app.models import Episode, Memory
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()

# answer() now decomposes the question first (hybrid retrieval); a general question yields no
# filters and falls back to similarity, so every chat script starts with this decompose reply.
NO_FILTER = json.dumps({
    "entities": [], "time_range": None, "place": None, "semantic_query": "", "wants_all": False,
})


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

    # answer() now only decomposes (no filters) + replies — the write is off the request path
    llm = FakeLLM([NO_FILTER, "Your name is Pooja!"])
    result = answer(
        db_session, llm, SETTINGS,
        user_id=user, conversation_id="c1", message="what is my name?",
    )

    assert "Pooja" in result.reply
    assert "Name is Pooja" in result.memories_used  # it actually retrieved the memory
    # the request path writes nothing now — answer() creates no episodes
    episodes = list(db_session.exec(select(Episode).where(Episode.user_id == user)).all())
    assert len(episodes) == 0


def test_chat_retrieves_photo_episodes(db_session):
    """A photo fed via Sources is retrievable in chat — the episodic layer reaches the prompt."""
    from datetime import UTC, datetime

    user = _user()
    photo = Episode(
        user_id=user,
        conversation_id=None,
        occurred_at=datetime(2023, 5, 22, 19, 40, tzinfo=UTC),
        content="A small, fluffy dog with a light brown coat is lying on a patterned blanket.",
        context={"source": "image", "kind": "photo", "entities": []},
        embedding=fake_embedding("A small, fluffy dog with a light brown coat"),
    )
    db_session.add(photo)
    db_session.flush()

    llm = FakeLLM([NO_FILTER, "Your photos show your fluffy dog!"])
    result = answer(
        db_session, llm, SETTINGS,
        user_id=user, conversation_id="c1", message="what do my photos show about my dog?",
    )

    assert len(result.photos_used) == 1
    assert result.photos_used[0].startswith("[captured 2023-05-22]")
    assert "fluffy dog" in result.photos_used[0]


def test_background_write_teaches_memory(db_session):
    """The write moved off the request path: the background task (learn_from_exchange) runs
    record_exchange. We run that write directly here — in-session, so it stays isolated — to
    confirm a chat exchange still teaches the memory the same way it always did."""
    user = _user()
    # extraction over the exchange finds one fact, the decision ADDs it (no decompose call here —
    # record_exchange is the write path, not the read path)
    llm = FakeLLM([
        json.dumps({"facts": ["Name is Alex"]}),
        json.dumps({"event": "ADD", "target_index": None, "text": "Name is Alex"}),
    ])
    record_exchange(
        db_session, llm, SETTINGS,
        user_id=user, conversation_id="c1",
        messages=[
            {"role": "user", "content": "Hi, I'm Alex"},
            {"role": "assistant", "content": "Nice to meet you, Alex!"},
        ],
    )
    db_session.flush()

    memories = _live_memories(db_session, user)
    assert any("Alex" in m.content for m in memories)  # the exchange taught the memory