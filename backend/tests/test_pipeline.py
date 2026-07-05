"""The Mem0 pipeline behaviour: ADD, UPDATE, DELETE, NOOP, and the audit trail.

Each test scripts the fake LLM's replies (extraction, then one decision per fact) and asserts
the resulting rows. Zero tokens spent — the fake stands in for the gateway.
"""

import json
import uuid

from sqlmodel import select

from app.config import get_settings
from app.memory.pipeline import record_exchange
from app.models import Episode, Memory, MemoryHistory
from tests.conftest import FakeLLM

SETTINGS = get_settings()


def _user() -> str:
    return f"test-{uuid.uuid4()}"


def _extract(*facts: str) -> str:
    return json.dumps({"facts": list(facts)})


def _decide(event: str, target_index=None, text: str = "") -> str:
    return json.dumps({"event": event, "target_index": target_index, "text": text})


def _live_memories(session, user_id) -> list[Memory]:
    return list(
        session.exec(
            select(Memory).where(Memory.user_id == user_id, Memory.is_deleted == False)  # noqa: E712
        ).all()
    )


def _history(session, memory_id) -> list[MemoryHistory]:
    return list(
        session.exec(select(MemoryHistory).where(MemoryHistory.memory_id == memory_id)).all()
    )


def test_add_creates_memory_and_episode_and_history(db_session):
    user = _user()
    llm = FakeLLM([
        _extract("Name is Pooja"),      # extraction
        _decide("ADD", None, "Name is Pooja"),  # decision for the one fact
    ])
    result = record_exchange(
        db_session, llm, SETTINGS,
        user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "Hi, I'm Pooja"}],
    )
    db_session.flush()

    memories = _live_memories(db_session, user)
    assert len(memories) == 1
    assert memories[0].content == "Name is Pooja"
    # episode written for the raw message
    episodes = list(db_session.exec(select(Episode).where(Episode.user_id == user)).all())
    assert len(episodes) == 1
    # provenance: the fact links back to its source episode
    assert str(episodes[0].id) in memories[0].source_episode_ids
    # audit trail records the ADD
    hist = _history(db_session, memories[0].id)
    assert [h.event for h in hist] == ["ADD"]
    assert result.operations[0].event == "ADD"


def test_update_corrects_existing_memory(db_session):
    user = _user()
    # first exchange: ADD "Works as a frontend developer"
    record_exchange(
        db_session, FakeLLM([_extract("Works as a frontend developer"),
                             _decide("ADD", None, "Works as a frontend developer")]),
        SETTINGS, user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "I'm a frontend developer"}],
    )
    db_session.flush()
    before = _live_memories(db_session, user)[0]

    # second exchange: the job fact changes -> UPDATE (same concept slot -> search finds it)
    record_exchange(
        db_session, FakeLLM([_extract("Works as a backend developer"),
                             _decide("UPDATE", 0, "Works as a backend developer")]),
        SETTINGS, user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "I moved to backend"}],
    )
    db_session.flush()

    memories = _live_memories(db_session, user)
    assert len(memories) == 1  # updated in place, not duplicated
    assert memories[0].id == before.id
    assert memories[0].content == "Works as a backend developer"
    events = [h.event for h in _history(db_session, before.id)]
    assert events == ["ADD", "UPDATE"]
    upd = next(h for h in _history(db_session, before.id) if h.event == "UPDATE")
    assert upd.old_content == "Works as a frontend developer"
    assert upd.new_content == "Works as a backend developer"


def test_delete_marks_memory_obsolete(db_session):
    user = _user()
    record_exchange(
        db_session, FakeLLM([_extract("Loves working with Python"),
                             _decide("ADD", None, "Loves working with Python")]),
        SETTINGS, user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "I love Python"}],
    )
    db_session.flush()
    mem_id = _live_memories(db_session, user)[0].id

    record_exchange(
        db_session, FakeLLM([_extract("No longer uses Python"),
                             _decide("DELETE", 0, "")]),
        SETTINGS, user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "I stopped using Python"}],
    )
    db_session.flush()

    assert _live_memories(db_session, user) == []  # no live memories remain
    events = [h.event for h in _history(db_session, mem_id)]
    assert events == ["ADD", "DELETE"]


def test_noop_when_nothing_durable(db_session):
    user = _user()
    result = record_exchange(
        db_session, FakeLLM([_extract()]),  # extraction returns no facts -> no decision call
        SETTINGS, user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "what's the weather?"}],
    )
    db_session.flush()
    assert _live_memories(db_session, user) == []
    assert result.operations == []
    # the episode is still recorded (the event happened, even if no fact came of it)
    episodes = list(db_session.exec(select(Episode).where(Episode.user_id == user)).all())
    assert len(episodes) == 1