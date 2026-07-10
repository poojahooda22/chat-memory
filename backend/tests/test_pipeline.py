"""The Mem0 pipeline behaviour: ADD, UPDATE, DELETE, NOOP, and the audit trail.

Each test scripts the fake LLM's replies (extraction, then one decision per fact) and asserts
the resulting rows. Zero tokens spent — the fake stands in for the gateway.
"""

import json
import uuid

from sqlmodel import col, select

from app.config import get_settings
from app.memory.lineage import _belief_chain_ids
from app.memory.pipeline import distil_text, record_exchange, search_memories
from app.models import Episode, Memory, MemoryHistory
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()


def _user() -> str:
    return f"test-{uuid.uuid4()}"


def _extract(*facts: str) -> str:
    return json.dumps({"facts": list(facts)})


def _decide(event: str, target_index=None, text: str = "") -> str:
    return json.dumps({"event": event, "target_index": target_index, "text": text})


def _live_memories(session, user_id) -> list[Memory]:
    """Live = neither deleted nor superseded — what search and the Memory page surface."""
    return list(
        session.exec(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.is_deleted == False,  # noqa: E712
                Memory.is_superseded == False,  # noqa: E712
            )
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


def test_update_supersedes_existing_memory(db_session):
    """A correction retires the old belief (kept as provenance) and writes a successor row."""
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

    # exactly one LIVE memory — the successor, not the original row
    memories = _live_memories(db_session, user)
    assert len(memories) == 1
    successor = memories[0]
    assert successor.id != before.id
    assert successor.content == "Works as a backend developer"

    # the old belief is retired in place: content intact, pointing at its successor
    db_session.refresh(before)
    assert before.is_superseded is True
    assert before.superseded_by_id == successor.id
    assert before.content == "Works as a frontend developer"

    # the successor carries the whole evidence trail (old episodes + the correcting ones)
    assert set(before.source_episode_ids) <= set(successor.source_episode_ids)
    assert len(successor.source_episode_ids) > len(before.source_episode_ids)

    # audit: the transition on the retired row, a birth on the successor
    assert [h.event for h in _history(db_session, before.id)] == ["ADD", "UPDATE"]
    upd = next(h for h in _history(db_session, before.id) if h.event == "UPDATE")
    assert upd.old_content == "Works as a frontend developer"
    assert upd.new_content == "Works as a backend developer"
    assert [h.event for h in _history(db_session, successor.id)] == ["ADD"]

    # lineage survives on the LIVE row: the chain walk gathers successor + its predecessor, so the
    # merged history is the full ADD -> UPDATE -> ADD story, not the successor's bare ['ADD']
    chain = _belief_chain_ids(db_session, successor)
    assert set(chain) == {before.id, successor.id}
    merged = list(
        db_session.exec(
            select(MemoryHistory)
            .where(col(MemoryHistory.memory_id).in_(chain))
            .order_by(col(MemoryHistory.created_at))
        ).all()
    )
    assert [h.event for h in merged] == ["ADD", "UPDATE", "ADD"]


def test_lower_trust_cannot_supersede_higher_trust(db_session):
    """The CRITICAL guard: a 0.7 inference may NOT retire a 1.0 directly-stated fact.

    Both share a concept slot (the "communication style" keyword), so the decision phase sees the
    stated fact as the top similar memory and asks to UPDATE it — the trust guard must refuse."""
    user = _user()
    # a directly-stated style preference at full confidence
    stated = Memory(
        user_id=user, content="Communication style: the user asked for concise answers",
        embedding=fake_embedding("communication style"), source_episode_ids=[],
        source="chat", confidence=1.0,
    )
    db_session.add(stated)
    db_session.flush()

    # a lower-confidence inference lands in the same slot and the decision asks to UPDATE it
    ops = distil_text(
        db_session,
        FakeLLM([
            _extract("Communication style: the user writes long, detailed messages"),
            _decide("UPDATE", 0, "Communication style: the user writes long, detailed messages"),
        ]),
        SETTINGS, user_id=user, text="(a style inference)",
        source_ids=[], source="inferred", confidence=0.7,
    )
    db_session.flush()

    # the inference was refused: NOOP, no successor, the stated fact untouched and still live
    assert [o.event for o in ops] == ["NOOP"]
    live = _live_memories(db_session, user)
    assert len(live) == 1
    assert live[0].id == stated.id
    db_session.refresh(stated)
    assert stated.is_superseded is False
    assert stated.confidence == 1.0
    assert stated.content == "Communication style: the user asked for concise answers"


def test_higher_trust_supersedes_lower_trust_seed(db_session):
    """The mirror: a 1.0 stated fact DOES supersede a 0.6 quiz seed — the seed was a guess."""
    user = _user()
    seed = Memory(
        user_id=user, content="Works as a developer",
        embedding=fake_embedding("developer"), source_episode_ids=[],
        source="quiz", confidence=0.6,
    )
    db_session.add(seed)
    db_session.flush()

    record_exchange(
        db_session,
        FakeLLM([_extract("Works as a backend developer"),
                 _decide("UPDATE", 0, "Works as a backend developer")]),
        SETTINGS, user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "I'm a backend developer"}],
    )
    db_session.flush()

    live = _live_memories(db_session, user)
    assert len(live) == 1
    assert live[0].content == "Works as a backend developer"
    assert live[0].confidence == 1.0
    db_session.refresh(seed)
    assert seed.is_superseded is True
    assert seed.superseded_by_id == live[0].id


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


def test_add_stamps_default_trust(db_session):
    """A chat-observed fact lands with source='chat' and full confidence (the P1 trust model)."""
    user = _user()
    record_exchange(
        db_session, FakeLLM([_extract("Loves hiking"), _decide("ADD", None, "Loves hiking")]),
        SETTINGS, user_id=user, conversation_id="c1",
        messages=[{"role": "user", "content": "I love hiking"}],
    )
    db_session.flush()

    mem = _live_memories(db_session, user)[0]
    assert mem.source == "chat"
    assert mem.confidence == 1.0
    assert mem.is_superseded is False


def test_distil_text_carries_callers_source(db_session):
    """A non-chat writer (image ingest, a future import) stamps its own origin + confidence."""
    user = _user()
    distil_text(
        db_session, FakeLLM([_extract("Has a dog"), _decide("ADD", None, "Has a dog")]),
        SETTINGS, user_id=user, text="A photo from my life shows a dog",
        source_ids=[], source="photo", confidence=0.8,
    )
    db_session.flush()

    mem = _live_memories(db_session, user)[0]
    assert mem.source == "photo"
    assert mem.confidence == 0.8


def test_superseded_memory_is_excluded_from_search(db_session):
    """A retired (superseded) fact is never retrieved — the read-side of the trust model."""
    user = _user()
    live = Memory(
        user_id=user, content="Works as a developer",
        embedding=fake_embedding("developer"), source_episode_ids=[],
    )
    retired = Memory(
        user_id=user, content="Old guess: works a developer job",
        embedding=fake_embedding("developer job"), source_episode_ids=[], is_superseded=True,
    )
    db_session.add(live)
    db_session.add(retired)
    db_session.flush()

    # both embed to the same concept slot, so only the is_superseded filter separates them
    got = search_memories(db_session, FakeLLM([]), SETTINGS, user_id=user, query="developer", limit=10)
    contents = [m.content for m in got]
    assert "Works as a developer" in contents
    assert all("Old guess" not in c for c in contents)