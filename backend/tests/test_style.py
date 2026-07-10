"""Style inference: traits read off the user's own messages, landed through the pipeline.

Each test seeds episodes directly (the sample the inference reads), scripts the fake LLM's
replies (the trait read, then one decision per trait), and asserts the resulting rows — the
same zero-token pattern as the pipeline tests.
"""

import json
import uuid

from sqlmodel import select

from app.config import get_settings
from app.memory.style import MIN_MESSAGES, infer_style
from app.models import Episode, Memory
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()


def _user() -> str:
    return f"test-{uuid.uuid4()}"


def _traits(*traits: str) -> str:
    return json.dumps({"traits": list(traits)})


def _decide(event: str, target_index=None, text: str = "") -> str:
    return json.dumps({"event": event, "target_index": target_index, "text": text})


def _seed_user_messages(session, user_id: str, n: int) -> list[Episode]:
    episodes = []
    for i in range(n):
        ep = Episode(
            user_id=user_id,
            conversation_id="c1",
            content=f"user message number {i}",
            context={"role": "user", "source": "chat"},
            embedding=fake_embedding(f"user message number {i}"),
        )
        session.add(ep)
        episodes.append(ep)
    session.flush()
    return episodes


def _live_memories(session, user_id) -> list[Memory]:
    return list(
        session.exec(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.is_deleted == False,  # noqa: E712
                Memory.is_superseded == False,  # noqa: E712
            )
        ).all()
    )


def test_infer_style_writes_inferred_traits_with_receipts(db_session):
    """Traits land as facts: source='inferred', mid confidence, provenance = the sampled messages."""
    user = _user()
    episodes = _seed_user_messages(db_session, user, MIN_MESSAGES)
    llm = FakeLLM([
        _traits(
            "Communication style: writes short, direct messages",
            "Communication style: asks concrete questions",
        ),
        _decide("ADD", None, "Communication style: writes short, direct messages"),
        _decide("ADD", None, "Communication style: asks concrete questions"),
    ])

    ops = infer_style(db_session, llm, SETTINGS, user_id=user)
    db_session.flush()

    assert [o.event for o in ops] == ["ADD", "ADD"]
    memories = _live_memories(db_session, user)
    assert len(memories) == 2
    for m in memories:
        assert m.source == "inferred"
        assert m.confidence == 0.7
        # the receipts are the real messages the trait was read from
        assert set(m.source_episode_ids) == {str(e.id) for e in episodes}


def test_infer_style_skips_when_too_few_messages(db_session):
    """Below the floor there is no style to read — no LLM call, no facts."""
    user = _user()
    _seed_user_messages(db_session, user, MIN_MESSAGES - 1)
    # an empty script: any LLM call would pop from an empty list and fail loudly
    assert infer_style(db_session, FakeLLM([]), SETTINGS, user_id=user) == []
    assert _live_memories(db_session, user) == []


def test_infer_style_reads_only_the_users_chat_messages(db_session):
    """Assistant replies and photo episodes never count toward (or feed) the sample."""
    user = _user()
    for i in range(MIN_MESSAGES):
        db_session.add(Episode(
            user_id=user, conversation_id="c1", content=f"assistant reply {i}",
            context={"role": "assistant", "source": "chat"},
            embedding=fake_embedding(f"assistant reply {i}"),
        ))
    db_session.add(Episode(
        user_id=user, conversation_id=None, content="A photo from my life shows a park",
        context={"source": "image"},
        embedding=fake_embedding("photo park"),
    ))
    db_session.flush()

    # only non-user-authored episodes exist -> the sample is empty -> skip, no LLM call
    assert infer_style(db_session, FakeLLM([]), SETTINGS, user_id=user) == []


def test_reinference_supersedes_previous_trait(db_session):
    """A second inference lands as UPDATE: the old trait is retired, the successor is live."""
    user = _user()
    episodes = _seed_user_messages(db_session, user, MIN_MESSAGES)

    # first inference: one trait, ADDed
    infer_style(
        db_session,
        FakeLLM([
            _traits("Communication style: writes long, detailed messages"),
            _decide("ADD", None, "Communication style: writes long, detailed messages"),
        ]),
        SETTINGS, user_id=user,
    )
    db_session.flush()
    first = _live_memories(db_session, user)[0]

    # second inference: the style shifted -> UPDATE (same concept slot -> search finds it)
    infer_style(
        db_session,
        FakeLLM([
            _traits("Communication style: writes short, direct messages"),
            _decide("UPDATE", 0, "Communication style: writes short, direct messages"),
        ]),
        SETTINGS, user_id=user,
    )
    db_session.flush()

    live = _live_memories(db_session, user)
    assert len(live) == 1
    assert live[0].content == "Communication style: writes short, direct messages"
    assert live[0].id != first.id

    db_session.refresh(first)
    assert first.is_superseded is True
    assert first.superseded_by_id == live[0].id
    # the successor still carries the message receipts
    assert {str(e.id) for e in episodes} <= set(live[0].source_episode_ids)