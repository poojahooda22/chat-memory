"""Entity labeling: naming a detected entity creates the entity, the episode link, and a
deduplicated semantic fact — without ever rewriting the episode (single-shot invariant)."""

import json
import uuid
from datetime import UTC, datetime

import pytest
from sqlmodel import select

from app.config import get_settings
from app.memory.entities import LabelError, apply_label
from app.models import Entity, Episode, EpisodeEntity, Memory
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()


def _photo_episode(db_session, user_id: str) -> Episode:
    episode = Episode(
        user_id=user_id,
        conversation_id=None,
        occurred_at=datetime(2023, 5, 22, 19, 40, tzinfo=UTC),
        content="A small, fluffy dog with a light brown coat is lying on a patterned blanket.",
        context={
            "source": "image",
            "kind": "photo",
            "entities": [
                {"type": "pet", "description": "a golden retriever", "species": "dog",
                 "label": None, "confidence": 0.9},
            ],
        },
        embedding=fake_embedding("a fluffy dog photo"),
    )
    db_session.add(episode)
    db_session.flush()
    return episode


def test_label_creates_entity_link_and_fact(db_session):
    user = f"test-{uuid.uuid4()}"
    episode = _photo_episode(db_session, user)
    original_context = dict(episode.context)

    llm = FakeLLM([json.dumps({"event": "ADD", "target_index": None,
                               "text": "Monty is the user's pet dog: a golden retriever"})])
    res = apply_label(
        db_session, llm, SETTINGS, episode_id=episode.id, entity_index=0, name="Monty"
    )

    assert res.entity.name == "Monty" and res.entity.type == "pet"
    assert res.memory_event == "ADD"
    assert res.reused_existing is False
    # the link exists; the episode itself was NOT rewritten (single-shot preserved)
    links = list(db_session.exec(
        select(EpisodeEntity).where(EpisodeEntity.episode_id == episode.id)).all())
    assert len(links) == 1 and links[0].entity_index == 0
    db_session.refresh(episode)
    assert episode.context == original_context
    # the label became a semantic memory with provenance to the photo episode
    mems = list(db_session.exec(select(Memory).where(Memory.user_id == user)).all())
    assert len(mems) == 1 and "Monty" in mems[0].content
    assert mems[0].source_episode_ids == [str(episode.id)]


def test_labeling_same_name_reuses_entity(db_session):
    user = f"test-{uuid.uuid4()}"
    first = _photo_episode(db_session, user)
    second = _photo_episode(db_session, user)

    llm = FakeLLM([
        json.dumps({"event": "ADD", "target_index": None,
                    "text": "Monty is the user's pet dog"}),
        json.dumps({"event": "NOOP", "target_index": None, "text": ""}),
    ])
    apply_label(db_session, llm, SETTINGS, episode_id=first.id, entity_index=0, name="Monty")
    res2 = apply_label(  # case-insensitive reuse
        db_session, llm, SETTINGS, episode_id=second.id, entity_index=0, name="monty"
    )

    assert res2.reused_existing is True
    assert res2.memory_event == "NOOP"  # the decision phase refused a duplicate fact
    entities = list(db_session.exec(select(Entity).where(Entity.user_id == user)).all())
    assert len(entities) == 1  # one Monty, two photos
    links = list(db_session.exec(
        select(EpisodeEntity).where(EpisodeEntity.entity_id == entities[0].id)).all())
    assert len(links) == 2  # ...linked to both episodes: the graph's co-occurrence substrate


def test_label_rejects_bad_slot(db_session):
    user = f"test-{uuid.uuid4()}"
    episode = _photo_episode(db_session, user)
    with pytest.raises(LabelError):
        apply_label(db_session, FakeLLM([]), SETTINGS,
                    episode_id=episode.id, entity_index=5, name="Monty")