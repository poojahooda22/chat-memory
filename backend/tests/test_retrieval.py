"""Hybrid retrieval: the question is decomposed into filters, photos are filtered EXACTLY by
entity/time/place (all matches, not a top-k), and a filterless question falls back to cosine.
The decomposition LLM call is scripted by the FakeLLM; the filters run against real Postgres."""

import json
import uuid
from datetime import UTC, date, datetime

from app.config import get_settings
from app.memory.retrieval import decompose_query, recall, retrieve
from app.models import Entity, Episode, EpisodeEntity, Memory
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()


def _photo(db, user, when: datetime, content: str, entity: Entity | None = None) -> Episode:
    ep = Episode(
        user_id=user, occurred_at=when, content=content,
        context={"source": "image", "kind": "photo", "entities": []},
        embedding=fake_embedding(content),
    )
    db.add(ep)
    db.flush()
    if entity is not None:
        db.add(EpisodeEntity(episode_id=ep.id, entity_id=entity.id, entity_index=0))
        db.flush()
    return ep


def test_decompose_parses_entity_and_year():
    llm = FakeLLM([json.dumps({
        "entities": ["Monty"],
        "time_range": {"start": "2023-01-01", "end": "2023-12-31"},
        "place": None,
        "semantic_query": "photos of the dog",
        "wants_all": True,
    })])
    spec = decompose_query(llm, SETTINGS, "how many photos of Monty do I have from 2023?")
    assert spec.entities == ["Monty"]
    assert spec.time_range == (date(2023, 1, 1), date(2023, 12, 31))
    assert spec.wants_all is True
    assert spec.has_filters is True


def test_decompose_general_question_has_no_filters():
    llm = FakeLLM([json.dumps({
        "entities": [], "time_range": None, "place": None,
        "semantic_query": "how are you", "wants_all": False,
    })])
    spec = decompose_query(llm, SETTINGS, "how's it going?")
    assert spec.has_filters is False


def test_filter_by_entity_and_year_returns_all_matches(db_session):
    user = f"r-{uuid.uuid4()}"
    monty = Entity(user_id=user, name="Monty", type="pet", description="a dog",
                   embedding=fake_embedding("Monty"))
    db_session.add(monty)
    db_session.flush()

    # three 2023 Monty photos + one 2021 Monty photo + one 2023 photo with no entity
    for m in (2, 5, 8):
        _photo(db_session, user, datetime(2023, m, 10, tzinfo=UTC), f"Monty photo {m}", monty)
    _photo(db_session, user, datetime(2021, 6, 1, tzinfo=UTC), "old Monty photo", monty)
    _photo(db_session, user, datetime(2023, 7, 1, tzinfo=UTC), "a rooftop, no pet", None)

    # decompose -> {Monty, 2023, wants_all}; then facts search (extraction returns nothing)
    llm = FakeLLM([
        json.dumps({"entities": ["Monty"],
                    "time_range": {"start": "2023-01-01", "end": "2023-12-31"},
                    "place": None, "semantic_query": "Monty photos", "wants_all": True}),
    ])
    result = retrieve(db_session, llm, SETTINGS, user_id=user, message="all my Monty photos from 2023")

    # exactly the three 2023 Monty photos — not the 2021 one, not the no-entity one, not a top-k cap
    assert len(result.photos) == 3
    assert all("Monty photo" in p.content for p in result.photos)
    assert all(p.occurred_at.year == 2023 for p in result.photos)


def test_unknown_entity_returns_nothing_not_a_guess(db_session):
    user = f"r-{uuid.uuid4()}"
    _photo(db_session, user, datetime(2023, 1, 1, tzinfo=UTC), "some photo", None)
    llm = FakeLLM([json.dumps({
        "entities": ["Bruno"],  # never labeled
        "time_range": None, "place": None, "semantic_query": "Bruno", "wants_all": True,
    })])
    result = retrieve(db_session, llm, SETTINGS, user_id=user, message="photos of Bruno")
    assert result.photos == []  # honest empty, not a similar-looking guess


def test_filterless_question_falls_back_to_similarity(db_session):
    user = f"r-{uuid.uuid4()}"
    _photo(db_session, user, datetime(2023, 1, 1, tzinfo=UTC), "a fluffy dog on a blanket", None)
    # decompose -> no filters; then it falls back to search_image_episodes (cosine)
    llm = FakeLLM([json.dumps({
        "entities": [], "time_range": None, "place": None,
        "semantic_query": "a fluffy dog on a blanket", "wants_all": False,
    })])
    result = retrieve(db_session, llm, SETTINGS, user_id=user, message="tell me about a dog")
    assert result.spec.has_filters is False
    assert len(result.photos) == 1  # found by similarity, no filter path


def test_recall_bundle_carries_provenance_and_confidence(db_session):
    """recall() wraps each result with its receipts: a fact keeps source/confidence/provenance,
    a photo keeps its date + place, and the bundle carries one per-recall confidence (P2)."""
    user = f"r-{uuid.uuid4()}"
    mem = Memory(
        user_id=user, content="Loves cycling", embedding=fake_embedding("Loves cycling"),
        source_episode_ids=["ep-1"], source="chat", confidence=1.0,
    )
    db_session.add(mem)
    db_session.flush()
    _photo(db_session, user, datetime(2023, 5, 1, tzinfo=UTC), "a bike on a trail", None)

    llm = FakeLLM([json.dumps({
        "entities": [], "time_range": None, "place": None,
        "semantic_query": "cycling", "wants_all": False,
    })])
    bundle = recall(db_session, llm, SETTINGS, user_id=user, message="what are my hobbies?")

    assert bundle.fact_lines == ["Loves cycling"]
    fact = bundle.facts[0]
    assert fact.source == "chat" and fact.confidence == 1.0
    assert fact.source_episode_ids == ["ep-1"]  # provenance travels with the fact
    assert bundle.confidence == 1.0  # one confident fact
    assert len(bundle.photos) == 1
    assert bundle.photos[0].occurred_at == date(2023, 5, 1)
    assert bundle.photo_lines[0].startswith("[captured 2023-05-01]")


def test_recall_confidence_zero_when_no_facts(db_session):
    """No facts recalled -> the per-recall confidence is 0 (what the proactive gate thresholds on)."""
    user = f"r-{uuid.uuid4()}"
    llm = FakeLLM([json.dumps({
        "entities": [], "time_range": None, "place": None,
        "semantic_query": "x", "wants_all": False,
    })])
    bundle = recall(db_session, llm, SETTINGS, user_id=user, message="do you know anything?")
    assert bundle.facts == []
    assert bundle.confidence == 0.0