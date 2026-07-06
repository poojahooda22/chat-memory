"""The relationship graph: edges form when two entities share a photo, strengthen with more
shared photos, invalidate when a link is removed, and rank neighbours by weight. No LLM — the
graph is pure computation over the episode_entities links."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import select

from app.memory.graph import (
    neighbors,
    rebuild_edges_for_entity,
    rebuild_edges_for_episode,
)
from app.models import Entity, Episode, EpisodeEntity, Relationship


def _entity(db, user, name, type="pet") -> Entity:
    e = Entity(user_id=user, name=name, type=type, description="")
    db.add(e)
    db.flush()
    return e


def _photo(db, user, when: datetime) -> Episode:
    ep = Episode(
        user_id=user, occurred_at=when, content="a photo",
        context={"source": "image", "entities": []}, embedding=None,
    )
    db.add(ep)
    db.flush()
    return ep


def _link(db, episode, entity, index=0, by="user") -> None:
    db.add(EpisodeEntity(
        episode_id=episode.id, entity_id=entity.id, entity_index=index, labeled_by=by
    ))
    db.flush()


def _edges(db, user):
    return list(db.exec(select(Relationship).where(Relationship.user_id == user)).all())


def test_single_entity_makes_no_edge(db_session):
    user = f"g-{uuid4()}"
    monty = _entity(db_session, user, "Monty")
    ep = _photo(db_session, user, datetime.now(UTC))
    _link(db_session, ep, monty, 0)
    rebuild_edges_for_episode(db_session, episode_id=ep.id)
    assert _edges(db_session, user) == []  # a line needs two dots


def test_edge_forms_when_two_entities_share_a_photo(db_session):
    user = f"g-{uuid4()}"
    monty = _entity(db_session, user, "Monty", "pet")
    pooja = _entity(db_session, user, "Pooja", "person")
    ep = _photo(db_session, user, datetime.now(UTC))
    _link(db_session, ep, monty, 0)
    _link(db_session, ep, pooja, 1)
    rebuild_edges_for_episode(db_session, episode_id=ep.id)

    edges = _edges(db_session, user)
    assert len(edges) == 1  # ONE row for the pair, canonical order
    edge = edges[0]
    assert edge.cooccur_count == 1
    assert 0.0 < edge.weight <= 1.0
    assert {edge.src_entity_id, edge.dst_entity_id} == {monty.id, pooja.id}
    assert edge.is_valid is True
    assert str(ep.id) in edge.source_episode_ids  # receipts


def test_more_shared_photos_strengthen_the_edge(db_session):
    user = f"g-{uuid4()}"
    monty = _entity(db_session, user, "Monty", "pet")
    pooja = _entity(db_session, user, "Pooja", "person")

    now = datetime.now(UTC)
    ep1 = _photo(db_session, user, now)
    _link(db_session, ep1, monty, 0)
    _link(db_session, ep1, pooja, 1)
    rebuild_edges_for_episode(db_session, episode_id=ep1.id)
    weight_one = _edges(db_session, user)[0].weight

    for _ in range(2):
        ep = _photo(db_session, user, now)
        _link(db_session, ep, monty, 0)
        _link(db_session, ep, pooja, 1)
        rebuild_edges_for_episode(db_session, episode_id=ep.id)

    edge = _edges(db_session, user)[0]
    assert edge.cooccur_count == 3
    assert edge.weight > weight_one  # more shared photos → stronger bond


def test_removing_a_link_invalidates_the_edge(db_session):
    user = f"g-{uuid4()}"
    monty = _entity(db_session, user, "Monty", "pet")
    pooja = _entity(db_session, user, "Pooja", "person")
    ep = _photo(db_session, user, datetime.now(UTC))
    _link(db_session, ep, monty, 0)
    _link(db_session, ep, pooja, 1)
    rebuild_edges_for_episode(db_session, episode_id=ep.id)
    assert _edges(db_session, user)[0].is_valid is True

    # unlabel Pooja: delete her link, then recompute her edges
    link = db_session.exec(select(EpisodeEntity).where(
        EpisodeEntity.entity_id == pooja.id, EpisodeEntity.episode_id == ep.id)).first()
    db_session.delete(link)
    db_session.flush()
    rebuild_edges_for_entity(db_session, user_id=user, entity_id=pooja.id)

    edge = _edges(db_session, user)[0]
    assert edge.is_valid is False  # no shared photos left → invalidated, not deleted


def test_neighbors_ranked_by_weight(db_session):
    user = f"g-{uuid4()}"
    monty = _entity(db_session, user, "Monty", "pet")
    pooja = _entity(db_session, user, "Pooja", "person")
    akshay = _entity(db_session, user, "Akshay", "person")
    now = datetime.now(UTC)

    # Monty + Pooja share 3 photos (strong); Monty + Akshay share 1 (weak)
    for _ in range(3):
        ep = _photo(db_session, user, now)
        _link(db_session, ep, monty, 0)
        _link(db_session, ep, pooja, 1)
        rebuild_edges_for_episode(db_session, episode_id=ep.id)
    ep = _photo(db_session, user, now)
    _link(db_session, ep, monty, 0)
    _link(db_session, ep, akshay, 1)
    rebuild_edges_for_episode(db_session, episode_id=ep.id)

    ranked = neighbors(db_session, user_id=user, entity_id=monty.id)
    assert [e.name for e, _ in ranked] == ["Pooja", "Akshay"]  # strongest first
    assert ranked[0][1].weight > ranked[1][1].weight