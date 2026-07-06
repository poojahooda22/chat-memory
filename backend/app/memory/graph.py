"""The relationship graph: weighted co-occurrence edges between entities.

Mem0g's labeled edges (arXiv 2504.19413 §2.2), plus our weight. Nodes are entities; an edge
means two entities share photos. Edges are REBUILT from the episode_entities links (the source
of truth), so labeling and unlabeling stay consistent — never incrementally drifted.

    weight = cooccur_score × recency × confidence          (each in [0,1])
      cooccur_score = n / (n + K)          more shared photos → stronger, never quite 1
      recency       = 0.5 + 0.5·e^(−age/H)  bounded so an old life-bond doesn't vanish
      confidence    = mean label confidence (a hand-confirmed link counts full, auto counts less)

Below MIN_CONFIDENT_COOCCUR shared photos the edge is flagged "still learning" (the UI draws it
faint) — one shared photo is not yet a relationship (the project's never-overclaim rule).
"""

from datetime import UTC, datetime
from math import exp
from uuid import UUID

from sqlmodel import Session, col, select

from app.models import Entity, Episode, EpisodeEntity, Relationship

K = 2.0  # co-occurrence smoothing
RECENCY_HALFLIFE_DAYS = 365.0  # gentle: a year-old bond keeps ~0.68 of its recency
MIN_CONFIDENT_COOCCUR = 3  # below this, the edge is "still learning"
_CONFIDENCE = {"user": 1.0, "memory": 0.75}  # a confirmed label outweighs an auto one


def _canonical(a: UUID, b: UUID) -> tuple[UUID, UUID]:
    """Order a pair so Monty–Pooja and Pooja–Monty are the SAME edge row."""
    return (a, b) if str(a) < str(b) else (b, a)


def _aware(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(UTC)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def compute_weight(count: int, last_seen: datetime | None, mean_confidence: float) -> float:
    cooccur = count / (count + K)
    age_days = max(0.0, (datetime.now(UTC) - _aware(last_seen)).total_seconds() / 86400.0)
    recency = 0.5 + 0.5 * exp(-age_days / RECENCY_HALFLIFE_DAYS)
    return round(cooccur * recency * mean_confidence, 4)


def rebuild_edges_for_episode(session: Session, *, episode_id: UUID) -> None:
    """After a label lands on a photo, (re)compute the edges among that photo's entities.

    Fires from apply_label — so the moment a photo has a SECOND named entity, the pair's edge
    forms or strengthens. One entity on a photo makes no edge (a line needs two dots).
    """
    episode = session.get(Episode, episode_id)
    if episode is None:
        return
    entity_ids = list({
        link.entity_id
        for link in session.exec(
            select(EpisodeEntity).where(EpisodeEntity.episode_id == episode_id)
        ).all()
    })
    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            src, dst = _canonical(entity_ids[i], entity_ids[j])
            _recompute_edge(session, user_id=episode.user_id, src=src, dst=dst)


def rebuild_edges_for_entity(session: Session, *, user_id: str, entity_id: UUID) -> None:
    """Recompute every edge touching this entity — used after an unlabel, where the entity has
    left an episode and its edges must lose that shared photo (or invalidate if none remain)."""
    others: set[UUID] = set()
    for episode_id in _links_by_episode(session, entity_id):
        for link in session.exec(
            select(EpisodeEntity).where(EpisodeEntity.episode_id == episode_id)
        ).all():
            if link.entity_id != entity_id:
                others.add(link.entity_id)
    # also any entity we currently have an edge with (catches the pair that just lost its photo)
    for r in session.exec(
        select(Relationship).where(
            (Relationship.src_entity_id == entity_id) | (Relationship.dst_entity_id == entity_id)
        )
    ).all():
        others.add(r.dst_entity_id if r.src_entity_id == entity_id else r.src_entity_id)
    for other in others:
        src, dst = _canonical(entity_id, other)
        _recompute_edge(session, user_id=user_id, src=src, dst=dst)


def _links_by_episode(session: Session, entity_id: UUID) -> dict[UUID, EpisodeEntity]:
    return {
        link.episode_id: link
        for link in session.exec(
            select(EpisodeEntity).where(EpisodeEntity.entity_id == entity_id)
        ).all()
    }


def _recompute_edge(session: Session, *, user_id: str, src: UUID, dst: UUID) -> None:
    src_links = _links_by_episode(session, src)
    dst_links = _links_by_episode(session, dst)
    shared = set(src_links) & set(dst_links)

    edge = session.exec(
        select(Relationship).where(
            Relationship.src_entity_id == src,
            Relationship.dst_entity_id == dst,
            Relationship.rel_type == "co_occurs_with",
        )
    ).first()

    if not shared:
        # the pair lost all shared photos (e.g. an unlabel) — invalidate, never delete
        if edge is not None:
            edge.is_valid = False
            edge.updated_at = datetime.now(UTC)
            session.add(edge)
        return

    episodes = {
        e.id: e
        for e in session.exec(select(Episode).where(col(Episode.id).in_(list(shared)))).all()
    }
    count = len(shared)
    last_seen = max(episodes[eid].occurred_at for eid in shared)
    confidences = [
        (_CONFIDENCE.get(src_links[eid].labeled_by, 0.75)
         + _CONFIDENCE.get(dst_links[eid].labeled_by, 0.75)) / 2
        for eid in shared
    ]
    mean_confidence = sum(confidences) / len(confidences)

    if edge is None:
        edge = Relationship(
            user_id=user_id, src_entity_id=src, dst_entity_id=dst, rel_type="co_occurs_with"
        )
    edge.weight = compute_weight(count, last_seen, mean_confidence)
    edge.cooccur_count = count
    edge.last_seen_at = last_seen
    edge.mean_confidence = round(mean_confidence, 4)
    edge.is_valid = True
    edge.source_episode_ids = [str(eid) for eid in shared]
    edge.updated_at = datetime.now(UTC)
    session.add(edge)


def neighbors(
    session: Session, *, user_id: str, entity_id: UUID, limit: int = 10
) -> list[tuple[Entity, Relationship]]:
    """Who is this entity connected to, strongest first — the graph query.

    Returns (other_entity, edge) so the caller sees the neighbour and the bond's strength.
    """
    edges = session.exec(
        select(Relationship)
        .where(
            Relationship.user_id == user_id,
            Relationship.is_valid == True,  # noqa: E712
            (Relationship.src_entity_id == entity_id)
            | (Relationship.dst_entity_id == entity_id),
        )
        .order_by(col(Relationship.weight).desc())
        .limit(limit)
    ).all()
    out: list[tuple[Entity, Relationship]] = []
    for edge in edges:
        other_id = edge.dst_entity_id if edge.src_entity_id == entity_id else edge.src_entity_id
        other = session.get(Entity, other_id)
        if other is not None:
            out.append((other, edge))
    return out