"""The relationship graph as JSON: nodes (entities) + weighted edges, for the Moments canvas
and any graph-aware query. Read-only; the graph is built on the label path (memory/graph.py)."""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, col, func, select

from app.db import get_session
from app.memory.graph import MIN_CONFIDENT_COOCCUR, neighbors
from app.models import Entity, EpisodeEntity, IngestJob, Relationship

router = APIRouter(tags=["graph"])


class GraphNode(BaseModel):
    id: uuid.UUID
    name: str
    type: str
    photo_count: int
    representative_job_id: uuid.UUID | None  # a photo to show inside the node bubble
    photo_job_ids: list[uuid.UUID] = []  # every photo this entity is in (membership spokes)


class GraphEdge(BaseModel):
    src: uuid.UUID
    dst: uuid.UUID
    weight: float
    cooccur_count: int
    is_learning: bool  # < MIN_CONFIDENT_COOCCUR shared photos → draw faint


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@router.get("/graph", operation_id="get_graph", response_model=GraphResponse)
def get_graph(user_id: str = "default", session: Session = Depends(get_session)) -> GraphResponse:
    entities = list(session.exec(select(Entity).where(Entity.user_id == user_id)).all())
    if not entities:
        return GraphResponse(nodes=[], edges=[])

    # photo_count per entity = distinct episodes it's linked to
    counts = dict(
        session.exec(
            select(EpisodeEntity.entity_id, func.count(col(EpisodeEntity.episode_id).distinct()))
            .group_by(col(EpisodeEntity.entity_id))
        ).all()
    )
    # every photo per entity (newest first): the first is the node's representative thumbnail,
    # all of them are the membership spokes the canvas draws from photo to entity
    photos: dict[uuid.UUID, list[uuid.UUID]] = {}
    rows = session.exec(
        select(EpisodeEntity.entity_id, IngestJob.id, IngestJob.created_at)
        .join(IngestJob, col(IngestJob.episode_id) == col(EpisodeEntity.episode_id))
        .order_by(col(IngestJob.created_at).desc())
    ).all()
    for entity_id, job_id, _created in rows:
        photos.setdefault(entity_id, []).append(job_id)

    nodes = [
        GraphNode(
            id=e.id, name=e.name, type=e.type,
            photo_count=int(counts.get(e.id, 0)),
            representative_job_id=photos.get(e.id, [None])[0],
            photo_job_ids=photos.get(e.id, []),
        )
        for e in entities
        if counts.get(e.id, 0) > 0  # only entities that actually appear in a photo
    ]

    edges = [
        GraphEdge(
            src=r.src_entity_id, dst=r.dst_entity_id, weight=r.weight,
            cooccur_count=r.cooccur_count, is_learning=r.cooccur_count < MIN_CONFIDENT_COOCCUR,
        )
        for r in session.exec(
            select(Relationship).where(
                Relationship.user_id == user_id, Relationship.is_valid == True  # noqa: E712
            )
        ).all()
    ]
    return GraphResponse(nodes=nodes, edges=edges)


class NeighbourRead(BaseModel):
    id: uuid.UUID
    name: str
    type: str
    weight: float
    cooccur_count: int


@router.get(
    "/entities/{entity_id}/neighbors",
    operation_id="entity_neighbors",
    response_model=list[NeighbourRead],
)
def entity_neighbors(
    entity_id: uuid.UUID, user_id: str = "default", session: Session = Depends(get_session)
) -> list[NeighbourRead]:
    return [
        NeighbourRead(
            id=other.id, name=other.name, type=other.type,
            weight=edge.weight, cooccur_count=edge.cooccur_count,
        )
        for other, edge in neighbors(session, user_id=user_id, entity_id=entity_id)
    ]