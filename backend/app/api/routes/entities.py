"""Entity labeling routes — thin wrappers over app.memory.entities (which owns the logic;
these own the HTTP shape and the commit)."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field as PydanticField
from sqlmodel import Session, col, select

from app.config import get_settings
from app.db import get_session
from app.memory.entities import LabelError, apply_label
from app.memory.graph import rebuild_edges_for_entity
from app.models import Entity, EpisodeEntity

router = APIRouter(tags=["entities"])


class LabelRequest(BaseModel):
    entity_index: int
    name: str = PydanticField(min_length=1, max_length=120)


class EntityRead(BaseModel):
    id: uuid.UUID
    name: str
    type: str
    description: str
    created_at: datetime


class LabelResponse(BaseModel):
    entity: EntityRead
    memory_event: str  # what the label did to semantic memory: ADD | UPDATE | NOOP | DELETE
    reused_existing: bool  # True when the name matched an entity that already existed


def _entity_out(e: Entity) -> EntityRead:
    return EntityRead(
        id=e.id, name=e.name, type=e.type, description=e.description, created_at=e.created_at
    )


@router.post(
    "/episodes/{episode_id}/label", operation_id="label_entity", response_model=LabelResponse
)
def label_entity(
    episode_id: uuid.UUID,
    req: LabelRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> LabelResponse:
    try:
        result = apply_label(
            session, request.app.state.llm, get_settings(),
            episode_id=episode_id, entity_index=req.entity_index, name=req.name,
        )
    except LabelError as exc:
        raise HTTPException(status_code=404 if "not found" in str(exc) else 422,
                            detail=str(exc)) from exc
    session.commit()
    return LabelResponse(
        entity=_entity_out(result.entity),
        memory_event=result.memory_event,
        reused_existing=result.reused_existing,
    )


@router.delete(
    "/episodes/{episode_id}/label/{entity_index}", operation_id="unlabel_entity"
)
def unlabel_entity(
    episode_id: uuid.UUID, entity_index: int, session: Session = Depends(get_session)
) -> dict[str, str]:
    """Detach a label from this photo (the undo for a wrong auto-recognition). The entity
    and any semantic memories stay — only the photo link is removed."""
    link = session.exec(
        select(EpisodeEntity).where(
            EpisodeEntity.episode_id == episode_id,
            EpisodeEntity.entity_index == entity_index,
        )
    ).first()
    if link is None:
        raise HTTPException(404, "No label on that slot")
    entity = session.get(Entity, link.entity_id)
    session.delete(link)
    session.flush()
    # the removed entity's edges lose this photo (or invalidate if it was their only shared one)
    if entity is not None:
        rebuild_edges_for_entity(session, user_id=entity.user_id, entity_id=entity.id)
    session.commit()
    return {"status": "unlabeled"}


@router.get("/entities", operation_id="list_entities", response_model=list[EntityRead])
def list_entities(
    user_id: str = "default", session: Session = Depends(get_session)
) -> list[EntityRead]:
    rows = session.exec(
        select(Entity)
        .where(Entity.user_id == user_id)
        .order_by(col(Entity.created_at).desc())
    ).all()
    return [_entity_out(e) for e in rows]