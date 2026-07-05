import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db import get_session
from app.models import Memory

router = APIRouter(tags=["memories"])


class MemoryRead(BaseModel):
    id: uuid.UUID
    user_id: str
    content: str
    source_episode_ids: list[str]
    created_at: datetime
    updated_at: datetime


@router.get("/memories", operation_id="list_memories", response_model=list[MemoryRead])
def list_memories(
    user_id: str = "default", session: Session = Depends(get_session)
) -> list[Memory]:
    # sync DB work in a plain def route: FastAPI runs it in the threadpool
    statement = (
        select(Memory)
        .where(Memory.user_id == user_id, Memory.is_deleted == False)  # noqa: E712
        .order_by(Memory.updated_at.desc())
    )
    return list(session.exec(statement).all())