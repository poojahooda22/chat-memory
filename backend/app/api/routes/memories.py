import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app.config import get_settings
from app.db import get_session
from app.memory.pipeline import record_exchange, refresh_summary
from app.models import Memory, MemoryHistory

router = APIRouter(tags=["memories"])


# ── schemas ──────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class RecordRequest(BaseModel):
    user_id: str = "default"
    conversation_id: str | None = None
    messages: list[Message]


class OperationOut(BaseModel):
    event: str
    memory_id: str | None
    text: str


class RecordResponse(BaseModel):
    episode_ids: list[str]
    operations: list[OperationOut]


class MemoryRead(BaseModel):
    id: uuid.UUID
    user_id: str
    content: str
    source_episode_ids: list[str]
    created_at: datetime
    updated_at: datetime


class HistoryRead(BaseModel):
    event: str
    old_content: str | None
    new_content: str | None
    created_at: datetime


# ── write path: the two-phase pipeline ───────────────────────────────────────
@router.post("/memories", operation_id="record_memories", response_model=RecordResponse)
def record_memories(
    req: RecordRequest,
    request: Request,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> RecordResponse:
    # sync pipeline (blocking DB + LLM I/O) in a plain def -> FastAPI runs it in the threadpool
    settings = get_settings()
    result = record_exchange(
        session, request.app.state.llm, settings,
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        messages=[m.model_dump() for m in req.messages],
    )
    session.commit()

    # refresh the rolling summary off the request path (fire-and-forget; a lost run is harmless)
    if req.conversation_id:
        background.add_task(
            _refresh_summary_task,
            request.app.state.engine, request.app.state.llm, settings, req.conversation_id,
        )

    return RecordResponse(
        episode_ids=result.episode_ids,
        operations=[
            OperationOut(event=o.event, memory_id=o.memory_id, text=o.text)
            for o in result.operations
        ],
    )


def _refresh_summary_task(engine, llm, settings, conversation_id: str) -> None:
    with Session(engine) as session:
        refresh_summary(session, llm, settings, conversation_id)
        session.commit()


# ── read path ────────────────────────────────────────────────────────────────
@router.get("/memories", operation_id="list_memories", response_model=list[MemoryRead])
def list_memories(
    user_id: str = "default", session: Session = Depends(get_session)
) -> list[Memory]:
    statement = (
        select(Memory)
        .where(Memory.user_id == user_id, Memory.is_deleted == False)  # noqa: E712
        .order_by(col(Memory.updated_at).desc())
    )
    return list(session.exec(statement).all())


@router.get(
    "/memories/{memory_id}/history",
    operation_id="memory_history",
    response_model=list[HistoryRead],
)
def memory_history(
    memory_id: uuid.UUID, session: Session = Depends(get_session)
) -> list[MemoryHistory]:
    statement = (
        select(MemoryHistory)
        .where(MemoryHistory.memory_id == memory_id)
        .order_by(col(MemoryHistory.created_at))
    )
    return list(session.exec(statement).all())


@router.delete("/memories/{memory_id}", operation_id="delete_memory")
def delete_memory(
    memory_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict[str, str]:
    memory = session.get(Memory, memory_id)
    if memory is None or memory.is_deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    memory.is_deleted = True
    session.add(memory)
    session.add(MemoryHistory(memory_id=memory.id, event="DELETE", old_content=memory.content))
    session.commit()
    return {"status": "deleted", "memory_id": str(memory_id)}