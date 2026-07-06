from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel
from sqlmodel import Session

from app.config import get_settings
from app.db import get_session
from app.memory.chat import answer
from app.memory.pipeline import run_summary_refresh

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str = "default"
    conversation_id: str | None = None
    message: str


class OperationOut(BaseModel):
    event: str
    memory_id: str | None
    text: str


class ChatResponse(BaseModel):
    reply: str
    memories_used: list[str]  # the memories the assistant drew on — visible for transparency
    photos_used: list[str]  # photo-derived episodes it drew on, with capture dates
    operations: list[OperationOut]  # what this turn changed in memory


@router.post("/chat", operation_id="chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> ChatResponse:
    # sync (blocking DB + LLM) in a plain def -> FastAPI runs it in the threadpool
    settings = get_settings()
    result = answer(
        session, request.app.state.llm, settings,
        user_id=req.user_id, conversation_id=req.conversation_id, message=req.message,
    )
    session.commit()

    if req.conversation_id:
        background.add_task(
            run_summary_refresh,
            request.app.state.engine, request.app.state.llm, settings, req.conversation_id,
        )

    return ChatResponse(
        reply=result.reply,
        memories_used=result.memories_used,
        photos_used=result.photos_used,
        operations=[
            OperationOut(event=o.event, memory_id=o.memory_id, text=o.text)
            for o in result.operations
        ],
    )
