from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session
from starlette.background import BackgroundTask

from app.auth import get_current_user
from app.config import get_settings
from app.db import get_session
from app.memory.chat import prepare_reply
from app.memory.pipeline import learn_from_exchange
from app.memory.style import maybe_refresh_style

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    # user_id is derived from the auth token, never sent by the client
    conversation_id: str | None = None
    message: str


@router.post("/chat", operation_id="chat")
def chat(
    req: ChatRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Stream the reply token by token, then learn from the exchange off the request path.

    RETRIEVE + prompt-build runs first (not streamed); GENERATE streams so the first words reach
    the user in ~2-3s instead of only after the whole reply is done. The memory WRITE
    (learn_from_exchange) runs as a background task AFTER the full reply is sent — the user never
    waits for it, and it opens its own DB session.
    """
    settings = get_settings()
    llm = request.app.state.llm
    prepared = prepare_reply(
        session, llm, settings,
        user_id=user_id, conversation_id=req.conversation_id, message=req.message,
    )
    reply_parts: list[str] = []

    def token_stream():
        completion = llm.chat.completions.create(
            model=settings.llm_model, messages=prepared.messages, temperature=0.7, stream=True,
        )
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                reply_parts.append(delta)
                yield delta

    def learn_after_reply() -> None:
        # runs once the stream is fully sent -> reply_parts holds the complete reply
        learn_from_exchange(
            request.app.state.engine, llm, settings,
            user_id=user_id, conversation_id=req.conversation_id,
            messages=[
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": "".join(reply_parts)},
            ],
        )
        # every REFRESH_EVERY user messages, re-read the user's communication style off their
        # own writing (no-op most turns; still off the request path either way)
        maybe_refresh_style(
            request.app.state.engine, llm, settings, user_id=user_id,
        )

    return StreamingResponse(
        token_stream(),
        media_type="text/plain; charset=utf-8",
        background=BackgroundTask(learn_after_reply),
    )