"""The manual style-inference trigger: re-read the user's communication style now.

The automatic path re-infers every REFRESH_EVERY user messages; this endpoint is the on-demand
button — run the inference immediately (off the request path) and watch the inferred traits
land on the Memory page with `source="inferred"` and their message receipts.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.auth import get_current_user
from app.config import get_settings
from app.memory.style import run_style_refresh

router = APIRouter(tags=["style"])


@router.post("/style/refresh", operation_id="refresh_style", status_code=202)
def refresh_style(
    request: Request,
    background: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> dict[str, str]:
    """Infer style from the user's recent messages in the background; 202 returns immediately.
    Skips silently when there are too few messages to read a style from."""
    background.add_task(
        run_style_refresh,
        request.app.state.engine, request.app.state.llm, get_settings(),
        user_id=user_id,
    )
    return {"status": "inferring"}