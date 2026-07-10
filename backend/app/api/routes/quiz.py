"""The taste-quiz onboarding surface: fetch the questions, submit answers.

Answers are seeded into memory OFF the request path (write-path rule) at low confidence — the
submit returns immediately; the seeded facts show up on the Memory page as they distil.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel

from app.auth import get_current_user
from app.config import get_settings
from app.memory.onboarding import QUESTIONS, seed_from_quiz

router = APIRouter(tags=["quiz"])


class QuizQuestion(BaseModel):
    id: str
    prompt: str


class QuizAnswer(BaseModel):
    question: str
    answer: str


class QuizSubmit(BaseModel):
    answers: list[QuizAnswer]


@router.get("/quiz", operation_id="get_quiz", response_model=list[QuizQuestion])
def get_quiz(_user: str = Depends(get_current_user)) -> list[QuizQuestion]:
    """The fixed onboarding questions (auth-gated like the rest of the API)."""
    return [QuizQuestion(**q) for q in QUESTIONS]


@router.post("/quiz", operation_id="submit_quiz", status_code=202)
def submit_quiz(
    req: QuizSubmit,
    request: Request,
    background: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> dict[str, int]:
    """Seed the non-empty answers into memory in the background; return how many are seeding."""
    answered = [a for a in req.answers if a.answer.strip()]
    if answered:
        background.add_task(
            seed_from_quiz,
            request.app.state.engine, request.app.state.llm, get_settings(),
            user_id=user_id, answers=[a.model_dump() for a in answered],
        )
    return {"seeding": len(answered)}