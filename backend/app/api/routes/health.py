from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", operation_id="health_check")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}