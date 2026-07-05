from fastapi import Request
from openai import OpenAI

from app.config import Settings


def build_llm_client(settings: Settings) -> OpenAI:
    """The one OpenAI-compatible client for all LLM + embedding calls.

    The base_url decides the provider (AI Gateway by default, Ollama locally);
    the rest of the app never knows which one is behind it.
    """
    return OpenAI(
        api_key=settings.ai_gateway_api_key or "unset",
        base_url=settings.ai_gateway_base_url,
    )


def get_llm(request: Request) -> OpenAI:
    """Dependency: the lifespan-owned client from app.state."""
    return request.app.state.llm