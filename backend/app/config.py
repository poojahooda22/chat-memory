from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single typed source of truth for configuration.

    Values come from environment variables, with `.env` as the local fallback.
    Nothing else in the app reads os.environ directly.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://chatmemory:chatmemory@localhost:5434/chatmemory"

    ai_gateway_api_key: str = ""
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v1"
    llm_model: str = "openai/gpt-4o-mini"
    # gpt-4o-mini is multimodal, so the image annotation rides the same gateway model by
    # default; override independently if a stronger vision model is ever needed.
    vision_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    # where uploaded originals live (gitignored); R2/object storage is the later fork
    uploads_dir: str = "uploads"

    # CORS: which frontend origins may call this API. "*" is fine while there's no auth/cookies;
    # restrict to the deployed frontend URL before going public.
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()