from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single typed source of truth for configuration.

    Values come from environment variables, with `.env` as the local fallback.
    Nothing else in the app reads os.environ directly.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://chatmemory:chatmemory@localhost:5434/chatmemory"

    # DB connection pool. Ceiling per PROCESS = db_pool_size + db_max_overflow, and each always-on
    # process (the REST app AND the MCP server) opens its OWN pool of this size against one Postgres.
    # Two separate concerns:
    #  - Per process, the MCP tool path opens one connection per worker thread, so a ceiling ABOVE
    #    the ~40 anyio to_thread limiter means that path is bounded by the thread limiter (graceful
    #    wait), never by blocking on the pool. NOTE this does NOT hold for the REST streaming /chat
    #    path: it pins its connection across the whole token stream + background write (and the
    #    background write opens a SECOND connection, so peak is 2 per in-flight stream → effective
    #    /chat capacity during that window is ~half the ceiling), so THERE the pool (not the thread
    #    limiter) caps concurrent streams. Releasing the connection before the LLM call is the
    #    deeper fix (deferred).
    #  - Across processes, the SUM of every always-on pool's ceiling must stay under Postgres
    #    max_connections (default 100) minus superuser_reserved_connections (default 3) ≈ 97. At 45
    #    each, one REST + one MCP = 90. Raise Postgres capacity before adding replicas.
    db_pool_size: int = Field(default=15, gt=0)
    db_max_overflow: int = Field(default=30, ge=0)

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

    # Supabase Auth: the frontend logs in with Supabase and sends its access-token JWT; the
    # backend verifies it against Supabase's PUBLIC keys (JWKS, asymmetric ES256) and reads the
    # user id from the 'sub' claim — the client never supplies its own user_id (non-negotiable #6).
    supabase_url: str = ""  # e.g. https://abcdefgh.supabase.co — set in the backend host env

    # reverse-geocoding (lat/lon -> place name) runs in the WORKER only, on photos that carry
    # GPS. Nominatim is fine for dev/low volume (single cached lookups); production bulk needs a
    # self-hosted instance or a paid geocoder — its usage policy forbids systematic querying.
    geocode_enabled: bool = True
    geocode_url: str = "https://nominatim.openstreetmap.org/reverse"
    geocode_zoom: int = 12  # ~town/suburb granularity
    geocode_user_agent: str = "chat-memory/0.1 (personal memory app)"

    # CORS: which frontend origins may call this API. "*" is fine while there's no auth/cookies;
    # restrict to the deployed frontend URL before going public.
    cors_origins: list[str] = ["*"]

    # pgvector HNSW search tuning (set per-transaction in db.install_hnsw_guc). iterative scan keeps
    # scanning the index until the post-filter (user_id/source) yields the LIMIT — without it a
    # multi-tenant filter silently under-returns. relaxed_order is the production default; we re-rank
    # by cosine in SQL so approximate ordering is harmless.
    hnsw_iterative_scan: str = "relaxed_order"  # off | strict_order | relaxed_order
    hnsw_ef_search: int = 100

    # Cross-conversation dialogue recall (retrieval.py). The chat read path searches past chat
    # episodes (hybrid keyword + dense, RRF-fused) so "did we talk about X?" is answerable.
    dialogue_candidates: int = 50   # per-channel candidate pool before fusion
    dialogue_top_k: int = 6         # excerpts injected into the prompt
    dialogue_max_distance: float = 0.6  # cosine-distance floor; a keyword hit bypasses it
    dialogue_window_bonus: float = 0.005  # additive time-boost; MUST stay < 1/(rrf_k+1) to avoid a hard tier
    dialogue_excerpt_chars: int = 400
    rrf_k: int = 60  # Reciprocal Rank Fusion constant (swept in eval; TREC default)

    # Remote MCP server (Streamable HTTP). The stdio entrypoint ignores all three. mcp_public_url
    # is the server's own resource identifier, advertised in WWW-Authenticate metadata — set it to
    # the deployed https URL in production.
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = 8010
    mcp_public_url: str = "http://127.0.0.1:8010/mcp"

    # the user's timezone, used to resolve "yesterday" and render excerpt dates in local time.
    # single-user default; per-user tz arrives with auth. Date-anchored recall is correct for ONE
    # tz until then — a second-tz user gets the wrong "yesterday" boundary (documented pre-mortem).
    user_tz: str = "Asia/Kolkata"


@lru_cache
def get_settings() -> Settings:
    return Settings()