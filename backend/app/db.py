from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine, event
from sqlmodel import Session, create_engine

from app.config import Settings

_ALLOWED_ITERATIVE_SCAN = {"off", "strict_order", "relaxed_order"}


def install_hnsw_guc(engine: Engine, settings: Settings) -> None:
    """Set pgvector's HNSW search GUCs per-transaction on every connection this engine opens.

    Without iterative scan a `WHERE user_id = ?` filter is applied AFTER the index returns only
    `ef_search` candidates, so a tenant owning a small fraction of rows silently gets fewer than
    the requested top-k (recall collapse). `relaxed_order` keeps scanning until the LIMIT is
    satisfied. SET LOCAL scopes it to the current transaction only, so it never leaks onto a
    pooled connection's next reuse — which is why it lives in the `begin` event, not `connect`.

    Registered on BOTH the app engine (build_engine) and the test engine (conftest) so the
    behavior under test matches production — a listener-less test engine would pass a `SHOW`
    assertion that production fails.
    """
    mode = settings.hnsw_iterative_scan
    if mode not in _ALLOWED_ITERATIVE_SCAN:
        mode = "relaxed_order"
    ef_search = int(settings.hnsw_ef_search)

    @event.listens_for(engine, "begin")
    def _set_hnsw_guc(conn) -> None:  # conn is inside the just-started transaction
        conn.exec_driver_sql(f"SET LOCAL hnsw.iterative_scan = {mode}")
        conn.exec_driver_sql(f"SET LOCAL hnsw.ef_search = {ef_search}")


def build_engine(settings: Settings) -> Engine:
    """Create the one engine (connection pool). Owned by the app lifespan.

    The pool is sized explicitly (not the SQLAlchemy default 5+10=15). Per process the ceiling is
    `db_pool_size + db_max_overflow`, set above the ~40 worker-thread limiter so the MCP tool path —
    one connection per worker thread — is bounded by the thread limiter, not by blocking on the
    pool. This does NOT hold for the REST streaming /chat path, which holds its connection across
    the whole stream + background write, so there the pool caps concurrent streams (releasing it
    before the LLM call is the deeper fix, deferred). The pool is per-process and each always-on
    process (REST, MCP) has its own — keep the sum under Postgres max_connections; see config.py.
    """
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    install_hnsw_guc(engine, settings)
    return engine


def get_session(request: Request) -> Iterator[Session]:
    """Per-request DB session, drawn from the lifespan-owned engine on app.state."""
    with Session(request.app.state.engine) as session:
        yield session