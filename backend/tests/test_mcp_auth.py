"""Remote-MCP auth: the Supabase token verifier, the transport-gated identity rule, and the
Streamable-HTTP 401/200 boundary.

No live Supabase and no DB — tokens are self-signed ES256 and the JWKS lookup is patched to the
matching public key, so the REAL decode path (signature, expiry, subject) runs end to end.
"""

import importlib
import threading
import time
from types import SimpleNamespace

import anyio
import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

import mcp_server
from app.config import Settings
from app.mcp_auth import SupabaseTokenVerifier


def _empty_bundle(*args, **kwargs):
    """A recall_bundle stand-in: records nothing, opens no DB, returns the shape `recall` maps."""
    return SimpleNamespace(
        facts=[], photo_lines=[], dialogue_lines=[], confidence=0.0, evidence="none"
    )

_KEY = generate_private_key(SECP256R1())
_PUBLIC = _KEY.public_key()
_SETTINGS = Settings(supabase_url="https://unit-test.supabase.co")

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.0.1"},
    },
}
_HEADERS = {"Accept": "application/json, text/event-stream"}


def _token(sub: str | None = "user-1", exp_delta: int = 3600, key=_KEY) -> str:
    claims: dict = {"exp": int(time.time()) + exp_delta, "aud": "authenticated"}
    if sub is not None:
        claims["sub"] = sub
    return jwt.encode(claims, key, algorithm="ES256")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def patched_jwks(monkeypatch):
    """Point signature verification at the test keypair instead of live Supabase JWKS."""
    monkeypatch.setattr("app.auth._signing_key", lambda token, settings: _PUBLIC)


# ── the verifier: valid in, AccessToken out; anything else is None (→ SDK 401) ──
@pytest.mark.anyio
async def test_verifier_accepts_valid_token(patched_jwks):
    got = await SupabaseTokenVerifier(_SETTINGS).verify_token(_token(sub="user-42"))
    assert got is not None
    assert got.subject == "user-42"
    assert got.client_id == "user-42"  # the user IS the principal
    assert got.expires_at is not None and got.expires_at > int(time.time())


@pytest.mark.anyio
async def test_verifier_rejects_wrong_key(patched_jwks):
    intruder_key = generate_private_key(SECP256R1())
    assert await SupabaseTokenVerifier(_SETTINGS).verify_token(_token(key=intruder_key)) is None


@pytest.mark.anyio
async def test_verifier_rejects_expired(patched_jwks):
    assert await SupabaseTokenVerifier(_SETTINGS).verify_token(_token(exp_delta=-60)) is None


@pytest.mark.anyio
async def test_verifier_rejects_missing_subject(patched_jwks):
    assert await SupabaseTokenVerifier(_SETTINGS).verify_token(_token(sub=None)) is None


@pytest.mark.anyio
async def test_verifier_rejects_garbage(patched_jwks):
    assert await SupabaseTokenVerifier(_SETTINGS).verify_token("not-a-jwt") is None


# ── the identity rule: token subject wins; the env fallback is stdio-only ──
def _set_auth_context(subject: str | None):
    token = AccessToken(token="raw", client_id="c", scopes=[], subject=subject)
    return auth_context_var.set(AuthenticatedUser(token))


def test_user_id_prefers_token_subject():
    ctx = _set_auth_context("user-7")
    try:
        assert mcp_server._user_id() == "user-7"
    finally:
        auth_context_var.reset(ctx)


def test_user_id_rejects_token_without_subject():
    ctx = _set_auth_context(None)
    try:
        with pytest.raises(RuntimeError, match="no subject"):
            mcp_server._user_id()
    finally:
        auth_context_var.reset(ctx)


def test_user_id_stdio_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(mcp_server, "TRANSPORT", "stdio")
    monkeypatch.setenv("MCP_USER_ID", "local-user")
    assert mcp_server._user_id() == "local-user"


def test_user_id_remote_never_uses_env(monkeypatch):
    """The isolation line: on a remote transport an unauthenticated call must FAIL, even with
    MCP_USER_ID present in the environment — identity comes from the token or not at all."""
    monkeypatch.setattr(mcp_server, "TRANSPORT", "streamable-http")
    monkeypatch.setenv("MCP_USER_ID", "local-user")
    with pytest.raises(RuntimeError, match="remote transport"):
        mcp_server._user_id()


# ── boot guards ──
def test_build_server_requires_supabase_url():
    with pytest.raises(RuntimeError, match="supabase_url"):
        mcp_server._build_server("streamable-http", Settings(supabase_url=""))


def test_transport_typo_fails_boot(monkeypatch):
    """A mis-spelled MCP_TRANSPORT must kill the boot, not silently run an unauthenticated stdio
    server where a remote one was intended."""
    monkeypatch.setenv("MCP_TRANSPORT", "streamble-http")
    try:
        with pytest.raises(RuntimeError, match="MCP_TRANSPORT"):
            importlib.reload(mcp_server)
    finally:
        monkeypatch.delenv("MCP_TRANSPORT")
        importlib.reload(mcp_server)  # restore the module for the rest of the suite


# ── the HTTP boundary: 401 without a token, 200 initialize with one ──
@pytest.mark.anyio
async def test_http_rejects_missing_token():
    server = mcp_server._build_server("streamable-http", _SETTINGS)
    app = server.streamable_http_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8010") as client:
        resp = await client.post("/mcp", json=_INIT, headers=_HEADERS)
    assert resp.status_code == 401
    assert "bearer" in resp.headers.get("www-authenticate", "").lower()


@pytest.mark.anyio
async def test_http_initialize_with_valid_token(patched_jwks):
    server = mcp_server._build_server("streamable-http", _SETTINGS)
    app = server.streamable_http_app()
    transport = httpx.ASGITransport(app=app)
    async with server.session_manager.run():
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8010") as client:
            resp = await client.post(
                "/mcp",
                json=_INIT,
                headers={**_HEADERS, "Authorization": f"Bearer {_token(sub='user-9')}"},
            )
    assert resp.status_code == 200
    assert "chat-memory" in resp.text  # the serverInfo of a successful initialize


# ── the end-to-end isolation proof: CONCURRENT tools/call requests each scope to their own subject ──
@pytest.mark.anyio
async def test_http_tool_call_scopes_to_token_subject(patched_jwks, monkeypatch):
    """The whole guarantee, over the wire and UNDER OVERLAP: two tools/call requests carrying
    DIFFERENT bearer tokens, forced to be in-flight simultaneously by a barrier, must each resolve
    `_user_id()` to their OWN token subject inside the running tool. Holding both bodies open at
    once is what would expose a contextvar cross-contamination bug (A's id leaking into B's
    concurrent request) — proving the AuthenticationMiddleware -> AuthContextMiddleware -> contextvar
    -> async tool -> to_thread chain both propagates AND isolates identity per request.

    Stateless streamable-http boots the session as already-initialized, so a single tools/call POST
    is dispatched directly (no handshake needed). The remote path is exercised via
    _build_server('streamable-http', ...) + real bearer-token auth — not via the module TRANSPORT,
    which the token-carrying identity path never reads."""
    barrier = threading.Barrier(2, timeout=5)
    seen: dict[str, str] = {}  # query (which carries the caller's subject) -> resolved user_id

    def _capture(session, client, settings, *, user_id, message):
        barrier.wait()  # both requests must reach here together — deadlocks if they serialize
        seen[message] = user_id
        return _empty_bundle()

    monkeypatch.setattr(mcp_server, "recall_bundle", _capture)
    server = mcp_server._build_server("streamable-http", _SETTINGS)
    app = server.streamable_http_app()
    transport = httpx.ASGITransport(app=app)

    async def _post(subject: str) -> None:
        body = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {"query": subject}},  # query == this caller's subject
        }
        headers = {**_HEADERS, "Authorization": f"Bearer {_token(sub=subject)}"}
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8010") as client:
            resp = await client.post("/mcp", json=body, headers=headers)
            assert resp.status_code == 200

    async with server.session_manager.run():
        async with anyio.create_task_group() as tg:
            tg.start_soon(_post, "user-A")
            tg.start_soon(_post, "user-B")

    # each concurrent request resolved to ITS OWN subject — no cross-contamination under overlap
    assert seen == {"user-A": "user-A", "user-B": "user-B"}


# ── the concurrency proof: the tool body runs OFF the event loop ──
@pytest.mark.anyio
async def test_tool_body_runs_off_the_event_loop(patched_jwks, monkeypatch):
    """Two concurrent recall() calls must run their blocking bodies in PARALLEL worker threads. The
    barrier needs BOTH to arrive within the timeout; if the body ran inline on the event loop the
    first call's blocking wait would freeze the loop, the second would never start, and the barrier
    would break. Passing proves the to_thread offload — the fix for the event-loop-block finding."""
    barrier = threading.Barrier(2, timeout=5)

    def _blocking_capture(session, client, settings, *, user_id, message):
        barrier.wait()  # only clears if a SECOND call reaches here concurrently
        return _empty_bundle()

    monkeypatch.setattr(mcp_server, "recall_bundle", _blocking_capture)
    ctx = _set_auth_context("user-concurrent")
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(mcp_server.recall, "q1")
            tg.start_soon(mcp_server.recall, "q2")
    finally:
        auth_context_var.reset(ctx)
    # reaching here without a BrokenBarrierError means both bodies ran concurrently, off the loop