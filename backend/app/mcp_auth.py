"""Token verification for the remote MCP server.

The remote MCP surface is an OAuth resource server: every request must carry a Supabase access
token as `Authorization: Bearer <jwt>`. The SDK wires this verifier into its auth middleware —
a request whose token fails here is rejected with 401 + WWW-Authenticate before any tool runs,
and a valid token's `sub` becomes the acting user id (the same identity rule as the REST API:
the client never supplies its own user_id).
"""

import logging

import anyio
from mcp.server.auth.provider import AccessToken

from app.auth import verify_token_claims
from app.config import Settings

log = logging.getLogger("chat-memory-mcp")


class SupabaseTokenVerifier:
    """Implements the SDK's TokenVerifier protocol over the app's existing JWKS verification.

    Returning None (never raising) is the protocol's "invalid token" signal — the SDK's
    RequireAuthMiddleware turns it into a 401 for the caller.
    """

    def __init__(self, settings: Settings):
        self._settings = settings

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            # PyJWKClient does blocking HTTP on a cold key cache — keep it off the event loop
            claims = await anyio.to_thread.run_sync(verify_token_claims, token, self._settings)
        except Exception:
            log.info("MCP auth: rejected an invalid or expired token")
            return None
        subject = claims.get("sub")
        if not subject:
            log.info("MCP auth: rejected a token without a subject")
            return None
        return AccessToken(
            token=token,
            # Supabase issues user tokens, not OAuth-client tokens: the user IS the principal,
            # so the subject doubles as the client id (keeps principal comparisons unique)
            client_id=str(subject),
            scopes=[],
            expires_at=claims.get("exp"),
            subject=str(subject),
            claims=claims,
        )
