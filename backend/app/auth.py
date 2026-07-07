"""Authentication: verify the Supabase Auth JWT and derive the user id from it.

The frontend logs in with Supabase and sends the access token as `Authorization: Bearer <jwt>`.
We verify the signature against Supabase's PUBLIC keys (JWKS, asymmetric ES256 — the new-project
default; RS256 also accepted), fetched and cached from the project's `/.well-known/jwks.json`
endpoint, and read the user id from the `sub` claim.

The client never supplies its own user_id — the id is derived from the verified token — so a
signed-in user cannot act as, read, or write another user's memory (non-negotiable #6). Every
user-scoped route depends on get_current_user; resource routes additionally check ownership.
"""

from functools import lru_cache

import jwt
from fastapi import Depends, Header, HTTPException
from jwt import PyJWKClient

from app.config import Settings, get_settings

# Supabase signs access tokens with the project's asymmetric key (ES256 by default); RS256 too.
_ALGORITHMS = ["ES256", "RS256"]


@lru_cache
def _jwks_client(jwks_url: str) -> PyJWKClient:
    """One client per JWKS URL — it fetches the public keys once and caches them in memory."""
    return PyJWKClient(jwks_url)


def _signing_key(token: str, settings: Settings):
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="Auth is not configured (supabase_url unset)")
    jwks_url = f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    return _jwks_client(jwks_url).get_signing_key_from_jwt(token).key


def verify_token(token: str, settings: Settings) -> str:
    """Verify a Supabase JWT and return its user id (`sub`). Raises 401 on any problem."""
    try:
        key = _signing_key(token, settings)
        claims = jwt.decode(
            token,
            key,
            algorithms=_ALGORITHMS,
            # Supabase sets aud='authenticated'; identity is the 'sub' claim, so we don't gate on aud
            options={"verify_aud": False},
        )
    except HTTPException:
        raise
    except Exception as exc:  # signature/expiry/format — all mean "not a valid token"
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return str(user_id)


def get_current_user(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency → the authenticated user's id (401 if missing/invalid)."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return verify_token(token.strip(), settings)