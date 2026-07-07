"""Auth: verify_token accepts a valid ES256 token and returns its user id ('sub'), and rejects
tampered / expired / wrong-key / subject-less tokens. Uses a locally-generated EC keypair and
monkeypatches the JWKS key fetch, so the suite needs no Supabase project and no network."""

import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from app import auth
from app.config import Settings

SETTINGS = Settings(supabase_url="https://test.supabase.co")


@pytest.fixture
def keypair():
    priv = ec.generate_private_key(ec.SECP256R1())  # the curve Supabase uses (ES256)
    return priv, priv.public_key()


def _use_key(monkeypatch, public_key):
    """Make verify_token trust `public_key` instead of fetching Supabase's JWKS."""
    monkeypatch.setattr(auth, "_signing_key", lambda token, settings: public_key)


def _token(private_key, **claims) -> str:
    return jwt.encode(claims, private_key, algorithm="ES256")


def test_valid_token_returns_sub(monkeypatch, keypair):
    priv, pub = keypair
    _use_key(monkeypatch, pub)
    uid = str(uuid.uuid4())
    token = _token(priv, sub=uid, aud="authenticated", exp=int(time.time()) + 3600)
    assert auth.verify_token(token, SETTINGS) == uid


def test_garbage_token_rejected(monkeypatch, keypair):
    _use_key(monkeypatch, keypair[1])
    with pytest.raises(HTTPException) as exc:
        auth.verify_token("not.a.jwt", SETTINGS)
    assert exc.value.status_code == 401


def test_expired_token_rejected(monkeypatch, keypair):
    priv, pub = keypair
    _use_key(monkeypatch, pub)
    token = _token(priv, sub="x", exp=int(time.time()) - 10)
    with pytest.raises(HTTPException) as exc:
        auth.verify_token(token, SETTINGS)
    assert exc.value.status_code == 401


def test_token_signed_by_wrong_key_rejected(monkeypatch, keypair):
    priv, _ = keypair
    other_pub = ec.generate_private_key(ec.SECP256R1()).public_key()
    _use_key(monkeypatch, other_pub)  # a different key than the one that signed it
    token = _token(priv, sub="x", exp=int(time.time()) + 3600)
    with pytest.raises(HTTPException) as exc:
        auth.verify_token(token, SETTINGS)
    assert exc.value.status_code == 401


def test_subjectless_token_rejected(monkeypatch, keypair):
    priv, pub = keypair
    _use_key(monkeypatch, pub)
    token = _token(priv, aud="authenticated", exp=int(time.time()) + 3600)  # no 'sub'
    with pytest.raises(HTTPException) as exc:
        auth.verify_token(token, SETTINGS)
    assert exc.value.status_code == 401