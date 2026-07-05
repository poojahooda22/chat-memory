"""Test fixtures.

`db_session` binds a Session to a single connection wrapped in a transaction that is rolled
back after each test — so tests run against the real dockerized Postgres (with pgvector and
the migrated schema) but never leave a trace. Requires `docker compose up -d` + migrations.

`FakeLLM` stands in for the OpenAI-compatible client so pipeline tests cost zero tokens. Chat
replies are scripted; embeddings are deterministic 1536-vectors keyed by a "concept slot", so
two texts about the same concept (e.g. any job) get identical vectors — giving cosine search a
controllable, exact top hit.
"""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session

from app.config import get_settings

EMBEDDING_DIM = 1536


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine(get_settings().database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()


def _concept_slot(text: str) -> int:
    """Map a text to a fixed embedding slot by keyword, so similar concepts collide."""
    t = text.lower()
    if any(w in t for w in ("developer", "engineer", "job", "work")):
        return 0
    if "python" in t:
        return 1
    if any(w in t for w in ("pet", "dog", "cat", "puppy")):
        return 2
    if "pooja" in t or "name" in t:
        return 3
    return 4 + (abs(hash(t)) % 100)  # everything else scatters into its own slot


def _slot_vector(slot: int) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    vec[slot % EMBEDDING_DIM] = 1.0
    return vec


class FakeLLM:
    """Minimal stand-in for the openai client: .embeddings.create and .chat.completions.create."""

    def __init__(self, chat_replies: list[str]) -> None:
        self._chat_replies = list(chat_replies)
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _embed(self, model: str, input: str):  # noqa: A002 - mirrors openai's param name
        vector = _slot_vector(_concept_slot(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=vector)])

    def _chat(self, **kwargs):
        content = self._chat_replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])