"""Belief-revision lineage: predecessors_for() batching, recall() surfacing revised/previously/
has_older, the MCP Fact serialization, and the linearity invariant that keeps the 1:1
predecessor->successor map honest. Zero tokens — recall's only LLM call (decompose) is scripted.
"""

import json
import uuid
from datetime import datetime

from sqlalchemy import event
from sqlmodel import col, select

from app.config import get_settings
from app.memory.lineage import predecessors_for
from app.memory.retrieval import recall
from app.models import Memory
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()

# a decompose reply with no filters -> recall falls to plain fact cosine search
_NO_FILTERS = json.dumps(
    {"entities": [], "time_range": None, "place": None, "semantic_query": "x", "wants_all": False}
)


def _user() -> str:
    return f"lin-{uuid.uuid4()}"


def _mem(db, user, content, *, superseded_by=None, is_superseded=False) -> Memory:
    """Insert a Memory row directly (no pipeline) so a supersede chain can be wired by hand."""
    m = Memory(
        user_id=user, content=content, embedding=fake_embedding(content),
        source_episode_ids=[], source="chat", confidence=1.0,
        is_superseded=is_superseded, superseded_by_id=superseded_by,
    )
    db.add(m)
    db.flush()
    return m


def _chain(db, user, *contents: str) -> list[Memory]:
    """Build oldest..live for the given contents: the last is live, each earlier one is superseded
    by the next. Returns [oldest, ..., live]. superseded_by_id must point forward, so create the
    live row first and walk backward."""
    rows: list[Memory] = [None] * len(contents)  # type: ignore[list-item]
    successor_id = None
    for i in range(len(contents) - 1, -1, -1):
        rows[i] = _mem(
            db, user, contents[i],
            superseded_by=successor_id, is_superseded=(i != len(contents) - 1),
        )
        successor_id = rows[i].id
    return rows


def test_predecessors_for_is_batched_no_n_plus_1(db_session):
    """N live facts (some revised, some fresh) resolve in a BOUNDED number of queries, not per-fact."""
    user = _user()
    # two revision chains + one fresh fact -> 3 live facts, 2 of them with predecessors
    _a1, b1 = _chain(db_session, user, "was a frontend developer", "is a backend developer")
    _a2, _b2, c2 = _chain(db_session, user, "lived in Delhi", "lived in Pune", "lives in Bengaluru")
    fresh = _mem(db_session, user, "enjoys hiking")

    conn = db_session.connection()
    calls: list[int] = []

    def _tick(*a, **k):
        calls.append(1)

    event.listen(conn, "after_cursor_execute", _tick)
    try:
        got = predecessors_for(db_session, [b1.id, c2.id, fresh.id])
    finally:
        event.remove(conn, "after_cursor_execute", _tick)

    # exactly the two revised facts are present; the fresh one is absent
    assert set(got.keys()) == {b1.id, c2.id}
    assert got[b1.id].content == "was a frontend developer"
    assert got[b1.id].has_older is False  # A->B, no older
    assert got[c2.id].content == "lived in Pune"
    assert got[c2.id].has_older is True  # A->B->C, an older revision exists
    # the whole thing is two batched queries regardless of N — never an N+1 loop
    assert len(calls) <= 2


def test_predecessors_for_empty_input_hits_no_db(db_session):
    conn = db_session.connection()
    calls: list[int] = []

    def _tick(*a, **k):
        calls.append(1)

    event.listen(conn, "after_cursor_execute", _tick)
    try:
        assert predecessors_for(db_session, []) == {}
    finally:
        event.remove(conn, "after_cursor_execute", _tick)
    assert calls == []  # short-circuits before touching the DB


def test_recall_surfaces_a_single_revision(db_session):
    user = _user()
    _old, live = _chain(db_session, user, "works as a frontend developer", "works as a backend developer")

    bundle = recall(db_session, FakeLLM([_NO_FILTERS]), SETTINGS, user_id=user, message="what's my job?")

    assert [f.content for f in bundle.facts] == ["works as a backend developer"]
    fact = bundle.facts[0]
    assert fact.revised is True
    assert fact.previously == "works as a frontend developer"
    assert fact.has_older is False
    # sourced from the LIVE row's created_at (populated only when revised), not the predecessor's
    assert isinstance(fact.ingested_at, datetime)
    assert live.id in {uuid.UUID(fact.memory_id)}  # the recalled row IS the live successor


def test_recall_multi_hop_sets_has_older(db_session):
    user = _user()
    # all three share the "developer/job" concept-slot so cosine returns the live one
    rows = _chain(db_session, user, "was a junior developer", "was a mid developer", "is a senior developer")
    live = rows[-1]

    bundle = recall(db_session, FakeLLM([_NO_FILTERS]), SETTINGS, user_id=user, message="what is my job?")

    fact = next(f for f in bundle.facts if f.memory_id == str(live.id))
    assert fact.content == "is a senior developer"
    assert fact.previously == "was a mid developer"  # only the IMMEDIATE prior, not "junior"
    assert fact.has_older is True  # ...but the flag signals the deeper history


def test_recall_fresh_fact_is_not_revised(db_session):
    user = _user()
    _mem(db_session, user, "works as a developer")

    bundle = recall(db_session, FakeLLM([_NO_FILTERS]), SETTINGS, user_id=user, message="what is my job?")

    fact = bundle.facts[0]
    assert fact.revised is False
    assert fact.previously is None
    assert fact.ingested_at is None
    assert fact.has_older is False


def test_mcp_fact_serializes_lineage_fields():
    """The MCP Fact model accepts and serializes the lineage fields (ingested_at as an ISO string)."""
    from mcp_server import Fact

    f = Fact(
        content="uses Svelte", source="chat", confidence=1.0, source_episode_ids=[],
        revised=True, previously="used Vue", ingested_at="2026-07-10T12:00:00+00:00", has_older=True,
    )
    dumped = f.model_dump()
    assert dumped["revised"] is True
    assert dumped["previously"] == "used Vue"
    assert dumped["ingested_at"] == "2026-07-10T12:00:00+00:00"
    assert dumped["has_older"] is True
    # defaults keep a fresh fact quiet
    fresh = Fact(content="x", source="chat", confidence=1.0, source_episode_ids=[])
    assert fresh.revised is False and fresh.previously is None and fresh.ingested_at is None


def test_supersede_chain_stays_linear(db_session):
    """The 1:1 predecessor->successor map is load-bearing: at most one row may point at any live
    fact via superseded_by_id. A future write-path change that forks a chain fails HERE, loudly."""
    user = _user()
    rows = _chain(db_session, user, "A", "B", "C")
    for r in rows:
        pointers = list(
            db_session.exec(select(Memory.id).where(col(Memory.superseded_by_id) == r.id)).all()
        )
        assert len(pointers) <= 1, f"row {r.content} has {len(pointers)} predecessors — chain forked"