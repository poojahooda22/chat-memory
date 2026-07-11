"""Cross-conversation recall — the "did we talk about TanStack?" fix.

Past chat episodes are searched at answer time by a HYBRID keyword+dense retrieval (RRF-fused), so a
question about an earlier conversation is answerable. These tests fail on the pre-fix behavior
(facts + image episodes only) and lock the round-3/4 negation findings: the keyword channel must
match a rare proper noun the dense channel misses, must never raise on punctuated/multi-word input,
and the relevance floor must keep a keyword-only (dense-far) hit.
"""

import json
import uuid
from datetime import UTC, date, datetime, timedelta

from app.config import get_settings
from app.memory.pipeline import EpisodeHit, safe_lexemes, search_episodes
from app.memory.retrieval import QuerySpec, _rerank_dialogue, recall
from app.models import Episode
from tests.conftest import FakeLLM, fake_embedding

SETTINGS = get_settings()


def _chat_ep(db, user, content, *, conversation_id="conv-A", when=None) -> Episode:
    ep = Episode(
        user_id=user, conversation_id=conversation_id,
        occurred_at=when or datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
        content=content, context={"source": "chat", "role": "assistant"},
        embedding=fake_embedding(content),
    )
    db.add(ep)
    db.flush()
    return ep


def _decompose(entities, semantic_query, time_range=None):
    return FakeLLM([json.dumps({
        "entities": entities, "time_range": time_range, "place": None,
        "semantic_query": semantic_query, "wants_all": False,
    })])


def test_safe_lexemes_sanitizes_and_never_produces_raw_query():
    # multi-word + punctuated terms are split to bare lexemes (to_tsquery would raise on them raw);
    # recall-scaffolding words are dropped; order-preserving de-dupe.
    assert safe_lexemes(["remember yesterday's TanStack chat?"]) == ["tanstack"]
    assert safe_lexemes(["TanStack Query", "Node.js"]) == ["tanstack", "query", "node", "js"]
    assert safe_lexemes(["what did we discuss about it"]) == []  # all scaffolding/stopwords


def test_keyword_builder_never_raises_on_punctuation(db_session):
    """Round-4 CRITICAL: to_tsquery raises on 'TanStack Query'/'Node.js'/'C++' and aborts the txn.
    The safe_lexemes builder must make the keyword channel crash-proof."""
    user = f"r-{uuid.uuid4()}"
    ep = _chat_ep(db_session, user, "We set up a Node.js and C++ toolchain for TanStack Query")
    llm = FakeLLM([])  # no LLM call — we pass the embedding
    hits = search_episodes(
        db_session, llm, SETTINGS, user_id=user, source="chat",
        semantic_query="tooling", keyword_sources=["Node.js", "C++", "TanStack Query", "user's data"],
        embedding=fake_embedding("tooling"),
    )
    assert any(h.episode.id == ep.id for h in hits)  # matched, and nothing raised


def test_incident_keyword_only_dense_far_survives_floor(db_session):
    """THE incident: a rare proper noun the dense channel misses (different concept slot => cosine
    distance 1.0 > floor) is caught by the keyword channel and MUST survive the relevance floor via
    its sparse_rank. Fails if the keyword channel is absent or the floor ignores keyword hits."""
    user = f"r-{uuid.uuid4()}"
    target = _chat_ep(db_session, user, "We discussed TanStack Query mutations and caching")
    llm = _decompose(entities=["TanStack Query"], semantic_query="TanStack Query discussion")
    bundle = recall(
        db_session, llm, SETTINGS, user_id=user,
        message="did we talk about TanStack?",
    )
    ids = [d.episode_id for d in bundle.dialogue]
    assert str(target.id) in ids, "the TanStack turn must be recalled via the keyword channel"
    # no distilled facts exist, but the answer IS available — evidence must not read as 'nothing'
    assert bundle.facts == []
    assert bundle.evidence == "dialogue"
    assert bundle.confidence == 0.0  # fact-confidence, not an 'empty' signal


def test_recent_turns_excluded_but_older_turns_still_recalled(db_session):
    """Dedup by the recent-turn episode IDs (not the whole conversation): the 6 injected turns are
    excluded from dialogue, but an OLDER turn of the same conversation stays searchable."""
    user = f"r-{uuid.uuid4()}"
    older = _chat_ep(db_session, user, "Way back we compared TanStack Query vs SWR",
                     when=datetime(2026, 7, 1, 9, 0, tzinfo=UTC))
    recent = [
        _chat_ep(db_session, user, f"recent turn {i} about TanStack",
                 when=datetime(2026, 7, 10, 12, i, tzinfo=UTC))
        for i in range(6)
    ]
    llm = FakeLLM([])
    hits = search_episodes(
        db_session, llm, SETTINGS, user_id=user, source="chat",
        semantic_query="TanStack", keyword_sources=["TanStack"],
        embedding=fake_embedding("TanStack"), exclude_episode_ids=[e.id for e in recent],
    )
    ids = {h.episode.id for h in hits}
    assert older.id in ids               # older same-conversation turn is still findable
    assert ids.isdisjoint({e.id for e in recent})  # the injected recent turns are excluded


def test_null_conversation_episode_is_recalled_with_exclusion_active(db_session):
    """Round-4: an episode with conversation_id=None must NOT be dropped when an exclusion set is
    active (the predicate must not silently swallow NULLs, and the empty/NULL cases must be safe)."""
    user = f"r-{uuid.uuid4()}"
    orphan = _chat_ep(db_session, user, "Imported note about TanStack Query", conversation_id=None)
    other = _chat_ep(db_session, user, "unrelated turn")
    llm = FakeLLM([])
    hits = search_episodes(
        db_session, llm, SETTINGS, user_id=user, source="chat",
        semantic_query="TanStack", keyword_sources=["TanStack"],
        embedding=fake_embedding("TanStack"), exclude_episode_ids=[other.id],
    )
    assert any(h.episode.id == orphan.id for h in hits)


def test_window_bonus_is_sub_tier_not_lexicographic():
    """Round-4: the additive time bonus must NOT dominate relevance — a strongly-relevant
    out-of-window turn outranks a weakly-relevant in-window one at the shipped default bonus."""
    user = "u"
    in_window = EpisodeHit(
        episode=Episode(user_id=user, content="weak but recent",
                        occurred_at=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
                        context={"source": "chat"}),
        cosine_distance=0.1, sparse_rank=None, rrf_score=1.0 / (SETTINGS.rrf_k + 30),
    )
    out_window = EpisodeHit(
        episode=Episode(user_id=user, content="strong but old",
                        occurred_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                        context={"source": "chat"}),
        cosine_distance=0.1, sparse_rank=None, rrf_score=1.0 / (SETTINGS.rrf_k + 1),
    )
    spec = QuerySpec(time_range=(date(2026, 7, 10), date(2026, 7, 10)))
    ranked = _rerank_dialogue([in_window, out_window], spec, SETTINGS, "Asia/Kolkata")
    assert ranked[0] is out_window  # relevance still wins; the in-window boost can't override it


def test_soft_window_fails_open_no_confident_denial(db_session):
    """A time cue with NOTHING in the window must still return the relevant excerpt (soft filter),
    never an empty set that becomes a confident false denial."""
    user = f"r-{uuid.uuid4()}"
    ep = _chat_ep(db_session, user, "Months ago we covered TanStack Query basics",
                  when=datetime(2026, 3, 1, 12, 0, tzinfo=UTC))
    # ask "yesterday" (a window that excludes the March turn) — it must still surface
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    llm = _decompose(entities=["TanStack Query"], semantic_query="TanStack Query",
                     time_range={"start": yesterday, "end": yesterday})
    bundle = recall(db_session, llm, SETTINGS, user_id=user, message="did we talk about TanStack yesterday?")
    assert str(ep.id) in [d.episode_id for d in bundle.dialogue]
