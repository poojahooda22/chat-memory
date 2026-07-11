"""Hybrid retrieval — the OmniQuery-style read path.

Plain cosine similarity answers "what do you know about my dog", but silently fails a
date / person / place question: a year isn't a topic, so similarity can't filter by it, and a
fixed top-k drops photos an aggregate ("how many") needs. So we DECOMPOSE the question into
structured filters (entity, time range, place), FILTER the store exactly (a real WHERE — all
matches, not a top-k), then RANK what's left by similarity. A general question with no filters
falls back to plain cosine so we never over-filter.

(OmniQuery also infers multi-photo "events" and a habit layer — those are a later move. This is
the decompose → filter → rank core, which is the actual correctness fix.)
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, func, select

from app.config import Settings
from app.memory.embeddings import embed_text
from app.memory.lineage import predecessors_for
from app.memory.pipeline import (
    EpisodeHit,
    search_episodes,
    search_image_episodes,
    search_memories,
)
from app.memory.prompts import build_decompose_messages, parse_json
from app.models import Entity, Episode, EpisodeEntity, Memory

FILTERED_CAP = 50   # an aggregate ("all" / "how many") needs the full filtered set, not a top-k
RANKED_TOP_K = 8    # for a semantic question inside a filter, how many to rank in
FACT_TOP_K = 5      # distilled facts injected alongside, as today


def _local_date(dt: datetime, tz: str) -> date:
    """The calendar date of an instant in the user's timezone (episodes are stored aware-UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ZoneInfo(tz)).date()


def _escape_excerpt(text: str, max_chars: int) -> str:
    """Flatten a stored turn into a single safe prompt line: strip control chars/newlines (so a
    crafted excerpt can't forge structure) and truncate. Best-effort — not a trust boundary."""
    cleaned = re.sub(r"[\x00-\x1f]", " ", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return cleaned


@dataclass
class QuerySpec:
    entities: list[str] = field(default_factory=list)
    time_range: tuple[date, date] | None = None
    place: str | None = None
    semantic_query: str = ""
    wants_all: bool = False  # a count / "all" / "every" question → return the whole filtered set

    @property
    def has_filters(self) -> bool:
        return bool(self.entities) or self.time_range is not None or self.place is not None


@dataclass
class RetrievalResult:
    facts: list[Memory]
    photos: list[Episode]
    spec: QuerySpec
    dialogue: list[EpisodeHit] = field(default_factory=list)  # past chat turns, floored + reranked


@dataclass
class RecalledFact:
    """A distilled fact WITH its receipts — the provenance + trust the layer + proactive gate read."""
    memory_id: str
    content: str
    source: str  # chat | photo | quiz | import | inferred | mcp (the P1 origin)
    confidence: float  # 0..1 trust (the P1 score)
    source_episode_ids: list[str]  # which episodes this fact was distilled from
    # belief-revision lineage: was this fact a correction of an earlier one, and of what?
    revised: bool = False  # True if this fact superseded a prior belief
    previously: str | None = None  # the immediate prior belief's text (what it replaced)
    # when the ENGINE recorded the correction (this live row's creation) — NOT the life-event moment
    # the user changed their mind; there is no valid-time column, so never present it as one
    ingested_at: datetime | None = None
    has_older: bool = False  # the prior belief itself superseded an even older one (multi-hop)


@dataclass
class RecalledPhoto:
    """A photo memory with its capture date + place (when known) and its episode id."""
    episode_id: str
    content: str  # the caption
    occurred_at: date
    place: str | None

    @property
    def line(self) -> str:
        where = f", in {self.place}" if self.place else ""
        return f"[captured {self.occurred_at.isoformat()}{where}] {self.content}"


@dataclass
class RecalledExchange:
    """A past chat turn surfaced by cross-conversation recall, display-ready.

    `content` is already escaped + truncated; `local_date` is the turn's date in the user's tz (so
    the [date] cue in .line matches how the user resolved "yesterday"). `cosine_distance` is None
    when the turn was found only by the keyword channel."""
    episode_id: str
    conversation_id: str | None
    role: str
    content: str
    occurred_at: datetime
    local_date: date
    cosine_distance: float | None

    @property
    def line(self) -> str:
        return f"[{self.local_date.isoformat()}] {self.role}: {self.content}"


@dataclass
class RecallBundle:
    """The recall payload: ranked facts + photo memories + past-conversation excerpts, each carrying
    provenance, plus the decomposed query and ONE per-recall confidence. The chat prompt is built
    from .fact_lines / .photo_lines / .dialogue_lines; the extra fields (source, confidence, episode
    ids, evidence) are what the MCP `recall` tool and the proactive WHEN-gate consume."""
    facts: list[RecalledFact]
    photos: list[RecalledPhoto]
    spec: QuerySpec
    confidence: float  # v1: mean confidence of the recalled facts, 0 when none were found
    dialogue: list[RecalledExchange] = field(default_factory=list)
    # what the answer can be grounded in: distilled facts, past-conversation excerpts, or nothing.
    # 'dialogue' with confidence 0 is NOT "nothing known" — the answer is in the excerpts.
    evidence: str = "none"  # facts | dialogue | none

    @property
    def fact_lines(self) -> list[str]:
        return [f.content for f in self.facts]

    @property
    def photo_lines(self) -> list[str]:
        return [p.line for p in self.photos]

    @property
    def dialogue_lines(self) -> list[str]:
        return [d.line for d in self.dialogue]

    @property
    def dialogue_present(self) -> bool:
        return bool(self.dialogue)


def _chat_json(client, model: str, messages: list[dict]) -> dict:
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return parse_json(response.choices[0].message.content)


def decompose_query(
    client, settings: Settings, message: str, *, now_local: date | None = None
) -> QuerySpec:
    """One LLM call → the structured filters a question implies (entity / time / place).

    `now_local` is today's date in the USER's timezone — so "yesterday" resolves to the user's
    calendar day, not the server's UTC day (a 5.5h gap for an IST user, enough to mis-date a
    late-night chat). Falls back to the UTC date only when no tz context is supplied.
    """
    today = (now_local or datetime.now(UTC).date()).isoformat()
    raw = _chat_json(client, settings.llm_model, build_decompose_messages(message, today))

    time_range: tuple[date, date] | None = None
    tr = raw.get("time_range")
    if isinstance(tr, dict) and tr.get("start") and tr.get("end"):
        try:
            time_range = (date.fromisoformat(tr["start"]), date.fromisoformat(tr["end"]))
        except (ValueError, TypeError):
            time_range = None

    entities = [e.strip() for e in (raw.get("entities") or []) if isinstance(e, str) and e.strip()]
    place = raw.get("place")
    return QuerySpec(
        entities=entities,
        time_range=time_range,
        place=place.strip() if isinstance(place, str) and place.strip() else None,
        semantic_query=str(raw.get("semantic_query") or message),
        wants_all=bool(raw.get("wants_all")),
    )


def _entity_episode_ids(session: Session, user_id: str, names: list[str]) -> set:
    """Episode ids linked to any of the named entities (case-insensitive). Empty set = the user
    named an entity we don't know about."""
    lowered = [n.lower() for n in names]
    rows = session.exec(
        select(EpisodeEntity.episode_id)
        .join(Entity, col(EpisodeEntity.entity_id) == col(Entity.id))
        .where(Entity.user_id == user_id, func.lower(Entity.name).in_(lowered))
    ).all()
    return set(rows)


def _filtered_photos(
    session: Session, client, settings: Settings, *, user_id: str, spec: QuerySpec
) -> list[Episode]:
    stmt = select(Episode).where(
        Episode.user_id == user_id,
        col(Episode.context)["source"].astext == "image",  # pyrefly: ignore[missing-attribute]
    )
    if spec.time_range:
        start = datetime.combine(spec.time_range[0], datetime.min.time(), tzinfo=UTC)
        end = datetime.combine(spec.time_range[1], datetime.max.time(), tzinfo=UTC)
        stmt = stmt.where(Episode.occurred_at >= start, Episode.occurred_at <= end)
    if spec.place:
        # pyrefly: ignore[missing-attribute]  — JSONB path -> text, case-insensitive contains
        stmt = stmt.where(col(Episode.context)["place"]["name"].astext.ilike(f"%{spec.place}%"))
    if spec.entities:
        ids = _entity_episode_ids(session, user_id, spec.entities)
        if not ids:
            return []  # asked about a named entity we don't know → honestly nothing, not a guess
        stmt = stmt.where(col(Episode.id).in_(ids))

    if spec.wants_all:
        # aggregate: the COMPLETE filtered set (capped for safety), newest first
        stmt = stmt.order_by(col(Episode.occurred_at).desc()).limit(FILTERED_CAP)
        return list(session.exec(stmt).all())

    # semantic: rank the filtered set by similarity to the rewritten query
    embedding = embed_text(client, settings.embedding_model, spec.semantic_query)
    stmt = (
        stmt.where(col(Episode.embedding).is_not(None))
        # pyrefly: ignore[missing-attribute]  — pgvector comparator missing from stubs
        .order_by(col(Episode.embedding).cosine_distance(embedding))
        .limit(RANKED_TOP_K)
    )
    return list(session.exec(stmt).all())


def _in_window(occurred_at: datetime, time_range: tuple[date, date] | None, tz: str) -> bool:
    """Is a turn inside the question's (soft, ±1 day) time window, in the user's local calendar?"""
    if time_range is None:
        return False
    d = _local_date(occurred_at, tz)
    return time_range[0] - timedelta(days=1) <= d <= time_range[1] + timedelta(days=1)


def _rerank_dialogue(
    hits: list[EpisodeHit], spec: QuerySpec, settings: Settings, tz: str
) -> list[EpisodeHit]:
    """Apply the relevance floor, then the SOFT time boost, then take the top-k.

    Floor: keep a turn if it is dense-close OR a keyword hit — so a rare proper noun caught only by
    keyword (dense-far) survives, which is the whole incident case. Time is a small ADDITIVE bonus
    (never a hard filter), so a strongly-relevant out-of-window turn can still outrank a weak
    in-window one, and an empty in-window set never produces a confident false denial."""
    kept: list[EpisodeHit] = []
    for h in hits:
        keyword_hit = h.sparse_rank is not None
        dense_close = (
            h.cosine_distance is not None and h.cosine_distance <= settings.dialogue_max_distance
        )
        if not (keyword_hit or dense_close):
            continue
        in_window = _in_window(h.episode.occurred_at, spec.time_range, tz)
        h.final_score = h.rrf_score + (settings.dialogue_window_bonus if in_window else 0.0)
        kept.append(h)
    kept.sort(key=lambda h: (h.final_score, str(h.episode.id)), reverse=True)
    return kept[: settings.dialogue_top_k]


def retrieve(
    session: Session, client, settings: Settings, *, user_id: str, message: str,
    exclude_episode_ids: Sequence = (), user_tz: str | None = None,
) -> RetrievalResult:
    """Decompose the question, filter the photo store exactly when it names an entity/time/place,
    else fall back to plain cosine; search facts by similarity; AND search PAST chat episodes
    (hybrid keyword+dense) so cross-conversation questions ("did we talk about X?") are answerable.
    `exclude_episode_ids` are the current conversation's recent turns already injected verbatim."""
    tz = user_tz or settings.user_tz
    now_local = datetime.now(ZoneInfo(tz)).date()
    spec = decompose_query(client, settings, message, now_local=now_local)
    # embed the raw question ONCE and reuse it for facts + the filterless image path
    query_vec = embed_text(client, settings.embedding_model, message)
    facts = search_memories(
        session, client, settings, user_id=user_id, query=message, limit=FACT_TOP_K,
        embedding=query_vec,
    )
    if spec.has_filters:
        photos = _filtered_photos(session, client, settings, user_id=user_id, spec=spec)
    else:
        photos = search_image_episodes(
            session, client, settings, user_id=user_id, query=message, limit=3,
            embedding=query_vec,
        )

    # PAST CHAT EPISODES — the cross-conversation dialogue read path. Dense channel ranks by the
    # decompose rewrite (reuse query_vec when it equals the raw message); keyword channel gets the
    # RAW message + entities + rewrite (raw terms preserve the proper noun a rewrite can drift).
    semantic_query = spec.semantic_query or message
    dialogue_vec = (
        query_vec if semantic_query == message
        else embed_text(client, settings.embedding_model, semantic_query)
    )
    dialogue_hits = search_episodes(
        session, client, settings, user_id=user_id, source="chat",
        semantic_query=semantic_query,
        keyword_sources=[message, *spec.entities, semantic_query],
        embedding=dialogue_vec, exclude_episode_ids=exclude_episode_ids,
    )
    dialogue = _rerank_dialogue(dialogue_hits, spec, settings, tz)
    return RetrievalResult(facts=facts, photos=photos, spec=spec, dialogue=dialogue)


def recall(
    session: Session, client, settings: Settings, *, user_id: str, message: str,
    exclude_episode_ids: Sequence = (), user_tz: str | None = None,
) -> RecallBundle:
    """The public read path: run hybrid retrieval, then wrap every result with its receipts.

    Reshapes retrieve()'s raw ORM rows into a provenance-carrying bundle. The chat prompt uses
    .fact_lines / .photo_lines / .dialogue_lines; the extra per-item fields (source, confidence,
    episode ids, evidence) are what the MCP layer and the proactive gate read.
    """
    tz = user_tz or settings.user_tz
    raw = retrieve(
        session, client, settings, user_id=user_id, message=message,
        exclude_episode_ids=exclude_episode_ids, user_tz=tz,
    )
    # one batched lineage lookup for ALL recalled facts (no per-fact N+1). ingested_at is the LIVE
    # row's own created_at (when the engine recorded the correction), never the predecessor's birth.
    lineage = predecessors_for(session, [m.id for m in raw.facts])
    facts = [
        RecalledFact(
            memory_id=str(m.id),
            content=m.content,
            source=m.source,
            confidence=m.confidence,
            source_episode_ids=list(m.source_episode_ids),
            revised=m.id in lineage,
            previously=lineage[m.id].content if m.id in lineage else None,
            ingested_at=m.created_at if m.id in lineage else None,
            has_older=lineage[m.id].has_older if m.id in lineage else False,
        )
        for m in raw.facts
    ]
    photos = [
        RecalledPhoto(
            episode_id=str(e.id),
            content=e.content,
            occurred_at=e.occurred_at.date(),
            place=((e.context or {}).get("place") or {}).get("name"),
        )
        for e in raw.photos
    ]
    dialogue = [
        RecalledExchange(
            episode_id=str(h.episode.id),
            conversation_id=h.episode.conversation_id,
            role=h.episode.context.get("role", "user"),
            content=_escape_excerpt(h.episode.content, settings.dialogue_excerpt_chars),
            occurred_at=h.episode.occurred_at,
            local_date=_local_date(h.episode.occurred_at, tz),
            cosine_distance=h.cosine_distance,
        )
        for h in raw.dialogue
    ]
    # v1 per-recall confidence: how much we trust the FACTS we're answering from (photos are
    # grounded episodes and carry no confidence field). 0 when no facts were found. Later this can
    # blend recency + the number of supporting episodes.
    confidence = sum(f.confidence for f in facts) / len(facts) if facts else 0.0
    # what the answer can stand on. dialogue-with-confidence-0 must NOT read as "nothing known".
    evidence = "facts" if facts else ("dialogue" if dialogue else "none")
    return RecallBundle(
        facts=facts, photos=photos, spec=raw.spec, confidence=confidence,
        dialogue=dialogue, evidence=evidence,
    )